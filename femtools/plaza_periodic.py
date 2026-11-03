"""Plaza refinement that knows about periodic faces, in serial and under MPI.

    refine_periodic(mesh, markers, box, axes)   df.refine(mesh, markers), periodic-safe
    TooManyCells                                 raised by max_cells / max_rank_cells

`constrained_domain` pairs dofs by coordinate, so if a face of a periodic mesh is refined
and the face it is identified with is not, the slave dofs with no master are silently left
unconstrained. dolfin's Python `refine` cannot say which edges get bisected, and marking
partner cells from outside does not converge: Plaza's conformity propagation reaches the
other axes from the interior. So the identification is imposed inside the propagation:
this is dolfin 2019.2.0's own `face_long_edge`, `enforce_rules` and `do_refine`,
JIT-compiled through `dolfin.compile_cpp_code`, with the periodic correction folded into
`enforce_rules`:

    for each periodic partner pair (a, b), possibly on different ranks:
        if is_marked(a) != is_marked(b): mark both; ++update_count

so propagation and symmetry reach a joint fixed point. Everything it calls on
`ParallelRefinement` and `PlazaRefinementND::get_simplices` is public API; only
`face_long_edge`, `enforce_rules` and `do_refine` are private, which is why they are
reproduced here. A face's longest edge is chosen by position on ties, the same on both
faces of a pair and on every rank (dolfin breaks ties on global vertex indices, which the
two faces do not share).

Partners come from `PeriodicBoundaryComputation::compute_periodic_pairs`, dolfin's own
cross-rank matching, with `femtools.periodic`'s `CppPeriodicBox` one axis at a time, so the
refinement and the dofmap agree on which points are identified and no call crosses into
Python. It gives the slave side only, so one all_to_all before the propagation tells each
master its partner; every propagation round then exchanges the marks of the pairs. At one
rank that is a local copy: serial and parallel are one code path. The final assembly calls
`ParallelRefinement::partition` on more than one rank, as `PlazaRefinementND::refine` does.

Parent facet markers are dropped; the parent-cell map is returned
(`femtools.hierarchical_transfer` is built on it).

No Python fallback: a compiler and `libdolfin-dev` are needed, and this raises without them.
"""
import dolfin as df

from femtools.periodic import PERIODIC_BOX_SOURCE

_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <vector>
#include <map>
#include <cmath>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/common/Array.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/refinement/ParallelRefinement.h>
#include <dolfin/refinement/PlazaRefinementND.h>
#include <dolfin/common/MPI.h>

namespace py = pybind11;
using namespace dolfin;

// The periodic pairing itself is `femtools.periodic`'s CppPeriodicBox, embedded
// here rather than restated: compute_periodic_pairs calls inside()/map() once per
// exterior mesh entity per axis -- and "exterior" includes every bead-surface edge,
// not just the box faces, so on the real packing that is of order a million calls
// (measured: the call count scales exactly with surface, 9520 -> 38240 for a
// 10^3 -> 20^3 cube). Through a Python-subclassed SubDomain each one crosses the
// pybind boundary; defined here, none of them leave C++. This module uses it one
// axis at a time, with a mask selecting that axis, which is exactly the single-axis
// restriction of the general class the constraint is built with -- so the refinement
// and the dofmap cannot disagree about which points are identified.
""" + PERIODIC_BOX_SOURCE + r"""

// A vertex's position with each upper periodic plane mapped onto the lower one: the
// same for a vertex and its periodic partner, and on every rank -- unlike its global
// index, which the two faces number independently.
std::array<double, 3> periodic_key(const Vertex& v, const std::vector<double>& box,
                                   const std::vector<std::size_t>& axes, double tol)
{
  std::array<double, 3> k = {v.x(0), v.x(1), v.x(2)};
  for (auto a : axes)
    if (std::abs(k[a] - box[a]) < tol) k[a] = 0.0;
  return k;
}

bool key_less(const std::array<double, 3>& p, const std::array<double, 3>& q, double tol)
{
  for (std::size_t d = 0; d < 3; ++d)
    if (std::abs(p[d] - q[d]) > tol) return p[d] < q[d];
  return false;
}

// dolfin's PlazaRefinementND::face_long_edge (private there), with the choice made the
// same on the two faces of a periodic pair. dolfin takes the strictly longest edge and
// breaks an exact tie on the global index of the opposite vertex (edge i is opposite
// vertex i). Both go wrong across a periodic pair, and did on the real packing
// (2026-09-29): an exact tie is broken by indices the two faces do not share, and two
// edges equal to the last bit come out in either order, since partner coordinates agree
// only to ~2e-16 -- so the faces bisected different edges and their triangulations
// drifted apart ("flips"), invisible to a vertex check until a flipped edge was refined.
// Here edges within a relative 1e-10 of the longest are tied, and a tie goes to the
// largest periodic_key of the opposite vertex: a position, so the same on both faces and
// on every rank, which is all dolfin's global index was there for.
void face_long_edge(std::vector<unsigned int>& long_edge, const Mesh& mesh,
                    const std::vector<double>& box, const std::vector<std::size_t>& axes,
                    double tol)
{
  mesh.init(1); mesh.init(2); mesh.init(2, 1);
  long_edge.resize(mesh.num_faces());
  std::vector<double> edge_length(mesh.num_edges());
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_length[e->index()] = e->length();
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    const unsigned int* face_edges = f->entities(1);
    double max_len = 0.0;
    for (unsigned int i = 0; i < 3; ++i)
      max_len = std::max(max_len, edge_length[face_edges[i]]);
    std::size_t imax = 3;
    std::array<double, 3> best = {0.0, 0.0, 0.0};
    for (unsigned int i = 0; i < 3; ++i)
    {
      if (edge_length[face_edges[i]] < max_len * (1.0 - 1e-10)) continue;
      const std::array<double, 3> k
        = periodic_key(Vertex(mesh, f->entities(0)[i]), box, axes, tol);
      if (imax == 3 || key_less(best, k, tol)) { imax = i; best = k; }
    }
    long_edge[f->index()] = face_edges[imax];
  }
}

// For one periodic SubDomain (one axis), the bidirectional local<->remote edge partner
// list: for every local edge that is a slave or a master under this identification,
// (local index, remote rank, remote local index). Appends into the caller's arrays, so
// several axes accumulate into one combined list -- an edge on a domain edge or corner,
// identified by more than one axis at once, ends up in the list once per axis, which is
// right: the marking loop unions over every pair it is given.
//
// PeriodicBoundaryComputation::compute_periodic_pairs gives the slave side only: a
// slave's owning rank learns (master_rank, master_local), but the master's owning rank
// is never told. The second half here is one more all_to_all telling it -- (my_rank,
// slave_local) sent to master_rank -- built once, not per propagation iteration.
void add_periodic_partners(const Mesh& mesh, const SubDomain& sub_domain,
                           std::vector<std::size_t>& partner_local,
                           std::vector<std::size_t>& partner_rank,
                           std::vector<std::size_t>& partner_remote)
{
  const MPI_Comm comm = mesh.mpi_comm();
  const std::size_t num_ranks = dolfin::MPI::size(comm);

  auto pairs = PeriodicBoundaryComputation::compute_periodic_pairs(mesh, sub_domain, 1);

  for (auto& kv : pairs)
  {
    partner_local.push_back(kv.first);
    partner_rank.push_back(kv.second.first);
    partner_remote.push_back(kv.second.second);
  }

  std::vector<std::vector<std::size_t>> send(num_ranks);
  for (auto& kv : pairs)
  {
    const std::size_t master_rank = kv.second.first;
    send[master_rank].push_back(kv.second.second);  // master_local, on master_rank
    send[master_rank].push_back(kv.first);           // slave_local, on this rank
  }
  std::vector<std::vector<std::size_t>> recv;
  dolfin::MPI::all_to_all(comm, send, recv);
  for (std::size_t p = 0; p < num_ranks; ++p)
  {
    const std::vector<std::size_t>& rp = recv[p];
    for (std::size_t i = 0; i + 1 < rp.size(); i += 2)
    {
      partner_local.push_back(rp[i]);       // my local master edge
      partner_rank.push_back(p);            // the rank holding its slave
      partner_remote.push_back(rp[i + 1]);  // the slave's local index there
    }
  }
}

// dolfin's enforce_rules, with the periodic correction generalised from a local pair
// lookup to an MPI exchange over the partner list. At one rank, partner_rank[i] is
// always 0 (this rank), so MPI::all_to_all degenerates to a local copy and the serial
// and parallel cases are one code path.
void enforce_rules_periodic(ParallelRefinement& p_ref, const Mesh& mesh,
                            const std::vector<unsigned int>& long_edge,
                            const std::vector<std::size_t>& partner_local,
                            const std::vector<std::size_t>& partner_rank,
                            const std::vector<std::size_t>& partner_remote)
{
  const MPI_Comm comm = mesh.mpi_comm();
  const std::size_t num_ranks = dolfin::MPI::size(comm);
  std::size_t update_count = 1;
  while (update_count != 0)
  {
    update_count = 0;
    p_ref.update_logical_edgefunction();
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      const std::size_t long_e = long_edge[f->index()];
      if (p_ref.is_marked(long_e)) continue;
      bool any_marked = false;
      for (EdgeIterator e(*f); !e.end(); ++e)
        any_marked |= p_ref.is_marked(e->index());
      if (any_marked) { p_ref.mark(long_e); ++update_count; }
    }

    std::vector<std::vector<std::size_t>> send(num_ranks);
    for (std::size_t i = 0; i < partner_local.size(); ++i)
    {
      send[partner_rank[i]].push_back(partner_remote[i]);
      send[partner_rank[i]].push_back(p_ref.is_marked(partner_local[i]) ? 1 : 0);
    }
    std::vector<std::vector<std::size_t>> recv;
    dolfin::MPI::all_to_all(comm, send, recv);
    for (std::size_t p = 0; p < num_ranks; ++p)
    {
      const std::vector<std::size_t>& rp = recv[p];
      for (std::size_t i = 0; i + 1 < rp.size(); i += 2)
      {
        const std::size_t local_idx = rp[i];
        const bool other_marked = rp[i + 1] != 0;
        if (other_marked != p_ref.is_marked(local_idx))
        {
          p_ref.mark(local_idx);
          ++update_count;
        }
      }
    }

    update_count = dolfin::MPI::sum(comm, update_count);
  }
}

// The edges no refinement may bisect, from `seeds` (a slab's inlet edges, say),
// closed under the longest-edge rule: a face whose long edge is frozen freezes
// its other two, since bisecting either would make enforce_rules bisect the long edge as
// well. Closed under shared edges between ranks (a second ParallelRefinement is the
// storage, its update_logical_edgefunction the exchange) and under periodic partners,
// as enforce_rules is, to a joint fixed point.
void freeze_closure(ParallelRefinement& f_ref, const Mesh& mesh,
                    const std::vector<std::size_t>& seeds,
                    const std::vector<unsigned int>& long_edge,
                    const std::vector<std::size_t>& partner_local,
                    const std::vector<std::size_t>& partner_rank,
                    const std::vector<std::size_t>& partner_remote)
{
  const MPI_Comm comm = mesh.mpi_comm();
  const std::size_t num_ranks = dolfin::MPI::size(comm);
  for (auto e : seeds) f_ref.mark(e);
  std::size_t update_count = 1;
  while (update_count != 0)
  {
    update_count = 0;
    f_ref.update_logical_edgefunction();
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (!f_ref.is_marked(long_edge[f->index()])) continue;
      for (EdgeIterator e(*f); !e.end(); ++e)
        if (!f_ref.is_marked(e->index())) { f_ref.mark(e->index()); ++update_count; }
    }
    std::vector<std::vector<std::size_t>> send(num_ranks);
    for (std::size_t i = 0; i < partner_local.size(); ++i)
      if (f_ref.is_marked(partner_local[i]))
        send[partner_rank[i]].push_back(partner_remote[i]);
    std::vector<std::vector<std::size_t>> recv;
    dolfin::MPI::all_to_all(comm, send, recv);
    for (std::size_t p = 0; p < num_ranks; ++p)
      for (auto e : recv[p])
        if (!f_ref.is_marked(e)) { f_ref.mark(e); ++update_count; }
    update_count = dolfin::MPI::sum(comm, update_count);
  }
}

// dolfin's do_refine, minus parent-cell/parent-facet bookkeeping, dispatching to the
// parallel assembly (redistribute) rather than build_local when running on more than
// one rank -- exactly what unmodified PlazaRefinementND::refine does.
void do_refine(Mesh& new_mesh, const Mesh& mesh, ParallelRefinement& p_ref,
               const std::vector<unsigned int>& long_edge, bool redistribute,
               std::vector<std::size_t>& parent_cell)
{
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cell_edges = tdim*3 - 3;
  const std::size_t num_cell_vertices = tdim + 1;

  p_ref.create_new_vertices();
  const std::map<std::size_t, std::size_t>& new_vertex_map
    = *(p_ref.edge_to_new_vertex());

  std::vector<std::size_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<std::size_t> marked_edge_list;
  std::vector<std::size_t> simplex_set;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t j = 0;
    for (VertexIterator v(*cell); !v.end(); ++v) indices[j++] = v->global_index();

    marked_edge_list = p_ref.marked_edge_list(*cell);
    if (marked_edge_list.size() == 0)
    {
      p_ref.new_cell(*cell);
      parent_cell.push_back(cell->index());
    }
    else
    {
      indices.resize(num_cell_vertices + num_cell_edges);
      std::vector<bool> markers(num_cell_edges, false);
      for (auto &p : marked_edge_list)
      {
        markers[p] = true;
        const std::size_t edge_index = cell->entities(1)[p];
        auto it = new_vertex_map.find(edge_index);
        indices[num_cell_vertices + p] = it->second;
      }
      std::vector<std::size_t> longest_edge;
      for (FaceIterator f(*cell); !f.end(); ++f)
        longest_edge.push_back(long_edge[f->index()]);
      for (auto &p : longest_edge)
        for (EdgeIterator ej(*cell); !ej.end(); ++ej)
          if (p == ej->index()) { p = ej.pos(); break; }

      PlazaRefinementND::get_simplices(simplex_set, markers, longest_edge, tdim, false);
      const std::size_t ncells = simplex_set.size() / num_cell_vertices;
      for (std::size_t i = 0; i != ncells; ++i)
        parent_cell.push_back(cell->index());
      for (auto &it : simplex_set) it = indices[it];
      p_ref.new_cells(simplex_set);
    }
  }

  const bool serial = (dolfin::MPI::size(mesh.mpi_comm()) == 1);
  if (serial)
    p_ref.build_local(new_mesh);
  else
    p_ref.partition(new_mesh, redistribute);

  // The parent map is meaningful under exactly dolfin's own condition -- its
  // do_refine stores "parent_cell" into new_mesh.data() when serial or not
  // redistributed, because redistribution scatters the new cells across ranks and
  // the local ordering built above no longer describes them. Cleared rather than
  // returned wrong.
  if (!serial && redistribute)
    parent_cell.clear();
}

// The cells do_refine would make, from the edges enforce_rules left marked -- its own
// loop, emitting nothing -- summed over ranks and on the most loaded rank (which sets a
// job's peak). Cheap beside the refinement it predicts (no vertices, no cells, no
// partition), so a refinement too large for what follows it can be refused before it
// is paid for.
std::pair<std::size_t, std::size_t> count_children(const Mesh& mesh, ParallelRefinement& p_ref,
                                                   const std::vector<unsigned int>& long_edge)
{
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cell_edges = tdim*3 - 3;
  const std::size_t num_cell_vertices = tdim + 1;
  std::vector<std::size_t> simplex_set;
  std::size_t n = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<std::size_t> marked_edge_list = p_ref.marked_edge_list(*cell);
    if (marked_edge_list.size() == 0) { ++n; continue; }
    std::vector<bool> markers(num_cell_edges, false);
    for (auto &p : marked_edge_list) markers[p] = true;
    std::vector<std::size_t> longest_edge;
    for (FaceIterator f(*cell); !f.end(); ++f)
      longest_edge.push_back(long_edge[f->index()]);
    for (auto &p : longest_edge)
      for (EdgeIterator ej(*cell); !ej.end(); ++ej)
        if (p == ej->index()) { p = ej.pos(); break; }
    PlazaRefinementND::get_simplices(simplex_set, markers, longest_edge, tdim, false);
    n += simplex_set.size() / num_cell_vertices;
  }
  return std::make_pair(dolfin::MPI::sum(mesh.mpi_comm(), n),
                        dolfin::MPI::max(mesh.mpi_comm(), n));
}

// max_cells > 0 or max_rank_cells > 0: count the children first, and if there would be
// more than that in all or on some rank, return (npair, no parent map, the count, the
// most loaded rank's count, ...) with new_mesh left empty. Both counts are reduced, so
// every rank takes the same branch.
//
// `frozen` (local edge indices, may be empty): edges never to bisect. Their closure is
// taken (freeze_closure), and a marked cell holding a frozen edge is left unmarked --
// which is enough: every edge enforce_rules marks is then a face's long edge in a face
// holding a marked, unfrozen edge, and a face with a frozen long edge has only frozen
// edges. Checked anyway, and thrown on. The tuple's last two entries are the cells so
// unmarked and the cells holding a frozen edge, both summed over ranks.
py::tuple refine_periodic(Mesh& new_mesh, const Mesh& mesh,
                          const MeshFunction<bool>& marker,
                          const std::vector<std::size_t>& axes,
                          const std::vector<double>& box,
                          double tol, bool redistribute, std::size_t max_cells,
                          std::size_t max_rank_cells,
                          const std::vector<std::size_t>& frozen)
{
  std::vector<unsigned int> long_edge;
  face_long_edge(long_edge, mesh, box, axes, tol);

  std::vector<std::size_t> partner_local, partner_rank, partner_remote;
  for (auto axis : axes)
  {
    std::vector<bool> mask(3, false);
    mask[axis] = true;
    CppPeriodicBox sub(std::vector<double>(3, 0.0), box, mask, tol, tol);
    add_periodic_partners(mesh, sub, partner_local, partner_rank, partner_remote);
  }

  ParallelRefinement p_ref(mesh);
  // the freeze is collective: every rank takes this branch or none does
  const bool freezing = dolfin::MPI::max(mesh.mpi_comm(), frozen.size()) > 0;
  std::size_t n_unmarked = 0, n_frozen_cells = 0;
  if (!freezing)
    p_ref.mark(marker);
  else
  {
    ParallelRefinement f_ref(mesh);
    freeze_closure(f_ref, mesh, frozen, long_edge, partner_local, partner_rank,
                   partner_remote);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      bool holds = false;
      for (EdgeIterator e(*c); !e.end(); ++e) holds |= f_ref.is_marked(e->index());
      n_frozen_cells += holds;
      if (!marker[*c]) continue;
      if (holds) { ++n_unmarked; continue; }
      p_ref.mark(*c);
    }
    enforce_rules_periodic(p_ref, mesh, long_edge, partner_local, partner_rank,
                           partner_remote);
    std::size_t bad = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      bad += f_ref.is_marked(e->index()) && p_ref.is_marked(e->index());
    if (dolfin::MPI::sum(mesh.mpi_comm(), bad) > 0)
      throw std::runtime_error("refine_periodic: the marking reached a frozen edge");
    n_unmarked = dolfin::MPI::sum(mesh.mpi_comm(), n_unmarked);
    n_frozen_cells = dolfin::MPI::sum(mesh.mpi_comm(), n_frozen_cells);
  }
  if (!freezing)
    enforce_rules_periodic(p_ref, mesh, long_edge, partner_local, partner_rank,
                           partner_remote);
  std::vector<std::size_t> parent_cell;
  std::size_t n_children = 0, n_rank = 0;
  if (max_cells > 0 || max_rank_cells > 0)
  {
    const auto counts = count_children(mesh, p_ref, long_edge);
    n_children = counts.first;
    n_rank = counts.second;
    if ((max_cells > 0 && n_children > max_cells)
        || (max_rank_cells > 0 && n_rank > max_rank_cells))
      return py::make_tuple(partner_local.size(), parent_cell, n_children, n_rank,
                            n_unmarked, n_frozen_cells);
  }
  do_refine(new_mesh, mesh, p_ref, long_edge, redistribute, parent_cell);
  // as a NumPy array over the vector itself, moved to the heap and freed with it: a
  // list of Python ints was ~40 B a new cell at the refinement's peak
  auto* held = new std::vector<std::size_t>(std::move(parent_cell));
  py::capsule free_it(held, [](void* p) { delete static_cast<std::vector<std::size_t>*>(p); });
  py::array_t<std::size_t> parents(held->size(), held->data(), free_it);
  return py::make_tuple(partner_local.size(), parents, n_children, n_rank,
                        n_unmarked, n_frozen_cells);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("refine_periodic", &refine_periodic);
}
"""

_MODULE = [None]


def _cpp():
    """The compiled extension, built once; raises where it will not build."""
    if _MODULE[0] is None:
        _MODULE[0] = df.compile_cpp_code(_CODE)
    return _MODULE[0]


class TooManyCells(Exception):
    """The refinement would have made more cells than `max_cells`, or more on one rank
    than `max_rank_cells`; nothing was refined.

    Raised on every rank alike (both counts are reduced), before any vertex or cell of the
    new mesh exists -- so a caller can stop for the price of the marking."""

    def __init__(self, n_cells, max_cells, n_rank=0, max_rank_cells=0):
        self.n_cells, self.max_cells = int(n_cells), int(max_cells or 0)
        self.n_rank, self.max_rank_cells = int(n_rank), int(max_rank_cells or 0)
        if self.max_cells and self.n_cells > self.max_cells:
            what = "{} cells, over the {} allowed".format(self.n_cells, self.max_cells)
        else:
            what = "{} cells on the most loaded rank, over the {} allowed a rank".format(
                self.n_rank, self.max_rank_cells)
        super().__init__("the refinement would make {}; nothing was refined".format(what))


def refine_periodic(mesh, markers, box, axes, redistribute=True, tol=1e-10,
                    max_cells=None, max_rank_cells=None, frozen=None, report=None):
    """`df.refine(mesh, markers)`, with the identified faces bisected alike.

    `box` is the periodic box's upper corner (the lower is the origin), `axes` the periodic
    axes. Works at any rank count. `redistribute=True` matches `df.refine`'s default; a
    pass that grows the mesh a few percent is much faster without the repartition, and
    gives the same mesh.

    Returns `(new_mesh, npair, parent_cell)`. `parent_cell` maps each new local cell to
    the old local cell it was cut from -- the map dolfin's own `do_refine` stores as
    "parent_cell", valid under dolfin's own condition: serial, or parallel without
    redistribution. After a redistribution the new cells are scattered across ranks and
    the map is meaningless, so it comes back as `None` on every rank.

    `max_cells`: count the children once the marking has propagated (exactly: do_refine's
    own loop, emitting nothing) and raise `TooManyCells` instead of refining when there
    would be more. `max_rank_cells` likewise for the most loaded rank's children, which
    is what sets a job's peak.

    `frozen`: local edge indices never to bisect, closed under the longest-edge rule,
    shared edges and periodic partners; a marked cell holding a frozen edge is left
    unmarked, and the refinement then never reaches one (checked, and raised on).
    Collective: give it on every rank, empty where a rank holds none. `report`, a dict,
    receives `unmarked` (the marked cells so left) and `frozen_cells` (the cells holding
    a frozen edge), summed over ranks.
    """
    import numpy as np
    new_mesh = df.Mesh()
    cap, rank_cap = int(max_cells or 0), int(max_rank_cells or 0)
    if cap < 0 or rank_cap < 0:
        raise ValueError("max_cells and max_rank_cells must be positive, got {} and {}"
                         .format(max_cells, max_rank_cells))
    npair, parent, n_children, n_rank, n_unmarked, n_frozen = _cpp().refine_periodic(
        new_mesh, mesh, markers, [int(a) for a in axes], [float(b) for b in box], tol,
        redistribute, cap, rank_cap,
        [] if frozen is None else [int(e) for e in np.asarray(frozen).ravel()])
    if report is not None:
        report.update(unmarked=int(n_unmarked), frozen_cells=int(n_frozen))
    if (cap and n_children > cap) or (rank_cap and n_rank > rank_cap):
        raise TooManyCells(n_children, cap, n_rank, rank_cap)
    # Whether the map survives is decided by `redistribute` and the rank count, both
    # of which every rank agrees on -- never by whether *this* rank's map came back
    # empty. A rank that owns no cells has an empty map and a perfectly valid one,
    # and testing emptiness would hand it None while its peers got arrays: callers
    # branch on this to choose between two collective paths, so a per-rank answer
    # deadlocks rather than failing. This mirrors do_refine's own clearing condition.
    if df.MPI.size(mesh.mpi_comm()) > 1 and redistribute:
        return new_mesh, npair, None
    parent_cell = np.asarray(parent, dtype=np.int64)
    if len(parent_cell) != new_mesh.num_cells():
        raise RuntimeError(
            "the parent map has {} entries for {} new cells; do_refine's clearing "
            "condition and this one have drifted apart".format(
                len(parent_cell), new_mesh.num_cells()))
    return new_mesh, npair, parent_cell
