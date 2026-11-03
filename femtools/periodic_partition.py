"""A mesh partition dolfin 2019's parallel periodic dofmap cannot get wrong, and the
check that it did not.

**The defect.** `PeriodicBoundaryComputation::
compute_periodic_pairs` finds slave entities only among the global-exterior facets a
rank holds locally, and `DofMapBuilder::build_constrained_vertex_indices` gives a slave
its master's index only on the ranks that found it -- the step that would tell the
other ranks sharing it is a comment with no code under it. A rank holding a vertex (or
an edge) of an upper periodic face only through cells that touch the face at a point
or along an edge never learns it is a slave: it numbers it as an ordinary shared vertex
(an extra, unconstrained dof) or keeps -1 plus its offset (**another vertex's dof**).
Most face vertices have such cells, on any tetrahedral mesh, and on a triangle mesh a
vertex of a periodic edge does too, so whether it fires is up to where the partition
happens to cut. Everything here takes triangle and tetrahedron meshes.

The box is given by its upper corner `L`, which is all the repair needs (it looks at
the upper faces); the check also needs the lower corner, `lower`, the origin unless
given.

**The fix is the partition.** `read_mesh` reads the mesh in blocks and partitions it
with ParMETIS, and `repaired` moves any rank's cells at an endangered slave entity to a
rank that holds an exterior facet there -- a handful of cells per partition. Only the
cells touching an upper face go to rank 0 to decide that: every facet that matters
contains an upper-face vertex, so all of its cells are among them, which makes the
restriction exact and small. When nothing moves -- most partitions --
the mesh is used as read. Otherwise it is built again in memory with those cells on
their new ranks (`rebuild`: dolfin's `build_distributed_mesh` from the mesh's own
arrays, as it builds a mesh it reads). No rank reads or holds the whole mesh.

**The check** (`check_pairing`) runs after every space is built: every face node of the
mixed space -- vertices and edge midpoints, velocity and pressure -- must carry the one
global dof of its periodic class. It raises otherwise; nothing here falls back.
"""
import os

import numpy as np

EDGES = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))     # dolfin's local edge order
TRIANGLE_CELL_EDGES = ((1, 2), (0, 2), (0, 1))     # ... and a triangle's

#: "on a face" for the repair and the check when the caller gives no tolerance: this
#: times the box's largest side. A caller whose constraint has its own tolerance passes
#: that, so the three judge the same nodes alike
FACE_RTOL = 1e-8


def _face_tol(L, tol):
    return FACE_RTOL * float(np.max(L)) if tol is None else float(tol)


def _as3(v, fill=0.0):
    """An axis-wise sequence (L, lower, periodic) padded to three axes with `fill`."""
    v = list(v)
    return v + [fill] * (3 - len(v))


def _cell_edges(nv):
    """(the edges of a cell of `nv` vertices that can be slaves, the edges of a facet) in
    dolfin's local numbering. On a triangle mesh an edge on a periodic face is a facet,
    held with the cell that has it, so only vertices can be slaves there."""
    return (EDGES, TRIANGLE_EDGES) if nv == 4 else ((), ())


def upper_mask(x, L, periodic, tol):
    """Per vertex, bit `a` set when it lies on the upper face of periodic axis `a`."""
    m = np.zeros(len(x), np.uint8)
    for a in range(min(np.shape(x)[1], len(periodic))):
        if periodic[a]:
            m |= (np.abs(x[:, a] - L[a]) < tol).astype(np.uint8) << a
    return m


TRIANGLE_EDGES = ((0, 1), (0, 2), (1, 2))


def _held(topo, owner, mask, edges):
    """The slave entities each element of `topo` holds, with its rank, distinct, as rows
    (kind, a, b, rank): a vertex (0, v, -1) on an upper face, an edge (1, lo, hi) whose
    ends lie on one upper face."""
    rows = []
    for c in range(topo.shape[1]):
        m = mask[topo[:, c]] > 0
        n = int(m.sum())
        rows.append(np.stack([np.zeros(n, np.int64), topo[m, c],
                              np.full(n, -1, np.int64), owner[m]], 1))
    for a, b in edges:
        m = (mask[topo[:, a]] & mask[topo[:, b]]) > 0
        va, vb = topo[m, a], topo[m, b]
        rows.append(np.stack([np.ones(len(va), np.int64), np.minimum(va, vb),
                              np.maximum(va, vb), owner[m]], 1))
    return _unique_rows(np.concatenate(rows).astype(np.int64))


def _unique_rows(rows):
    """`np.unique(rows, axis=0)` for int rows, the same rows in the same order, many times
    faster than numpy's sort of the rows as one structured value each (the serial repair's
    largest cost): the columns packed, each from its minimum, into one int64 that sorts in
    the rows' order when their ranges fit 63 bits (vertex numbers, ranks and the kind do),
    else a lexsort and a neighbour comparison."""
    if not len(rows):
        return rows.reshape(0, rows.shape[1])
    lo = rows.min(axis=0)
    bits = [int(w).bit_length() for w in rows.max(axis=0) - lo]
    if sum(bits) <= 63:
        key = np.zeros(len(rows), np.int64)
        for c, b in enumerate(bits):
            key = (key << b) | (rows[:, c] - lo[c])
        key = np.unique(key)
        out = np.empty((len(key), rows.shape[1]), rows.dtype)
        for c in range(rows.shape[1] - 1, -1, -1):
            out[:, c] = (key & ((1 << bits[c]) - 1)) + lo[c]
            key = key >> bits[c]
        return out
    s = rows[np.lexsort(rows.T[::-1])]
    return s[np.r_[True, np.any(s[1:] != s[:-1], axis=1)]]


def _find_rows(a, b):
    """For each row of `a`, the index of the equal row of `b` (unique rows), or -1."""
    if not len(a) or not len(b):
        return np.full(len(a), -1, np.int64)
    both = np.concatenate([b, a])
    order = np.lexsort(both.T[::-1])
    s = both[order]
    first = np.r_[True, np.any(s[1:] != s[:-1], axis=1)]
    group = np.cumsum(first) - 1
    head = np.full(group[-1] + 1, -1, np.int64)
    from_b = order < len(b)
    head[group[from_b]] = order[from_b]
    out = np.empty(len(both), np.int64)
    out[order] = head[group]
    return out[len(b):]


def _endangered(mask, topo, owner):
    """`endangered` on vertex masks (`upper_mask`) rather than coordinates: `(bad,
    (keys, low))`, `bad` the rows (kind, a, b, rank) of slave entities a rank holds
    without an exterior facet there, `keys` every entity some rank has such a facet at
    and `low` the lowest of those ranks. Arrays throughout."""
    nv = topo.shape[1]
    faces = np.sort(np.stack([np.delete(topo, k, axis=1) for k in range(nv)], 1), axis=2)
    fcell = np.repeat(np.arange(len(topo)), nv)
    ff = faces.reshape(-1, nv - 1)
    order = np.lexsort(ff.T[::-1])
    fs = ff[order]
    dup = np.all(fs[1:] == fs[:-1], axis=1)
    single = np.ones(len(fs), bool)
    single[1:] &= ~dup
    single[:-1] &= ~dup
    ext = order[single]
    ext = ext[(mask[ff[ext]] > 0).any(1)]   # only facets at an upper-face vertex matter
    cell_edges, facet_edges = _cell_edges(nv)
    held = _held(topo, owner, mask, cell_edges)
    seen = _held(ff[ext], owner[fcell[ext]], mask, facet_edges)
    bad = held[_find_rows(held, seen) < 0]
    # seen is sorted: the first row of each entity has its lowest rank
    if len(seen):
        first = np.r_[True, np.any(seen[1:, :3] != seen[:-1, :3], axis=1)]
        return bad, (seen[first, :3], seen[first, 3])
    return bad, (seen[:, :3], seen[:, 3])


def endangered(x, topo, owner, L, periodic, tol):
    """Slave entities some rank holds without an exterior facet containing them.

    `topo` and `owner` may be any set of cells containing every cell that touches an
    upper periodic face. Returns `(bad, facets)`: `bad` the rows (kind, a, b, rank) --
    kind 0 a vertex a, kind 1 the edge (a, b) -- and `facets` as `_endangered`'s.
    """
    return _endangered(upper_mask(x, L, periodic, tol), np.asarray(topo, np.int64),
                       np.asarray(owner, np.int64))


def _repair(mask, topo, owner, max_rounds=50):
    """`repair` on vertex masks: each cell at an endangered entity goes to the lowest rank
    holding an exterior facet there, round after round; cells found through a
    vertex-to-cells map built once."""
    topo = np.asarray(topo, np.int64)
    owner = np.asarray(owner, np.int64).copy()
    flat = topo.ravel()
    by_vertex = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[by_vertex], np.arange(len(mask) + 1))

    def cells_of(v):
        return by_vertex[starts[v]:starts[v + 1]] // topo.shape[1]

    moved = 0
    for _ in range(max_rounds):
        bad, (keys, low) = _endangered(mask, topo, owner)
        if not len(bad):
            return owner, moved
        at = _find_rows(bad[:, :3], keys)
        if (at < 0).any():
            k = bad[at < 0][0]
            raise RuntimeError('a slave {} {} has no exterior facet on any rank'.format(
                'vertex' if k[0] == 0 else 'edge', int(k[1]) if k[0] == 0
                else (int(k[1]), int(k[2]))))
        for (kind, a, b, p), to in zip(bad, low[at]):
            cells = cells_of(a) if kind == 0 else np.intersect1d(cells_of(a), cells_of(b))
            cells = cells[owner[cells] == p]
            owner[cells] = to
            moved += len(cells)
    raise RuntimeError('the partition repair did not converge in {} rounds'.format(max_rounds))


def repair(x, topo, owner, L, periodic, tol, max_rounds=50):
    """Move cells until no slave entity is endangered; returns (owner, cells moved)."""
    return _repair(upper_mask(x, L, periodic, tol), topo, owner, max_rounds)


class PairingError(RuntimeError):
    """A periodic space whose pairing dolfin broke on this partition."""


_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/CellType.h>
#include <stdexcept>
namespace py = pybind11;
using namespace dolfin;
typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IMat;
typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1> IVec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DMat;

// The distributed mesh `cells` (global vertex ids; triangles or tetrahedra, by the number
// of columns) with global ids `cell_ids`, the
// vertices this rank holds being the block [vertex_offset, vertex_offset + rows(x)) of
// the global numbering, as dolfin's own data for a mesh read from a file: with
// `partition`, each local cell's destination rank. Filled first and built after
// (`build`), so the caller's arrays can go before the build runs beside this copy.
std::shared_ptr<LocalMeshData> local_data(
    const Mesh& like, Eigen::Ref<const IMat> cells, Eigen::Ref<const IVec> cell_ids,
    Eigen::Ref<const DMat> x, std::int64_t vertex_offset, std::int64_t n_global_vertices,
    std::int64_t n_global_cells, Eigen::Ref<const Eigen::VectorXi> partition)
{
  auto data = std::make_shared<LocalMeshData>(like.mpi_comm());
  const int nv = cells.cols(), gdim = x.cols();
  if (nv != 3 && nv != 4)
    throw std::runtime_error("local_data: triangles or tetrahedra only");
  data->topology.dim = nv - 1;
  data->topology.cell_type = nv == 4 ? CellType::Type::tetrahedron
                                     : CellType::Type::triangle;
  data->topology.num_vertices_per_cell = nv;
  data->topology.num_global_cells = n_global_cells;
  const std::size_t n = cells.rows();
  data->topology.cell_vertices.resize(boost::extents[n][nv]);
  data->topology.global_cell_indices.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    for (int j = 0; j < nv; ++j)
      data->topology.cell_vertices[i][j] = cells(i, j);
    data->topology.global_cell_indices[i] = cell_ids(i);
  }
  data->topology.cell_partition.assign(partition.data(), partition.data() + partition.size());
  data->geometry.dim = gdim;
  data->geometry.num_global_vertices = n_global_vertices;
  const std::size_t m = x.rows();
  data->geometry.vertex_coordinates.resize(boost::extents[m][gdim]);
  data->geometry.vertex_indices.resize(m);
  for (std::size_t i = 0; i < m; ++i)
  {
    for (int j = 0; j < gdim; ++j)
      data->geometry.vertex_coordinates[i][j] = x(i, j);
    data->geometry.vertex_indices[i] = vertex_offset + i;
  }
  return data;
}

// The mesh `local_data` describes, built by dolfin as it builds a mesh read from a file.
// No ghosts.
void build(Mesh& out, const LocalMeshData& data)
{
  MeshPartitioning::build_distributed_mesh(out, data, "none");
}

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<LocalMeshData, std::shared_ptr<LocalMeshData>>(m, "LocalMeshData");
  m.def("local_data", &local_data);
  m.def("build", &build);
}
"""

_MODULE = [None]


def _cpp():
    """The compiled rebuild, once; it raises without a compiler, like every JIT helper
    here."""
    import dolfin as df
    if _MODULE[0] is None:
        _MODULE[0] = df.compile_cpp_code(_CODE)
    return _MODULE[0]


class Rebuild:
    """A mesh's new partition, decided and not yet built: everything the build and the
    arrays' move need (`build`, `carry`), the mesh itself not among them, so a caller can
    let it go before the new one is built. Each local cell goes to `partition`'s rank; cells keep their global ids,
    vertices their global ids and coordinates, bit for bit, the vertices first sent to the
    rank holding their block of the global numbering (`hdf5_rows.to_owners`), as dolfin
    builds a mesh it reads. Collective."""

    def __init__(self, mesh, partition):
        import dolfin as df
        from femtools.hdf5_rows import agreed, block_edges, to_owners
        comm = mesh.mpi_comm()
        # an empty mesh holds the communicator: a mesh's own is freed with it, and the
        # build runs after the caller has let `mesh` go
        self.like = df.Mesh(comm)
        tdim = mesh.topology().dim()
        gv = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
        gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.int64)
        self.n = (mesh.num_entities_global(tdim), mesh.num_entities_global(0))
        self.ids = (gc, gv)
        cells = np.ascontiguousarray(gv[mesh.cells()])
        edges = block_edges(self.n[1], comm.size)
        got_i, got_x = to_owners(comm, edges, gv, mesh.coordinates())
        lo, hi = int(edges[comm.rank]), int(edges[comm.rank + 1])
        x = np.empty((hi - lo, mesh.geometry().dim()))
        seen = np.zeros(hi - lo, bool)
        x[got_i - lo], seen[got_i - lo] = got_x, True
        got_i = got_x = None
        if agreed(comm, not seen.all()):
            raise RuntimeError('rebuild: the global vertex numbering has gaps; no rank '
                               'goes on')
        self.data = _cpp().local_data(self.like, cells, gc, x, lo, self.n[1], self.n[0],
                                      np.asarray(partition, dtype=np.int32))

    def build(self):
        """The mesh, built; once."""
        import dolfin as df
        out = df.Mesh(self.like.mpi_comm())
        _cpp().build(out, self.data)
        self.data = None
        return out

    def carry(self, new, cells=(), vertices=()):
        """`carry_by_id` from the mesh this was decided on to `new`, its build."""
        return _carry(new.mpi_comm(), self.n, self.ids, new, cells, vertices)


def rebuild(mesh, partition):
    """`mesh` built again from its own arrays in memory, nothing written: each local cell
    to `partition`'s rank (`Rebuild`). Any rank-local map onto `mesh` -- a parent-cell
    chain -- does not survive it; global ids do (`carry_by_id`). Collective."""
    return Rebuild(mesh, partition).build()


def dual_graph_of(cells, comm):
    """`(rows, cols, offset)`: the dual graph of this rank's cells `cells` (n, 4) or (n, 3)
    of global vertex ids -- two cells adjacent when they share a facet, periodic images
    not --
    in ParMETIS's numbering (local cell i is offset + i, offsets in rank order), as an
    edge list (rows local, cols global). Faces are named by their sorted global vertex ids
    and paired within the rank first (`local_dual`, C++: a sort of one 64-bit key a face,
    the triples compared wherever keys meet), then the rest -- the faces on rank
    boundaries and the domain's -- through the rank a face hashes to. Collective."""
    from femtools.hdf5_rows import agreed, rank_of, route
    size, n = comm.size, len(cells)
    nf = np.shape(cells)[1] - 1                      # a facet's vertices
    offset = comm.exscan(n) or 0
    rows, cols, f, o = _graph_module().local_dual(np.ascontiguousarray(cells, np.int64),
                                                  offset)
    rows, cols = [np.asarray(rows)], [np.asarray(cols)]
    f, o = np.asarray(f).reshape(-1, nf), np.asarray(o)
    # on one rank too: faces whose keys collided within it are paired here
    routed = route(comm, rank_of(f, size), np.concatenate([f, o[:, None]], 1))
    got, src = routed.received[0], np.repeat(np.arange(size), routed.recv_counts)
    k = np.lexsort(got[:, nf - 1::-1].T)
    g, s_ = got[k], src[k]
    same = np.all(g[1:, :nf] == g[:-1, :nf], axis=1)
    if agreed(comm, (same[1:] & same[:-1]).any()):
        raise RuntimeError('dual graph: a facet of three or more cells; the mesh is not a '
                           'manifold, and no rank goes on')
    m = np.nonzero(same)[0]
    # each pair's two cells told of each other, at their own ranks
    back = np.concatenate([np.stack([g[m, nf], g[m + 1, nf]], 1),
                           np.stack([g[m + 1, nf], g[m, nf]], 1)])
    e, = route(comm, np.concatenate([s_[m], s_[m + 1]]), back).received
    rows.append(e[:, 0] - offset)
    cols.append(e[:, 1])
    return np.concatenate(rows), np.concatenate(cols), offset


_PARMETIS_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <dolfin/mesh/Mesh.h>
#include <parmetis.h>
#include <dlfcn.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>
namespace py = pybind11;
using namespace dolfin;
typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1> IVec;
typedef Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IMat;

// ParMETIS, opened here with RTLD_DEEPBIND rather than linked: libdolfin loads PT-SCOTCH's
// ParMETIS compatibility layer, whose ParMETIS_V3_PartKway (and the METIS it calls) would
// otherwise take the calls.
// DEEPBIND binds ParMETIS and its METIS to themselves, whatever was loaded first.
typedef int (*repart_fn)(idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*,
                         idx_t*, idx_t*, real_t*, real_t*, real_t*, idx_t*, idx_t*, idx_t*,
                         MPI_Comm*);
typedef int (*kway_fn)(idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*, idx_t*,
                       idx_t*, real_t*, real_t*, idx_t*, idx_t*, idx_t*, MPI_Comm*);
// PETSc's own build first ($PETSC_DIR/$PETSC_ARCH/lib, the images' --download-parmetis,
// not on the loader's path there), then the system's.
static void* lib()
{
  static void* h = nullptr;
  std::vector<std::string> names;
  const char* dir = getenv("PETSC_DIR");
  const char* arch = getenv("PETSC_ARCH");
  if (dir && arch)
    names.push_back(std::string(dir) + "/" + arch + "/lib/libparmetis.so");
  for (const char* name : {"libparmetis.so", "libparmetis.so.4.0", "libparmetis.so.4"})
    names.push_back(name);
  if (!h)
    for (const std::string& name : names)
      if ((h = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND)))
        break;
  if (!h)
    throw std::runtime_error(std::string("ParMETIS: cannot open libparmetis: ") + dlerror());
  return h;
}

// Each local vertex's rank from ParMETIS on a graph (CSR, rows local, columns in the
// rank-offset numbering `vtxdist`): AdaptiveRepart from the current partition, weighing the
// communication a cut costs against a vertex's move by `itr`; or PartKway from scratch.
// `weighted`: `vwgt` is each vertex's weight, and its size when it moves. The current
// partition is each rank's own vertices (ParMETIS's coupled default), or with `uncoupled`
// `current`, each vertex's part wherever the graph has put it.
std::vector<int> partition(const Mesh& mesh, Eigen::Ref<const IVec> vtxdist,
                           Eigen::Ref<const IVec> xadj, Eigen::Ref<const IVec> adjncy,
                           bool weighted, Eigen::Ref<const IVec> vwgt, bool uncoupled,
                           Eigen::Ref<const IVec> current, bool adaptive, double itr,
                           double imbalance)
{
  MPI_Comm comm = mesh.mpi_comm();
  int size; MPI_Comm_size(comm, &size);
  // a cell's number in the graph (vtxdist, adjncy) is global: past idx_t's range it would
  // wrap silently. Every rank holds the same vtxdist, so every rank throws together
  if (vtxdist(vtxdist.size() - 1) > (std::int64_t)std::numeric_limits<idx_t>::max())
    throw std::runtime_error("ParMETIS: " + std::to_string(vtxdist(vtxdist.size() - 1))
                             + " cells, past this ParMETIS's " + std::to_string(8 * sizeof(idx_t))
                             + "-bit idx_t: build it (and METIS) with IDXTYPEWIDTH 64");
  std::vector<idx_t> vd(vtxdist.data(), vtxdist.data() + vtxdist.size());
  std::vector<idx_t> xa(xadj.data(), xadj.data() + xadj.size());
  std::vector<idx_t> ad(adjncy.data(), adjncy.data() + adjncy.size());
  if (ad.empty()) ad.push_back(0);                // a valid pointer for a rank with no edges
  const std::size_t n = xa.size() - 1;
  std::vector<idx_t> vw(vwgt.data(), vwgt.data() + vwgt.size());
  if (weighted && vw.size() != n)
    throw std::runtime_error("ParMETIS: a weight a vertex, not " + std::to_string(vw.size()));
  if (vw.empty()) vw.push_back(1);                // likewise, for a rank with no vertices
  idx_t* w = weighted ? vw.data() : nullptr;
  idx_t wgtflag = weighted ? 2 : 0, numflag = 0, ncon = 1, nparts = size, edgecut = 0;
  std::vector<real_t> tpwgts(nparts, 1.0 / nparts);
  real_t ubvec = imbalance, ritr = itr;
  idx_t options[4] = {0, 0, 0, 0};
  std::vector<idx_t> part(n > 0 ? n : 1);
  if (uncoupled)
  {
    if ((std::size_t) current.size() != n)
      throw std::runtime_error("ParMETIS: a current part a vertex, not "
                               + std::to_string(current.size()));
    std::copy(current.data(), current.data() + n, part.begin());
    options[0] = 1;
    options[3] = PARMETIS_PSR_UNCOUPLED;
  }
  int rc;
  if (adaptive)
  {
    auto f = (repart_fn)dlsym(lib(), "ParMETIS_V3_AdaptiveRepart");
    rc = f(vd.data(), xa.data(), ad.data(), w, w, nullptr, &wgtflag, &numflag,
           &ncon, &nparts, tpwgts.data(), &ubvec, &ritr, options, &edgecut, part.data(),
           &comm);
  }
  else
  {
    auto f = (kway_fn)dlsym(lib(), "ParMETIS_V3_PartKway");
    rc = f(vd.data(), xa.data(), ad.data(), w, nullptr, &wgtflag, &numflag, &ncon,
           &nparts, tpwgts.data(), &ubvec, options, &edgecut, part.data(), &comm);
  }
  if (rc != METIS_OK)
    throw std::runtime_error("ParMETIS failed");
  return std::vector<int>(part.begin(), part.begin() + n);
}

static inline std::uint64_t mix(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// The rank's own half of the dual graph: its faces (sorted global vertex ids) keyed by a
// 64-bit hash and sorted, faces with equal keys compared by their vertices. Returns
// (rows, cols) for the pairs within the rank -- rows local, cols offset + local -- and
// the faces left alone, (vertices, cell offset + local), for the exchange across ranks;
// faces whose keys collide are left alone too, and the exchange pairs them.
py::tuple local_dual(Eigen::Ref<const IMat> cells, std::int64_t offset)
{
  const std::int64_t n = cells.rows();
  const int nv = cells.cols(), nf = nv - 1;     // a cell's facets, and a facet's vertices
  if (nv != 3 && nv != 4)
    throw std::runtime_error("local_dual: triangles or tetrahedra only");
  static const int drop4[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  static const int drop3[3][3] = {{1, 2, 0}, {0, 2, 0}, {0, 1, 0}};
  auto tri = [&](std::int64_t f, std::int64_t* t) {
    const std::int64_t c = f / nv;
    for (int j = 0; j < nf; ++j)
      t[j] = cells(c, nv == 4 ? drop4[f % nv][j] : drop3[f % nv][j]);
    std::sort(t, t + nf);
  };
  auto same = [&](const std::int64_t* t, const std::int64_t* u) {
    for (int j = 0; j < nf; ++j)
      if (t[j] != u[j]) return false;
    return true;
  };
  std::vector<std::pair<std::uint64_t, std::int64_t>> key(nv * n);
  std::int64_t t[3], u[3];
  for (std::int64_t f = 0; f < nv * n; ++f)
  {
    tri(f, t);
    std::uint64_t k = mix(t[0]);
    for (int j = 1; j < nf; ++j) k = mix(k ^ t[j]);
    key[f] = {k, f};
  }
  std::sort(key.begin(), key.end());
  std::vector<std::int64_t> rows, cols, lone_t, lone_c;
  rows.reserve(4 * n); cols.reserve(4 * n);
  for (std::size_t a = 0; a < key.size();)
  {
    std::size_t b = a + 1;
    while (b < key.size() && key[b].first == key[a].first) ++b;
    if (b - a == 2)
    {
      tri(key[a].second, t); tri(key[a + 1].second, u);
      if (same(t, u))
      {
        const std::int64_t p = key[a].second / nv, q = key[a + 1].second / nv;
        rows.push_back(p); cols.push_back(offset + q);
        rows.push_back(q); cols.push_back(offset + p);
        a = b;
        continue;
      }
    }
    // otherwise -- one face, or faces whose keys collide -- they go to the exchange,
    // which pairs faces by their vertices and nothing else
    for (std::size_t k = a; k < b; ++k)
    {
      tri(key[k].second, t);
      lone_t.insert(lone_t.end(), t, t + nf);
      lone_c.push_back(offset + key[k].second / nv);
    }
    a = b;
  }
  auto arr = [](const std::vector<std::int64_t>& v) {
    return py::array_t<std::int64_t>(v.size(), v.data()); };
  return py::make_tuple(arr(rows), arr(cols), arr(lone_t), arr(lone_c));
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("partition", &partition);
  m.def("local_dual", &local_dual);
  m.attr("IDX_BITS") = 8 * sizeof(idx_t);
}
"""

_PARMETIS = [None]


def _graph_module():
    """The compiled ParMETIS and dual-graph helper, once."""
    import dolfin as df
    if _PARMETIS[0] is None:
        _PARMETIS[0] = df.compile_cpp_code(_PARMETIS_CODE, libraries=['dl'])
    return _PARMETIS[0]

#: ParMETIS's balance tolerance: the busiest rank's cells at most this times the mean
IMBALANCE = 1.02
#: AdaptiveRepart's ratio of a cut's communication to a cell's move (ParMETIS recommends
#: 1000: one repartition per many solver iterations)
ITR = 1000.0


def parmetis_partition(mesh, adaptive=True):
    """Each local cell's rank from ParMETIS on its dual graph: `AdaptiveRepart` from the
    current partition, or `PartKway` from scratch. Collective."""
    gv = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    return _parmetis(mesh, dual_csr(np.ascontiguousarray(gv[mesh.cells()]), mesh.mpi_comm()),
                     adaptive)


def dual_csr(cells, comm):
    """`dual_graph_of` as ParMETIS takes it, `(xadj, adjncy)`: each local cell's
    neighbours in the rank-offset numbering, in order. Each edge packed into one int64
    (row, column) that sorts in the edges' order, so the edge list goes before the CSR is
    made and the sort is of one key, not two; a lexsort where the two do not fit 63 bits.
    Collective."""
    from mpi4py import MPI
    rows, cols, _ = dual_graph_of(cells, comm)
    n = len(cells)
    bits = int(comm.allreduce(int(cols.max(initial=0)), op=MPI.MAX)).bit_length()
    if int(n).bit_length() + bits > 63:
        order = np.lexsort((cols, rows))
        xadj = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype(np.int64)
        return xadj, np.ascontiguousarray(cols[order], dtype=np.int64)
    key = (rows.astype(np.int64) << bits) | cols
    rows = cols = None
    key.sort()
    xadj = np.r_[0, np.cumsum(np.bincount(key >> bits, minlength=n))].astype(np.int64)
    np.bitwise_and(key, (1 << bits) - 1, out=key)
    return xadj, key


def _parmetis(like, csr, adaptive):
    """ParMETIS on a dual graph `(xadj, adjncy)` (`dual_csr`); `like` any dolfin Mesh on
    the communicator (its `mpi_comm` is what the C++ side takes)."""
    comm = like.mpi_comm()
    xadj, adjncy = csr
    vtxdist = np.r_[0, np.cumsum(comm.allgather(len(xadj) - 1))].astype(np.int64)
    none = np.zeros(0, np.int64)
    return np.asarray(_graph_module().partition(like, vtxdist, xadj, adjncy, False, none,
                                                False, none, bool(adaptive), ITR, IMBALANCE),
                      dtype=np.int64)


class Families:
    """A coarse mesh's dual graph, to partition a mesh refined from it by family: ParMETIS
    `AdaptiveRepart` on the coarse cells, each weighed by its descendants, and every
    refined cell to its coarse ancestor's part (`partition`). The graph is the coarse
    mesh's size whatever the refinement has made of it, and the same every pass: built once,
    its vertices the coarse file's rows in blocks. Collective."""

    def __init__(self, mesh_file, comm):
        import h5py
        from femtools.hdf5_rows import block_edges
        path, topo, _, ids = _mesh_datasets(mesh_file)
        self.path, self.ids = path, ids
        with h5py.File(path, 'r') as h:
            self.edges = block_edges(h[topo].shape[0], comm.size)
            a, b = int(self.edges[comm.rank]), int(self.edges[comm.rank + 1])
            cells = np.asarray(h[topo][a:b], dtype=np.int64)
        self.xadj, self.adjncy = dual_csr(cells, comm)

    def partition(self, like, ancestors):
        """Each local cell's rank, `ancestors` its coarse cell (a row of the coarse file):
        the coarse cells' parts from ParMETIS, each weighed by its descendants and starting
        from the rank holding most of them; `like` any mesh on the communicator.
        Collective."""
        from femtools.hdf5_rows import lookup, to_owners
        comm = like.mpi_comm()
        lo, hi = int(self.edges[comm.rank]), int(self.edges[comm.rank + 1])
        anc, count = np.unique(np.asarray(ancestors, dtype=np.int64), return_counts=True)
        idx, held = to_owners(comm, self.edges, anc,
                              np.stack([count, np.full(len(anc), comm.rank)], 1))
        weight = np.bincount(idx - lo, held[:, 0], minlength=hi - lo).astype(np.int64)
        # each coarse cell starts where most of its descendants are (all of them, once a
        # family partition has placed them): ties to the lowest rank
        current = np.full(hi - lo, comm.rank, np.int64)
        k = np.lexsort((held[:, 1], -held[:, 0], idx))
        first = np.r_[True, idx[k][1:] != idx[k][:-1]]
        current[idx[k][first] - lo] = held[k][first, 1]
        part = np.asarray(_graph_module().partition(
            like, self.edges, self.xadj, self.adjncy, True, weight, True, current, True, ITR,
            IMBALANCE), dtype=np.int64)
        return lookup(comm, self.edges, part, np.asarray(ancestors, dtype=np.int64))

    def partition_by_id(self, like, ancestors):
        """`partition` with `ancestors` the coarse cells' global ids, found as rows of the
        coarse file (a file written under MPI need not hold its cells in id order).
        Collective."""
        from femtools.hdf5_rows import rows_of_labels
        anc = np.asarray(ancestors, dtype=np.int64)
        rows = anc if self.ids is None else rows_of_labels(self.path, self.ids, anc,
                                                           like.mpi_comm())
        return self.partition(like, rows)


def _box(mesh, L, comm):
    """L, the box's upper corner, as an array; None means the coordinates' largest values
    (the repair looks at the upper faces alone, so the lower corner does not matter)."""
    if L is not None:
        return np.asarray(L, float)
    from mpi4py import MPI
    x = mesh.coordinates()
    local = x.max(axis=0) if len(x) else np.full(mesh.geometry().dim(), -np.inf)
    return np.array([comm.allreduce(float(v), op=MPI.MAX) for v in local])


def _moves(mesh, L, periodic, comm, log, owner, tol=None):
    """The cells the repair moves, as `(global cell id, from rank, to rank)` on every
    rank, from `owner` (each local cell's proposed rank)."""
    tol = _face_tol(L, tol)
    vmask = upper_mask(mesh.coordinates(), L, periodic, tol)
    tdim = mesh.topology().dim()
    cells = np.asarray(mesh.cells(), np.int64).reshape(-1, tdim + 1)
    cmask = vmask[cells]
    gv = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.int64)
    return _layer_moves(comm, gc, gv[cells], cmask, owner, log)


def _layer_moves(comm, gc, gcells, cmask, owner, log):
    """`_moves` from arrays: each local cell's global id, its global vertex ids (four, or
    three on a triangle mesh), their upper-face masks (`upper_mask`) and its proposed
    rank.

    Only the layer -- the cells with a vertex on an upper periodic face, ~1 % of the
    mesh -- goes to rank 0, at ~50 B a cell (ids and masks, no coordinates): every facet
    that matters contains an upper-face vertex, so all of its cells are in the layer,
    and the repair on the layer alone is exact. Vertices are named by their global
    index, which every rank agrees on. Rank 0's repair is array work (`_repair`).
    """
    nv = np.shape(gcells)[1]
    cmask = np.asarray(cmask, np.uint8).reshape(-1, nv)
    layer = np.nonzero((cmask > 0).any(1))[0]
    part = (np.asarray(gc, np.int64)[layer], np.asarray(gcells, np.int64)[layer],
            cmask[layer], np.asarray(owner, np.int64)[layer])
    parts = comm.gather(part, root=0)
    moves, failure = None, None
    if comm.rank == 0:
        try:
            ids = np.concatenate([p[0] for p in parts])
            vg = np.concatenate([p[1] for p in parts]).reshape(-1)
            vm = np.concatenate([p[2] for p in parts]).reshape(-1)
            owner = np.concatenate([p[3] for p in parts])
            del parts
            uniq, first, t = np.unique(vg, return_index=True, return_inverse=True)
            fixed, moved = _repair(vm[first], t.reshape(-1, nv), owner)
            m = np.nonzero(fixed != owner)[0]
            moves = np.stack([ids[m], owner[m], fixed[m]], 1)
            log('periodic partition: {} cells moved of {} touching an upper face'
                .format(len(m), len(ids)))
        except Exception as exc:          # every rank must hear it, or they wait forever
            failure = '{}: {}'.format(type(exc).__name__, exc)
    failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise RuntimeError('periodic partition: rank 0 could not repair the partition '
                           '({}); no rank goes on'.format(failure))
    return comm.bcast(moves, root=0)


def _apply_moves(gc, part, moves):
    """`part` (each local cell's rank) with the moves of the cells this rank holds."""
    part = np.array(part, dtype=np.int64)
    if len(moves):
        by_id = moves[np.argsort(moves[:, 0])]
        mine = np.isin(gc, by_id[:, 0])
        part[mine] = by_id[np.searchsorted(by_id[:, 0], gc[mine]), 2]
    return part


def repaired_partition(mesh, part, L, periodic, comm, log=print):
    """`part` (each local cell's rank) with the repair's moves applied: a partition the
    periodic dofmap cannot get wrong, decided before anything is built. Collective."""
    moves = _moves(mesh, _box(mesh, L, comm), periodic, comm, log, owner=part)
    gc = np.asarray(mesh.topology().global_indices(mesh.topology().dim()), dtype=np.int64)
    return _apply_moves(gc, part, moves), len(moves)


def repaired(mesh, L, periodic, comm, log=print):
    """A distributed mesh whose partition the periodic dofmap cannot get wrong.

    Returned as it is when no cell needs moving, which is the usual case. Otherwise it
    is built again in memory with the moved cells on their new ranks (`rebuild`). No rank
    ever holds the whole mesh. Any rank-local map onto `mesh` -- a parent-cell chain --
    does not survive a move; global cell ids do.
    """
    if comm.size == 1:
        return mesh
    part, moved = repaired_partition(mesh, np.full(mesh.num_cells(), comm.rank), L,
                                     periodic, comm, log)
    return rebuild(mesh, part) if moved else mesh


def _mesh_datasets(mesh_file):
    """`(h5 path, topology, coordinates, cell ids or None)` of a mesh file: dolfin's HDF5
    layout (`mesh/...`, cell ids `mesh/cell_indices`), or an XDMF mesh's two DataItems
    (the meshers', through meshio: the HDF5 beside it; a cell's id is its row)."""
    if not mesh_file.endswith('.xdmf'):
        return mesh_file, 'mesh/topology', 'mesh/coordinates', 'mesh/cell_indices'
    import xml.etree.ElementTree as ET
    root = ET.parse(mesh_file).getroot()
    where = {}
    for kind in ('Geometry', 'Topology'):
        item = root.find('.//{}/DataItem'.format(kind))
        if item is None or item.get('Format') != 'HDF':
            raise ValueError('{}: no HDF {} DataItem'.format(mesh_file, kind))
        where[kind] = item.text.strip().split(':', 1)
    (h5a, topo), (h5b, coords) = where['Topology'], where['Geometry']
    if h5a != h5b:
        raise ValueError('{}: topology and geometry in different files'.format(mesh_file))
    return os.path.join(os.path.dirname(mesh_file), h5a), topo, coords, None


def read_partitioned(mesh_file, L, periodic, comm, log=print, tol=None, families=None):
    """A mesh file read in blocks and distributed by ParMETIS, dolfin's partitioner never
    run: each rank reads a contiguous block of the cells and of the vertex rows (h5py),
    the dual graph is built from the cells (`dual_graph_of`), ParMETIS `PartKway`
    partitions it from scratch, the periodic repair moves what it must in that partition
    (the layer's upper-face masks fetched from the vertex blocks, `_layer_moves`), and
    dolfin builds the mesh once with it (`build_distributed_mesh`, as `rebuild`). A
    cell's global id is the file's (its `cell_indices`, or its row), a vertex's its row,
    as dolfin's own readers give them. `L` None is the coordinates' extent. Collective;
    under MPI only.

    `families(like, ids, comm)`, given, partitions it instead: each of this rank's cells
    `ids` (global cell ids) to the rank it returns, e.g. its coarse ancestor's part from
    `Families`, which spares ParMETIS a refined mesh's own dual graph."""
    import dolfin as df
    import h5py
    from mpi4py import MPI
    from femtools.hdf5_rows import block_edges, lookup
    path, topo_name, coords_name, ids_name = _mesh_datasets(mesh_file)
    with h5py.File(path, 'r') as h:
        n_c, n_v = h[topo_name].shape[0], h[coords_name].shape[0]
        ce, ve = block_edges(n_c, comm.size), block_edges(n_v, comm.size)
        a, b = int(ce[comm.rank]), int(ce[comm.rank + 1])
        cells = np.asarray(h[topo_name][a:b], dtype=np.int64)
        ids = (np.arange(a, b, dtype=np.int64) if ids_name is None
               else np.asarray(h[ids_name][a:b], dtype=np.int64))
        lo, hi = int(ve[comm.rank]), int(ve[comm.rank + 1])
        x = np.ascontiguousarray(h[coords_name][lo:hi], dtype=np.float64)
        gdim = h[coords_name].shape[1]
    like = df.Mesh(comm)
    if families is None:
        part = _parmetis(like, dual_csr(cells, comm), adaptive=False)
    else:
        part = np.asarray(families(like, ids, comm), dtype=np.int64)
    if L is None:
        local = x.max(axis=0) if len(x) else np.full(gdim, -np.inf)
        L = np.array([comm.allreduce(float(v), op=MPI.MAX) for v in local])
    L = np.asarray(L, float)
    vmask = upper_mask(x, L, periodic, _face_tol(L, tol))
    cmask = lookup(comm, ve, vmask, cells.reshape(-1)).reshape(cells.shape)
    part = _apply_moves(ids, part, _layer_moves(comm, ids, cells, cmask, part, log))
    data = _cpp().local_data(like, cells, ids, x, lo, n_v, n_c, part.astype(np.int32))
    cells = ids = x = cmask = vmask = part = None       # dolfin's copy alone through the build
    out = df.Mesh(comm)
    _cpp().build(out, data)
    data = None
    log('mesh: partitioned by ParMETIS for {} ranks{}'.format(
        comm.size, "" if families is None else ", by family"))
    return out


def read_mesh(mesh_file, L, periodic, comm, log=print, tol=None, families=None):
    """The mesh, distributed so that the periodic dofmap pairs every face entity.

    `mesh_file` is an XDMF mesh (the meshers') or a dolfin HDF5 mesh (`mesh.h5`,
    `refined_mesh.h5`), whose global cell ids are kept. `L` is the box's upper corner;
    None takes the coordinates' largest values.
    Under MPI read in blocks, partitioned by ParMETIS and repaired before the one build
    (`read_partitioned`, by `families` if given); serially
    dolfin's plain read, the defect needing more than one rank.
    """
    import dolfin as df
    periodic = tuple(bool(p) for p in periodic)
    if comm.size > 1:
        return read_partitioned(mesh_file, L, periodic, comm, log, tol, families)
    mesh = df.Mesh()
    if mesh_file.endswith('.xdmf'):
        with df.XDMFFile(comm, mesh_file) as f:
            f.read(mesh)
    else:
        with df.HDF5File(comm, mesh_file, 'r') as h:
            h.read(mesh, 'mesh', False)
    return mesh


def carry_by_id(old, new, cells=(), vertices=()):
    """Arrays that follow a mesh through `rebuild` / `repartition`: `cells` one row per
    local cell of `old`, `vertices` one per local vertex (None passes through), returned
    on `new`'s local cells and vertices, matched by global id -- which a rebuild keeps.
    Each kind is packed into one int64 table (float64 and uint64 bit for bit, bools as
    0/1) and moved in one exchange (`hdf5_rows.move_rows`). Collective."""
    top = old.topology()
    tdim = top.dim()
    return _carry(old.mpi_comm(), (old.num_entities_global(tdim), old.num_entities_global(0)),
                  (np.asarray(top.global_indices(tdim), dtype=np.int64),
                   np.asarray(top.global_indices(0), dtype=np.int64)), new, cells, vertices)


def _carry(comm, n, ids, new, cells, vertices):
    """`carry_by_id` from the global counts `n` and ids `ids` (cells, vertices) of the
    mesh the arrays were on."""
    from femtools.hdf5_rows import as_rows, move_rows

    def move(k, dim, arrays):
        after = np.asarray(new.topology().global_indices(dim), dtype=np.int64)
        keep = [a for a in arrays if a is not None]
        if not keep:
            return tuple(arrays)
        cols = [np.ascontiguousarray(as_rows(a)) for a in keep]
        for c in cols:
            # a column a value: anything narrower than int64 would shift the columns after it
            if c.dtype != bool and c.dtype.itemsize != 8:
                raise TypeError("carry_by_id: a {} array; only 8-byte dtypes and bools are "
                                "carried".format(c.dtype))
        table = np.concatenate([c.astype(np.int64) if c.dtype == bool else c.view(np.int64)
                                for c in cols], axis=1)
        got = move_rows(comm, n[k], ids[k], table, after)
        out, at = [], 0
        for a, c in zip(keep, cols):
            w = c.shape[1]
            part = np.ascontiguousarray(got[:, at:at + w])
            at += w
            part = part.astype(bool) if c.dtype == bool else part.view(c.dtype)
            out.append(part.reshape((len(after),) + np.asarray(a).shape[1:]))
        it = iter(out)
        return tuple(None if a is None else next(it) for a in arrays)
    return move(0, new.topology().dim(), tuple(cells)), move(1, 0, tuple(vertices))


def repartition(mesh, L, periodic, comm, log=print):
    """The mesh rebalanced in memory, nothing written: `repartition_plan`, built -- or
    `mesh` itself when there is nothing to move. Global cell and vertex ids are kept; any
    rank-local map onto `mesh` -- a parent-cell chain -- does not survive it. Collective."""
    plan = repartition_plan(mesh, L, periodic, comm, log)
    return mesh if plan is None else plan.build()


def repartition_plan(mesh, L, periodic, comm, log=print, propose=None):
    """The mesh's rebalance decided, not built (`Rebuild`): ParMETIS `AdaptiveRepart` from
    its current partition, or `propose(mesh)` (each local cell's rank: `Families.partition`),
    and the periodic repair applied to that partition -- or None, when the busiest rank is
    already within ParMETIS's tolerance (`IMBALANCE`) or no cell moves. Collective."""
    from mpi4py import MPI
    if comm.size == 1:
        return None
    # within ParMETIS's own tolerance there is nothing to rebalance: it would still move a
    # few hundred cells to polish the cut, and that costs a whole build
    n = mesh.num_cells()
    if comm.allreduce(n, op=MPI.MAX) <= IMBALANCE * comm.allreduce(n, op=MPI.SUM) / comm.size:
        return None
    part = parmetis_partition(mesh) if propose is None else propose(mesh)
    part, _ = repaired_partition(mesh, part, L, periodic, comm, log)
    if not comm.allreduce(bool((part != comm.rank).any()), op=MPI.LOR):
        return None                       # balanced already: nothing to build
    return Rebuild(mesh, part)


def _nodes_a_cell(tdim, degree):
    """A Lagrange cell's nodes: its vertices, and for degree 2 its edge midpoints too."""
    return tdim + 1 if degree == 1 else (tdim + 1) * (tdim + 2) // 2


def check_pairing(W, L, periodic, comm, tol=None, lower=None):
    """Raise unless every face node of W carries its periodic class's one global dof.

    W is the Taylor-Hood mixed space; its velocity's first component and its pressure
    are checked, at vertices and edge midpoints on every periodic face. The box runs
    from `lower` (default the origin) to `L`.
    """
    tdim = W.mesh().topology().dim()
    _check(W.mesh(), [(W.sub(0).sub(0).dofmap(), _nodes_a_cell(tdim, 2)),
                      (W.sub(1).dofmap(), _nodes_a_cell(tdim, 1))],
           L, periodic, comm, 'the Stokes space', tol=tol, lower=lower)


def check_space_pairing(V, L, periodic, comm, what='the space', exempt=None, lower=None):
    """`check_pairing` for a Lagrange space of degree 1 or 2, scalar or vector (its first
    component stands for the rest: the components share one dofmap layout).

    `exempt`, an (m, 3) array of this rank's node positions, is the cut exception: a class of identified face nodes may hold more than one dof when
    every node in it is exempt -- on a cut mesh's inlet or outlet, where the node the
    constraint maps them to is not in the mesh (a corner line's (L, L) and (L, 0) both
    map to (0, 0)), so dolfin leaves each its own dof."""
    degree = V.ufl_element().degree()
    if degree not in (1, 2):
        raise ValueError('check_space_pairing handles Lagrange degree 1 or 2, not {}'
                         .format(degree))
    dm = V.sub(0).dofmap() if V.num_sub_spaces() else V.dofmap()
    _check(V.mesh(), [(dm, _nodes_a_cell(V.mesh().topology().dim(), degree))], L, periodic,
           comm, what, exempt, lower=lower)


def _pad3(x):
    """Points (rows) with three coordinates, a 2D mesh's with a zero third."""
    x = np.asarray(x, float)
    return np.hstack([x, np.zeros((len(x), 3 - x.shape[1]))])


def _check(mesh, dofmaps, L, periodic, comm, what, exempt=None, tol=None, lower=None):
    """Every face node, in every (dofmap, nodes a cell) given, one dof per periodic class.
    Triangle and tetrahedron meshes; the box runs from `lower` (default the origin) to
    `L`."""
    from femtools.periodic import cell_dofs
    x = _pad3(mesh.coordinates())
    nv = mesh.topology().dim() + 1
    cells = np.asarray(mesh.cells(), np.int64).reshape(-1, nv)
    L = np.array(_as3(L), float)
    lo = np.zeros(3) if lower is None else np.array(_as3(lower), float)
    periodic = _as3(periodic, False)
    tol = _face_tol(L - lo, tol)
    axes = [a for a in range(3) if periodic[a]]
    if not axes:
        return
    on = np.zeros(len(x), bool)
    for a in axes:
        on |= (np.abs(x[:, a] - lo[a]) < tol) | (np.abs(x[:, a] - L[a]) < tol)
    touching = np.nonzero(on[cells].any(1))[0]
    # the nodes of every touching cell -- vertices, then edge midpoints in dolfin's
    # local edge order -- and which of them lie on a periodic face. Vectorised, with
    # the cells' dofs fetched in one compiled call (dolfin's Python has no bulk
    # accessor)
    xv = x[cells[touching]]
    edges = EDGES if nv == 4 else TRIANGLE_CELL_EDGES
    nodes = np.concatenate(
        [xv, np.stack([(xv[:, a] + xv[:, b]) / 2 for a, b in edges], 1)], 1)
    face = np.zeros(nodes.shape[:2], bool)
    for a in axes:
        face |= (np.abs(nodes[..., a] - lo[a]) < tol) | (np.abs(nodes[..., a] - L[a]) < tol)
    pos = nodes[face]
    dofs = []
    for dm, n in dofmaps:
        l2g = np.asarray(dm.tabulate_local_to_global_dofs(), dtype=np.int64)
        cd = cell_dofs(dm, touching, n)
        g = np.full(face.shape, -1, np.int64)       # -1: no dof at this node (P1's edges)
        g[:, :n] = l2g[cd]
        dofs.append(g[face])
    dofs = np.stack(dofs, 1) if dofs else np.zeros((len(pos), 0), np.int64)
    if exempt is not None:
        exempt = _pad3(exempt)
    split = _split_classes(comm, pos, dofs, L, axes, tol, exempt, lower=lo)
    if split:
        raise PairingError(
            'the periodic constraint left {} face node classes of {} with more than one '
            "dof: dolfin's parallel periodic dofmap broke the pairing on this partition "
            '(femtools.periodic_partition). The solve would be wrong at those nodes; it '
            'is refused.'.format(split, what))


def _keys(pos, tol):
    """A node position as an integer key (rounded to the tolerance), the same on every
    rank for the same node."""
    return np.round(np.asarray(pos, float) / tol).astype(np.int64)


def _split_classes(comm, pos, dofs, L, axes, tol, exempt, lower=None):
    """How many periodic classes of face nodes carry more than one dof of some dofmap,
    and are not forgiven (every node of the class an exempt position), on every rank.

    No rank holds the face nodes: each rank's rows -- (the node's own key, its class's
    key, one dof a dofmap) -- are deduplicated, then judged where their class's key
    hashes to (`hdf5_rows.route`). With `exempt`, a row is first marked at the rank its
    own position's key hashes to, where the exempt positions are sent too."""
    from mpi4py import MPI
    from femtools.hdf5_rows import as_rows, rank_of, route
    size = comm.Get_size()
    img = np.array(pos, float).reshape(-1, 3)
    lo = np.zeros(3) if lower is None else np.asarray(lower, float)
    for a in axes:
        # a node on an upper face, taken down to its image on the lower one
        img[:, a] = np.where(np.abs(img[:, a] - L[a]) < tol, img[:, a] - (L[a] - lo[a]),
                             img[:, a])
    rows = np.concatenate([_keys(pos, tol).reshape(-1, 3), _keys(img, tol),
                           as_rows(dofs)], 1)
    rows = rows[(rows[:, 6:] >= 0).any(1)] if rows.shape[1] > 6 else rows
    rows = np.unique(rows, axis=0) if len(rows) else rows.reshape(0, rows.shape[1])
    if exempt is None:
        ex = np.zeros(len(rows), np.int64)
    else:
        mine = _keys(exempt, tol).reshape(-1, 3)
        r_rows = route(comm, rank_of(rows[:, :3], size), rows)
        r_ex = route(comm, rank_of(mine, size), mine)
        got, ex_keys = r_rows.received[0], r_ex.received[0]
        if len(got) and len(ex_keys):
            both = np.concatenate([ex_keys, got[:, :3]])
            _, inv = np.unique(both, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            is_ex = np.zeros(inv.max() + 1, bool)
            is_ex[inv[:len(ex_keys)]] = True
            flag = is_ex[inv[len(ex_keys):]]
        else:
            flag = np.zeros(len(got), bool)
        rows, ex = got, flag.astype(np.int64)
    r = route(comm, rank_of(rows[:, 3:6], size), rows[:, 3:], ex)
    got, ex = r.received
    split = 0
    if len(got):
        order = np.lexsort(got[:, 2::-1].T)
        got, ex = got[order], ex[order]
        start = np.r_[True, np.any(got[1:, :3] != got[:-1, :3], axis=1)]
        cls = np.cumsum(start) - 1
        n_cls = int(cls[-1]) + 1
        forgiven = np.bincount(cls, weights=1 - ex, minlength=n_cls) == 0
        many = np.zeros(n_cls, bool)
        for j in range(got.shape[1] - 3):
            d = got[:, 3 + j]
            m = d >= 0
            pairs = np.unique(np.stack([cls[m], d[m]], 1), axis=0)
            many |= np.bincount(pairs[:, 0], minlength=n_cls) > 1
        split = int(np.count_nonzero(many & ~forgiven))
    return comm.allreduce(split, op=MPI.SUM)
