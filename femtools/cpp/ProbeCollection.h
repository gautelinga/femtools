#ifndef __PROBECOLLECTION_H
#define __PROBECOLLECTION_H

#include <dolfin/common/MPI.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

  // The components of V's values: one for a scalar, the product of the value
  // dimensions for a tensor
  inline std::size_t value_size_of(const FunctionSpace& V)
  {
    std::size_t n = 1;
    for (std::size_t i = 0; i < V.element()->value_rank(); i++)
      n *= V.element()->value_dimension(i);
    return n;
  }

  // Probes (P: Probe or GradProbe) at many points, each held by one rank: what Probes
  // and GradProbes share. Every rank is given the same points; a point is kept by the
  // lowest rank that finds it in its cells (one on a partition boundary is in several
  // ranks' cells), and by none when it lies outside the mesh. A probe's id is its
  // point's place among all points given.
  template <class P>
  class ProbeCollection
  {

  public:

    ProbeCollection(const Array<double>& x, const FunctionSpace& V) :
      _total(0), _value_size(value_size_of(V)), _num_evals(0)
    {
      add_positions(x, V);
    }

    // Probes at the points of another collection, for a space on the same mesh (no
    // search: the cells are the collection's)
    template <class Q>
    ProbeCollection(const ProbeCollection<Q>& located, const FunctionSpace& V) :
      _total(located.get_total_number_probes()), _value_size(value_size_of(V)),
      _num_evals(0)
    {
      bool other_mesh = false;
      for (auto& p : located.probes())
        other_mesh = other_mesh || (p.second->mesh().id() != V.mesh()->id());
      if (MPI::max(V.mesh()->mpi_comm(), (int) other_mesh))
        dolfin_error("ProbeCollection.h", "reuse probe positions",
                     "the function space is on another mesh than the probes");
      _probes.reserve(located.probes().size());
      for (auto& p : located.probes())
        _probes.emplace_back(p.first, std::unique_ptr<P>(
          new P(p.second->coordinates().data(), V, p.second->cell_index())));
    }

    ProbeCollection(const ProbeCollection& c) :
      _total(c._total), _value_size(c._value_size), _num_evals(c._num_evals)
    {
      _probes.reserve(c._probes.size());
      for (auto& p : c._probes)
        _probes.emplace_back(p.first, std::unique_ptr<P>(new P(*p.second)));
    }

    virtual ~ProbeCollection() {}

    // Evaluate every probe: one more snapshot
    void eval(const Function& u)
    {
      for (auto& p : _probes)
        p.second->eval(u);
      _num_evals++;
    }

    // More points, given alike on every rank; before any evaluation, since the new
    // probes would have no history. Collective.
    void add_positions(const Array<double>& x, const FunctionSpace& V)
    {
      if (_num_evals > 0)
        dolfin_error("ProbeCollection.h", "add probe positions",
                     "the probes have %d snapshots, which new ones would lack",
                     (int) _num_evals);
      const Mesh& mesh = *V.mesh();
      const MPI_Comm comm = mesh.mpi_comm();
      const std::size_t gdim = mesh.geometry().dim();
      const std::size_t N = x.size()/gdim;
      const std::size_t first = _total;
      _total += N;

      // the tree on every rank (collective), also those finding no point
      auto tree = mesh.bounding_box_tree();
      const std::size_t not_found = std::numeric_limits<unsigned int>::max();
      const unsigned long rank = MPI::rank(comm), size = MPI::size(comm);
      std::vector<std::size_t> cell(N);
      std::vector<unsigned long> mine(N), lowest(N);
      for (std::size_t i = 0; i < N; i++)
      {
        cell[i] = tree->compute_first_entity_collision(Point(gdim, x.data() + i*gdim));
        mine[i] = cell[i] == not_found ? size : rank;
      }
      MPI_Allreduce(mine.data(), lowest.data(), N, MPI_UNSIGNED_LONG, MPI_MIN, comm);
      for (std::size_t i = 0; i < N; i++)
        if (lowest[i] == rank)
          _probes.emplace_back(first + i, std::unique_ptr<P>(
            new P(x.data() + i*gdim, V, cell[i])));
    }

    // Write each probe's component i, or all, to filename_<id>.probe
    void dump(std::size_t i, std::string filename) const
    {
      for (auto& p : _probes)
        p.second->dump(i, filename + "_" + std::to_string(p.first) + ".probe", p.first);
    }

    void dump(std::string filename) const
    {
      for (auto& p : _probes)
        p.second->dump(filename + "_" + std::to_string(p.first) + ".probe", p.first);
    }

    // A copy of the local probe i, and its id
    std::shared_ptr<P> get_probe(std::size_t i) const
    {
      check(i);
      return std::make_shared<P>(*_probes[i].second);
    }

    std::size_t get_probe_id(std::size_t i) const
    {
      check(i);
      return _probes[i].first;
    }

    std::vector<std::size_t> get_probe_ids() const
    {
      std::vector<std::size_t> ids;
      for (auto& p : _probes)
        ids.push_back(p.first);
      return ids;
    }

    // One component at one snapshot, of every local probe
    std::vector<double> get_probes_component_and_snapshot(std::size_t comp,
                                                          std::size_t i) const
    {
      std::vector<double> vals;
      for (auto& p : _probes)
        vals.push_back(p.second->get_probe_component_and_snapshot(comp, i));
      return vals;
    }

    std::size_t local_size() const { return _probes.size(); }
    std::size_t value_size() const { return _value_size; }
    std::size_t get_total_number_probes() const { return _total; }
    std::size_t number_of_evaluations() const { return _num_evals; }

    // Remove snapshot i of every probe; remove every snapshot
    virtual void erase_snapshot(std::size_t i)
    {
      if (i >= _num_evals)
        dolfin_error("ProbeCollection.h", "erase a snapshot",
                     "no snapshot %d: there are %d", (int) i, (int) _num_evals);
      for (auto& p : _probes)
        p.second->erase_snapshot(i);
      _num_evals--;
    }

    virtual void clear()
    {
      for (auto& p : _probes)
        p.second->clear();
      _num_evals = 0;
    }

    // One snapshot from given values, value_size a local probe in local order: a
    // restart's
    void set_probes_from_ids(const std::vector<double>& u)
    {
      if (u.size() != _probes.size()*_value_size)
        dolfin_error("ProbeCollection.h", "set probe values",
                     "%d values for %d probes of %d components", (int) u.size(),
                     (int) _probes.size(), (int) _value_size);
      for (std::size_t k = 0; k < _probes.size(); k++)
      {
        const Array<double> values(_value_size, const_cast<double*>(u.data() + k*_value_size));
        _probes[k].second->restart_probe(values);
      }
      _num_evals++;
    }

    // Every local probe's every component at every snapshot: probe-major, then
    // component, then snapshot
    std::vector<double> values() const
    {
      std::vector<double> out(_probes.size()*_value_size*_num_evals);
      for (std::size_t k = 0; k < _probes.size(); k++)
        _probes[k].second->values(out.data() + k*_value_size*_num_evals);
      return out;
    }

    const std::vector<std::pair<std::size_t, std::unique_ptr<P>>>& probes() const
    {
      return _probes;
    }

  protected:

    void check(std::size_t i) const
    {
      if (i >= _probes.size())
        dolfin_error("ProbeCollection.h", "get a probe",
                     "no local probe %d: this rank holds %d", (int) i,
                     (int) _probes.size());
    }

    std::vector<std::pair<std::size_t, std::unique_ptr<P>>> _probes;

    std::size_t _total, _value_size, _num_evals;

  };

}

#endif
