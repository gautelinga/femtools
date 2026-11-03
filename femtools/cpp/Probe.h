#ifndef __PROBE_H
#define __PROBE_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace dolfin
{

  // A Function's values at one point, snapshot after snapshot: the point's cell and the
  // element's basis there are found once, and each evaluation is a dot product.
  class Probe
  {

  public:

    // A probe at x, its cell searched for on this process (an error if not there)
    Probe(const Array<double>& x, const FunctionSpace& V);

    // A probe in a known local cell (no search)
    Probe(const double* x, const FunctionSpace& V, std::size_t cell_id);

    Probe(const Probe& p);

    virtual ~Probe() {}

    // The local cell holding the probe, and its mesh
    std::size_t cell_index() const { return dolfin_cell->index(); }
    const Mesh& mesh() const { return dolfin_cell->mesh(); }

    // Evaluate u at the probe: one more snapshot
    virtual void eval(const Function& u);

    // Every snapshot of one component
    std::vector<double> get_probe_sub(std::size_t comp) const;

    // One snapshot, every component
    std::vector<double> get_probe_at_snapshot(std::size_t i) const;

    // One component at one snapshot
    double get_probe_component_and_snapshot(std::size_t comp, std::size_t i) const;

    // Write one component, or all, to filename; id names the probe in it
    void dump(std::size_t comp, std::string filename, std::size_t id=0) const;
    void dump(std::string filename, std::size_t id=0) const;

    // Components (one for a scalar), and snapshots taken
    std::size_t value_size() const { return _value_size_loc; }
    std::size_t number_of_evaluations() const { return _num_evals; }

    // The probe's position, three coordinates
    std::vector<double> coordinates() const { return _x; }

    // Remove snapshot i; remove every snapshot
    virtual void erase_snapshot(std::size_t i);
    virtual void clear();

    // Append a snapshot of given values (value_size of them): a restart's
    void restart_probe(const Array<double>& u);

    // Every snapshot of every component, out[comp*n + snapshot] for n snapshots
    void values(double* out) const;

  protected:

    // The coefficients of u on the probe's cell
    void restrict(const Function& u);

    // Append a snapshot of the values from the current coefficients
    void push_value();

    void check_snapshot(std::size_t i, std::size_t n, const char* what) const;
    void check_component(std::size_t comp, std::size_t n, const char* what) const;

    std::vector<std::vector<double>> basis_matrix;

    std::vector<double> coefficients;

    std::vector<double> vertex_coordinates;

    std::vector<double> _x;

    std::shared_ptr<const FiniteElement> _element;

    std::size_t _value_size_loc, _num_evals;

    std::vector<std::vector<double>> _probes;

    std::unique_ptr<Cell> dolfin_cell;

    ufc::cell ufc_cell;

  };

}

#endif
