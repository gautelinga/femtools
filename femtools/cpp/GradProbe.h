#ifndef __GRADPROBE_H
#define __GRADPROBE_H

#include "Probe.h"

namespace dolfin
{

  // A Probe that also takes the gradient: eval_grad adds a snapshot of the values and
  // one of the gradient, d(component j)/d(x_k) at j*gdim + k. Plain eval adds values
  // only, so the two histories line up only while every evaluation takes the gradient.
  class GradProbe : public Probe
  {

  public:

    GradProbe(const Array<double>& x, const FunctionSpace& V);

    GradProbe(const double* x, const FunctionSpace& V, std::size_t cell_id);

    GradProbe(const GradProbe& p) = default;

    // Evaluate u and its gradient at the probe: one more snapshot of each
    void eval_grad(const Function& u);

    std::vector<double> get_probe_grad_sub(std::size_t comp) const;
    std::vector<double> get_probe_grad_at_snapshot(std::size_t i) const;
    double get_probe_grad_component_and_snapshot(std::size_t comp, std::size_t i) const;

    std::size_t number_of_grad_evaluations() const { return _num_grad_evals; }

    // Snapshot i of the values, and of the gradient when the histories line up (every
    // evaluation took it); clear both, or the gradients alone
    void erase_snapshot(std::size_t i) override;
    void clear() override;
    void clear_grad();

    // Every gradient snapshot, out[comp*n + snapshot] for n of them
    void grad_values(double* out) const;

  protected:

    std::vector<std::vector<double>> basis_grad_matrix;

    std::size_t _num_grad_evals, _geom_dim;

    std::vector<std::vector<double>> _probes_grad;

  };

}

#endif
