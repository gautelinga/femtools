#ifndef __GRADPROBES_H
#define __GRADPROBES_H

#include "GradProbe.h"
#include "ProbeCollection.h"

namespace dolfin
{

  // A Function's values and gradients at many points (`ProbeCollection`, `GradProbe`)
  class GradProbes : public ProbeCollection<GradProbe>
  {

  public:

    GradProbes(const Array<double>& x, const FunctionSpace& V) :
      ProbeCollection<GradProbe>(x, V), _num_grad_evals(0),
      _geom_dim(V.mesh()->geometry().dim()) {}

    template <class Q>
    GradProbes(const ProbeCollection<Q>& located, const FunctionSpace& V) :
      ProbeCollection<GradProbe>(located, V), _num_grad_evals(0),
      _geom_dim(V.mesh()->geometry().dim()) {}

    // Evaluate every probe's values and gradient: one more snapshot of each
    void eval_grad(const Function& u);

    // One gradient component at one snapshot, of every local probe
    std::vector<double> get_probes_grad_component_and_snapshot(std::size_t comp,
                                                               std::size_t i) const;

    std::size_t geom_dim() const { return _geom_dim; }
    std::size_t number_of_grad_evaluations() const { return _num_grad_evals; }

    // As a GradProbe's: the gradients too while the histories line up
    void erase_snapshot(std::size_t i) override;
    void clear() override;
    void clear_grad();

    // `values` for the gradients: probe-major, then component, then snapshot
    std::vector<double> grad_values() const;

  private:

    std::size_t _num_grad_evals, _geom_dim;

  };

}

#endif
