#ifndef __GRADPROBES_H
#define __GRADPROBES_H

#include "GradProbe.h"

namespace dolfin { class Probes; }

namespace dolfin
{

  class GradProbes
  {

  public:

    GradProbes(const Array<double>& x, const FunctionSpace& V);

    GradProbes(const GradProbes& p);

    // Probes at the points of located, for a space on the same mesh (no search)
    GradProbes(const Probes& located, const FunctionSpace& V);
    GradProbes(const GradProbes& located, const FunctionSpace& V);

    GradProbes() {};

    virtual ~GradProbes();

    // evaluate all probes
    void eval(const Function& u);
    void eval_grad(const Function& u);

    // dump component i of probes to filename
    void dump(std::size_t i, std::string filename);

    // dump all probes to filename
    void dump(std::string filename);

    // Add new probe positions
    void add_positions(const Array<double>& x, const FunctionSpace& V);

    // Return an instance of probe i
    std::shared_ptr<GradProbe> get_probe(std::size_t i);

    // Return id of probe i
    std::size_t get_probe_id(std::size_t i);

    // Return ids of all probes
    std::vector<std::size_t> get_probe_ids();

    // Return one snapshot of one component of the probe
    std::vector<double> get_probes_component_and_snapshot(std::size_t comp, std::size_t i);
    std::vector<double> get_probes_grad_component_and_snapshot(std::size_t comp, std::size_t i);

    // Return the number of probes on this process
    std::size_t local_size() {return _allprobes.size();};

    // Return number of components probed for
    std::size_t value_size() {return _value_size;};

    // Return space dimension
    std::size_t geom_dim() { return _geom_dim; };

    // Return total number of probes on all processes combined
    std::size_t get_total_number_probes() {return total_number_probes;};

    // Return number of evaluations of probes
    std::size_t number_of_evaluations() {return _num_evals;};

    // Erase one snapshot ot the all probes
    virtual void erase_snapshot(std::size_t i);

    // Reset probe by deleting all previous evaluations
    virtual void clear();
    virtual void clear_grad();

    // Set probe from Array
    void set_probes_from_ids(const Array<double>& u);

  protected:

    friend class Probes;

    std::vector<std::pair<std::size_t, GradProbe*> > _allprobes;

    std::size_t total_number_probes, _value_size, _num_evals, _num_grad_evals, _geom_dim;

  };
}

#endif
