#include "GradProbes.h"

using namespace dolfin;

template class dolfin::ProbeCollection<dolfin::GradProbe>;

void GradProbes::eval_grad(const Function& u)
{
  for (auto& p : _probes)
    p.second->eval_grad(u);
  _num_evals++;
  _num_grad_evals++;
}

std::vector<double> GradProbes::get_probes_grad_component_and_snapshot(std::size_t comp,
                                                                       std::size_t i) const
{
  std::vector<double> vals;
  for (auto& p : _probes)
    vals.push_back(p.second->get_probe_grad_component_and_snapshot(comp, i));
  return vals;
}

void GradProbes::erase_snapshot(std::size_t i)
{
  const bool aligned = _num_grad_evals == _num_evals;
  ProbeCollection<GradProbe>::erase_snapshot(i);
  if (aligned)
    _num_grad_evals--;
}

void GradProbes::clear()
{
  ProbeCollection<GradProbe>::clear();
  _num_grad_evals = 0;
}

void GradProbes::clear_grad()
{
  for (auto& p : _probes)
    p.second->clear_grad();
  _num_grad_evals = 0;
}

std::vector<double> GradProbes::grad_values() const
{
  const std::size_t n = _value_size*_geom_dim*_num_grad_evals;
  std::vector<double> out(_probes.size()*n);
  for (std::size_t k = 0; k < _probes.size(); k++)
    _probes[k].second->grad_values(out.data() + k*n);
  return out;
}
