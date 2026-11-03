#include "GradProbe.h"

using namespace dolfin;

// The local cell that contains x, or unsigned int's max
static std::size_t locate(const double* x, const Mesh& mesh)
{
  const Point point(mesh.geometry().dim(), x);
  return mesh.bounding_box_tree()->compute_first_entity_collision(point);
}

GradProbe::GradProbe(const Array<double>& x, const FunctionSpace& V) :
  GradProbe(x.data(), V, locate(x.data(), *V.mesh()))
{
}

GradProbe::GradProbe(const double* x, const FunctionSpace& V, std::size_t cell_id) :
  Probe(x, V, cell_id), _num_grad_evals(0), _geom_dim(V.mesh()->geometry().dim())
{
  const std::size_t n = _value_size_loc*_geom_dim;
  _probes_grad.resize(n);
  basis_grad_matrix.assign(n, std::vector<double>(_element->space_dimension()));
  std::vector<double> basis_grad(n);
  const int cell_orientation = 0;
  for (std::size_t i = 0; i < _element->space_dimension(); ++i)
  {
    _element->evaluate_basis_derivatives(i, 1, basis_grad.data(), x,
                                         vertex_coordinates.data(), cell_orientation);
    for (std::size_t j = 0; j < n; ++j)
      basis_grad_matrix[j][i] = basis_grad[j];
  }
}

void GradProbe::eval_grad(const Function& u)
{
  restrict(u);
  push_value();
  for (std::size_t j = 0; j < _probes_grad.size(); j++)
  {
    double v = 0.0;
    for (std::size_t i = 0; i < coefficients.size(); i++)
      v += coefficients[i]*basis_grad_matrix[j][i];
    _probes_grad[j].push_back(v);
  }
  _num_grad_evals++;
}

void GradProbe::erase_snapshot(std::size_t i)
{
  const bool aligned = _num_grad_evals == _num_evals;
  Probe::erase_snapshot(i);
  if (aligned)
  {
    for (auto& p : _probes_grad)
      p.erase(p.begin() + i);
    _num_grad_evals--;
  }
}

void GradProbe::clear()
{
  Probe::clear();
  clear_grad();
}

void GradProbe::clear_grad()
{
  for (auto& p : _probes_grad)
    p.clear();
  _num_grad_evals = 0;
}

std::vector<double> GradProbe::get_probe_grad_sub(std::size_t comp) const
{
  check_component(comp, _probes_grad.size(), "get a gradient component");
  return _probes_grad[comp];
}

std::vector<double> GradProbe::get_probe_grad_at_snapshot(std::size_t i) const
{
  check_snapshot(i, _num_grad_evals, "get a gradient snapshot");
  std::vector<double> x(_probes_grad.size());
  for (std::size_t j = 0; j < x.size(); j++)
    x[j] = _probes_grad[j][i];
  return x;
}

double GradProbe::get_probe_grad_component_and_snapshot(std::size_t comp, std::size_t i) const
{
  check_component(comp, _probes_grad.size(), "get a gradient component");
  check_snapshot(i, _num_grad_evals, "get a gradient snapshot");
  return _probes_grad[comp][i];
}

void GradProbe::grad_values(double* out) const
{
  for (std::size_t j = 0; j < _probes_grad.size(); j++)
    std::copy(_probes_grad[j].begin(), _probes_grad[j].end(), out + j*_num_grad_evals);
}
