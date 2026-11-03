#include "Probe.h"
#include <limits>

using namespace dolfin;

// The local cell that contains x, or unsigned int's max
static std::size_t locate(const double* x, const Mesh& mesh)
{
  const Point point(mesh.geometry().dim(), x);
  return mesh.bounding_box_tree()->compute_first_entity_collision(point);
}

Probe::Probe(const Array<double>& x, const FunctionSpace& V) :
  Probe(x.data(), V, locate(x.data(), *V.mesh()))
{
}

Probe::Probe(const double* x, const FunctionSpace& V, std::size_t cell_id) :
  _element(V.element()), _num_evals(0)
{
  auto mesh = V.mesh();
  const std::size_t gdim = mesh->geometry().dim();

  if (cell_id == std::numeric_limits<unsigned int>::max())
    dolfin_error("Probe.cpp", "set probe", "Probe is not found on processor");

  _x.resize(3);
  for (std::size_t i = 0; i < 3; i++)
    _x[i] = (i < gdim ? x[i] : 0.0);

  // Components: one for a scalar, the product of the value dimensions for a tensor
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);
  _probes.resize(_value_size_loc);

  dolfin_cell.reset(new Cell(*mesh, cell_id));
  dolfin_cell->get_cell_data(ufc_cell);
  coefficients.resize(_element->space_dimension());
  dolfin_cell->get_vertex_coordinates(vertex_coordinates);

  // The basis at x, once: an evaluation is then a dot product
  basis_matrix.assign(_value_size_loc,
                      std::vector<double>(_element->space_dimension()));
  std::vector<double> basis(_value_size_loc);
  const int cell_orientation = 0;
  for (std::size_t i = 0; i < _element->space_dimension(); ++i)
  {
    _element->evaluate_basis(i, basis.data(), x, vertex_coordinates.data(),
                             cell_orientation);
    for (std::size_t j = 0; j < _value_size_loc; ++j)
      basis_matrix[j][i] = basis[j];
  }
}

Probe::Probe(const Probe& p) :
  basis_matrix(p.basis_matrix), coefficients(p.coefficients),
  vertex_coordinates(p.vertex_coordinates), _x(p._x), _element(p._element),
  _value_size_loc(p._value_size_loc), _num_evals(p._num_evals), _probes(p._probes),
  dolfin_cell(new Cell(p.dolfin_cell->mesh(), p.dolfin_cell->index())),
  ufc_cell(p.ufc_cell)
{
}

void Probe::restrict(const Function& u)
{
  u.restrict(coefficients.data(), *_element, *dolfin_cell,
             vertex_coordinates.data(), ufc_cell);
}

void Probe::push_value()
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
  {
    double v = 0.0;
    for (std::size_t i = 0; i < coefficients.size(); i++)
      v += coefficients[i]*basis_matrix[j][i];
    _probes[j].push_back(v);
  }
  _num_evals++;
}

void Probe::eval(const Function& u)
{
  restrict(u);
  push_value();
}

void Probe::check_snapshot(std::size_t i, std::size_t n, const char* what) const
{
  if (i >= n)
    dolfin_error("Probe.cpp", what, "no snapshot %d: there are %d", (int) i, (int) n);
}

void Probe::check_component(std::size_t comp, std::size_t n, const char* what) const
{
  if (comp >= n)
    dolfin_error("Probe.cpp", what, "no component %d: there are %d", (int) comp, (int) n);
}

void Probe::erase_snapshot(std::size_t i)
{
  check_snapshot(i, _num_evals, "erase a snapshot");
  for (auto& p : _probes)
    p.erase(p.begin() + i);
  _num_evals--;
}

void Probe::clear()
{
  for (auto& p : _probes)
    p.clear();
  _num_evals = 0;
}

std::vector<double> Probe::get_probe_sub(std::size_t comp) const
{
  check_component(comp, _value_size_loc, "get a component");
  return _probes[comp];
}

std::vector<double> Probe::get_probe_at_snapshot(std::size_t i) const
{
  check_snapshot(i, _num_evals, "get a snapshot");
  std::vector<double> x(_value_size_loc);
  for (std::size_t j = 0; j < _value_size_loc; j++)
    x[j] = _probes[j][i];
  return x;
}

double Probe::get_probe_component_and_snapshot(std::size_t comp, std::size_t i) const
{
  check_component(comp, _value_size_loc, "get a component");
  check_snapshot(i, _num_evals, "get a snapshot");
  return _probes[comp][i];
}

void Probe::values(double* out) const
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
    std::copy(_probes[j].begin(), _probes[j].end(), out + j*_num_evals);
}

void Probe::dump(std::size_t comp, std::string filename, std::size_t id) const
{
  check_component(comp, _value_size_loc, "dump a component");
  std::ofstream fp(filename.c_str(), std::ios_base::out);
  fp << std::scientific;
  fp << "Probe id = " << id << std::endl;
  fp << "Number of evaluations = " << _num_evals << std::endl;
  fp << "Coordinates:" << std::endl;
  fp << _x[0] << " " << _x[1] << " " << _x[2] << std::endl;
  fp << "Values for component " << comp << std::endl;
  for (double v : _probes[comp])
    fp << v << std::endl;
  fp << std::endl;
}

void Probe::dump(std::string filename, std::size_t id) const
{
  std::ofstream fp(filename.c_str(), std::ios_base::out);
  fp << std::scientific;
  fp << "Probe id = " << id << std::endl;
  fp << "Number of evaluations = " << _num_evals << std::endl;
  fp << "Coordinates:" << std::endl;
  fp << _x[0] << " " << _x[1] << " " << _x[2] << std::endl;
  fp << "Values for all components:" << std::endl;
  for (std::size_t s = 0; s < _num_evals; s++)
  {
    for (std::size_t j = 0; j < _value_size_loc; j++)
      fp << _probes[j][s] << " ";
    fp << std::endl;
  }
}

void Probe::restart_probe(const Array<double>& u)
{
  if (u.size() != _value_size_loc)
    dolfin_error("Probe.cpp", "restart a probe", "%d values for %d components",
                 (int) u.size(), (int) _value_size_loc);
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j].push_back(u[j]);
  _num_evals++;
}
