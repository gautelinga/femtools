#include "GradProbes.h"
#include "Probes.h"
#include <dolfin/common/MPI.h>

using namespace dolfin;

GradProbes::GradProbes(const Array<double>& x, const FunctionSpace& V) :
  total_number_probes(0), _num_evals(0)
{
  _value_size = 1;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);
  _num_grad_evals = 0;
  _geom_dim = V.mesh()->geometry().dim();
  add_positions(x, V);
}
//
// Probes at the points of another collection, for a space on the same mesh
template <class Located>
static void same_points(const Located& from, const FunctionSpace& V,
                        std::vector<std::pair<std::size_t, GradProbe*> >& to)
{
  bool other_mesh = false;
  for (auto& p : from)
    other_mesh = other_mesh || (p.second->mesh().id() != V.mesh()->id());
  if (MPI::max(V.mesh()->mpi_comm(), (int) other_mesh))
    dolfin_error("GradProbes.cpp", "reuse probe positions",
                 "the function space is on another mesh than the probes");
  to.reserve(from.size());
  for (auto& p : from)
    to.emplace_back(p.first, new GradProbe(p.second->coordinates().data(), V,
                                          p.second->cell_index()));
}

GradProbes::GradProbes(const Probes& located, const FunctionSpace& V) :
  total_number_probes(located.total_number_probes), _num_evals(0)
{
  _value_size = 1;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);
  _num_grad_evals = 0;
  _geom_dim = V.mesh()->geometry().dim();
  same_points(located._allprobes, V, _allprobes);
}

GradProbes::GradProbes(const GradProbes& located, const FunctionSpace& V) :
  total_number_probes(located.total_number_probes), _num_evals(0)
{
  _value_size = 1;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);
  _num_grad_evals = 0;
  _geom_dim = V.mesh()->geometry().dim();
  same_points(located._allprobes, V, _allprobes);
}
//
GradProbes::GradProbes(const GradProbes& p)
{
  _allprobes = p._allprobes;
  total_number_probes = p.total_number_probes;
  _value_size = p._value_size;
  _num_evals = p._num_evals;
  _num_grad_evals = p._num_grad_evals;
  _geom_dim = p._geom_dim;
  for (std::size_t i = 0; i < local_size(); i++)
  {    
    _allprobes[i].second = new GradProbe(*(p._allprobes[i].second));
  }
}
//
GradProbes::~GradProbes()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    delete _allprobes[i].second;   
  }
  _allprobes.clear();      
}
//
void GradProbes::add_positions(const Array<double>& x, const FunctionSpace& V)
{
  const Mesh& mesh = *V.mesh();
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t N = x.size() / gdim;
  const std::size_t old_N = total_number_probes;
  total_number_probes += N;

  // Build tree on all processes (collective), also those without points
  auto tree = mesh.bounding_box_tree();

  // Create probes only where found
  const std::size_t not_found = std::numeric_limits<unsigned int>::max();
  for (std::size_t i = 0; i < N; i++)
  {
    const double* xi = x.data() + i*gdim;
    const std::size_t cell = tree->compute_first_entity_collision(Point(gdim, xi));
    if (cell != not_found)
      _allprobes.emplace_back(old_N + i, new GradProbe(xi, V, cell));
  }
}
//
void GradProbes::eval(const Function& u)
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->eval(u);
  }
  _num_evals++;
}
//
void GradProbes::eval_grad(const Function& u)
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->eval_grad(u);
  }
  _num_evals++;
  _num_grad_evals++;
}
//
void GradProbes::erase_snapshot(std::size_t i)
{
  for (std::size_t j = 0; j < local_size(); j++)
  {
    _allprobes[j].second->erase_snapshot(i);
  }
  _num_evals--;
}
//
void GradProbes::clear()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->clear();
  }
  _num_evals = 0;
}
void GradProbes::clear_grad()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->clear_grad();
  }
  _num_grad_evals = 0;
}
//
void GradProbes::dump(std::size_t i, std::string filename)
{
  std::string ss="";  
  for (std::size_t j = 0; j < local_size(); j++)
  {
    ss = filename;
    ss.append("_");
    ss.append(boost::lexical_cast<std::string>(_allprobes[j].first));
    ss.append(".probe");
    _allprobes[j].second->dump(i, ss, get_probe_id(j));
    ss.clear();
  }
}
//
void GradProbes::dump(std::string filename)
{
  std::string ss="";  
  for (std::size_t j = 0; j < local_size(); j++)
  {
    ss = filename;
    ss.append("_");
    ss.append(boost::lexical_cast<std::string>(_allprobes[j].first));
    ss.append(".probe");
    _allprobes[j].second->dump(ss, get_probe_id(j));
    ss.clear();
  }
}
//
std::shared_ptr<GradProbe> GradProbes::get_probe(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("GradProbes.cpp", "get probe", "Wrong index!");
  }
  return std::make_shared<GradProbe>(*_allprobes[i].second);
}
//
std::size_t GradProbes::get_probe_id(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("GradProbes.cpp", "get probe_id", "Wrong index!");
  }
  return _allprobes[i].first;
}

std::vector<std::size_t> GradProbes::get_probe_ids()
{
  std::vector<std::size_t> ids;  
  for (std::size_t i = 0; i < local_size(); i++)
  {
    std::size_t probe_id = _allprobes[i].first;
    ids.push_back(probe_id);
  }
  return ids;
}

std::vector<double> GradProbes::get_probes_component_and_snapshot(std::size_t comp, std::size_t i)
{
  std::vector<double> vals;  
  for (std::size_t j = 0; j < local_size(); j++)
  {
    GradProbe* probe = (GradProbe*) _allprobes[j].second;
    vals.push_back(probe->get_probe_component_and_snapshot(comp, i));
  }
  return vals;
}

std::vector<double> GradProbes::get_probes_grad_component_and_snapshot(std::size_t comp, std::size_t i)
{
  std::vector<double> vals;
  for (std::size_t j = 0; j < local_size(); j++)
  {
    GradProbe* probe = (GradProbe*) _allprobes[j].second;
    vals.push_back(probe->get_probe_grad_component_and_snapshot(comp, i));
  }
  return vals;
}

void GradProbes::set_probes_from_ids(const Array<double>& u)
{
  assert(u.size() == local_size() * value_size()); 
  Array<double> _u(value_size());
  for (std::size_t i = 0; i < local_size(); i++)
  {
    GradProbe* probe = (GradProbe*) _allprobes[i].second;
    for (std::size_t j=0; j<value_size(); j++)
      _u[j] = u[i*value_size() + j];
    probe->restart_probe(_u);
  }
}

