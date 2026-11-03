/*
From fenicstools by M. Mortensen.
*/
#include "Probe.h"
#include "Probes.h"
#include "GradProbe.h"
#include "GradProbes.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace dolfin;
namespace py = pybind11;

// Points or values from Python, as a contiguous double array whatever was passed (a
// slice or a column of an array is copied, not read with the wrong strides)
using doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;

// dolfin's Array over a numpy array's data (no copy: an Array cannot be returned)
#define AS_ARRAY(name, a) const Array<double> name((a).size(), const_cast<double*>((a).data()))

static const FunctionSpace& space(const py::object& V)
{
  return V.attr("_cpp_object").cast<const FunctionSpace&>();
}

static const Function& function(const py::object& u)
{
  return u.attr("_cpp_object").cast<const Function&>();
}

// A flat C++ vector as a numpy array of the given shape
static py::array_t<double> shaped(const std::vector<double>& v, std::vector<py::ssize_t> shape)
{
  return py::array_t<double>(shape, v.data());
}

// What Probes and GradProbes share, bound on either
template <class C, class Class>
void bind_collection(Class& c)
{
  c.def("eval", [](C& self, py::object u){ self.eval(function(u)); })
   .def("dump", (void (C::*)(std::size_t, std::string) const) &C::dump, py::arg("i"), py::arg("filename"), "dump probes")
   .def("dump", (void (C::*)(std::string) const) &C::dump, py::arg("filename"), "dump probes")
   .def("add_positions", [](C& self, const doubles& x, py::object V){
      AS_ARRAY(a, x);
      self.add_positions(a, space(V));
    })
   .def("get_probe", &C::get_probe)
   .def("get_probe_id", &C::get_probe_id)
   .def("get_probe_ids", &C::get_probe_ids)
   .def("get_probes_component_and_snapshot", &C::get_probes_component_and_snapshot)
   .def("local_size", &C::local_size)
   .def("value_size", &C::value_size)
   .def("get_total_number_probes", &C::get_total_number_probes)
   .def("number_of_evaluations", &C::number_of_evaluations)
   .def("erase_snapshot", &C::erase_snapshot)
   .def("clear", &C::clear)
   .def("set_probes_from_ids", [](C& self, const doubles& u){
      self.set_probes_from_ids(std::vector<double>(u.data(), u.data() + u.size()));
    })
   .def("values", [](const C& self){
      return shaped(self.values(), {(py::ssize_t) self.local_size(),
                                    (py::ssize_t) self.value_size(),
                                    (py::ssize_t) self.number_of_evaluations()});
    }, "every local probe's every component at every snapshot, (probes, components, snapshots)");
}

// What Probe and GradProbe share
template <class P, class Class>
void bind_probe(Class& c)
{
  c.def(py::init([](const doubles& x, py::object V){ AS_ARRAY(a, x); return P(a, space(V)); }))
   .def("eval", [](P& self, py::object u){ self.eval(function(u)); })
   .def("get_probe_sub", &P::get_probe_sub)
   .def("get_probe_at_snapshot", &P::get_probe_at_snapshot)
   .def("get_probe_component_and_snapshot", &P::get_probe_component_and_snapshot)
   .def("dump", (void (P::*)(std::size_t, std::string, std::size_t) const) &P::dump, py::arg("i"), py::arg("filename"), py::arg("id")=0, "dump probes")
   .def("dump", (void (P::*)(std::string, std::size_t) const) &P::dump, py::arg("filename"), py::arg("id")=0, "dump probes")
   .def("value_size", &P::value_size)
   .def("number_of_evaluations", &P::number_of_evaluations)
   .def("coordinates", &P::coordinates)
   .def("erase_snapshot", &P::erase_snapshot)
   .def("clear", &P::clear)
   .def("restart_probe", [](P& self, const doubles& u){ AS_ARRAY(a, u); self.restart_probe(a); });
}

PYBIND11_MODULE(probe11, m)
{
  py::class_<Probe, std::shared_ptr<Probe>> probe(m, "Probe");
  bind_probe<Probe>(probe);

  py::class_<GradProbe, std::shared_ptr<GradProbe>> grad_probe(m, "GradProbe");
  bind_probe<GradProbe>(grad_probe);
  grad_probe
    .def("eval_grad", [](GradProbe& self, py::object u){ self.eval_grad(function(u)); })
    .def("get_probe_grad_sub", &GradProbe::get_probe_grad_sub)
    .def("get_probe_grad_at_snapshot", &GradProbe::get_probe_grad_at_snapshot)
    .def("get_probe_grad_component_and_snapshot", &GradProbe::get_probe_grad_component_and_snapshot)
    .def("number_of_grad_evaluations", &GradProbe::number_of_grad_evaluations)
    .def("clear_grad", &GradProbe::clear_grad);

  py::class_<Probes, std::shared_ptr<Probes>> probes(m, "Probes");
  probes
    .def(py::init([](const doubles& x, py::object V){
      AS_ARRAY(a, x);
      return std::make_shared<Probes>(a, space(V));
    }))
    .def(py::init([](const Probes& located, py::object V){
      return std::make_shared<Probes>(located, space(V));
    }))
    .def(py::init([](const GradProbes& located, py::object V){
      return std::make_shared<Probes>(located, space(V));
    }));
  bind_collection<Probes>(probes);

  py::class_<GradProbes, std::shared_ptr<GradProbes>> grad_probes(m, "GradProbes");
  grad_probes
    .def(py::init([](const doubles& x, py::object V){
      AS_ARRAY(a, x);
      return std::make_shared<GradProbes>(a, space(V));
    }))
    .def(py::init([](const Probes& located, py::object V){
      return std::make_shared<GradProbes>(located, space(V));
    }))
    .def(py::init([](const GradProbes& located, py::object V){
      return std::make_shared<GradProbes>(located, space(V));
    }));
  bind_collection<GradProbes>(grad_probes);
  grad_probes
    .def("eval_grad", [](GradProbes& self, py::object u){ self.eval_grad(function(u)); })
    .def("get_probes_grad_component_and_snapshot", &GradProbes::get_probes_grad_component_and_snapshot)
    .def("geom_dim", &GradProbes::geom_dim)
    .def("number_of_grad_evaluations", &GradProbes::number_of_grad_evaluations)
    .def("clear_grad", &GradProbes::clear_grad)
    .def("grad_values", [](const GradProbes& self){
      return shaped(self.grad_values(), {(py::ssize_t) self.local_size(),
                                         (py::ssize_t) (self.value_size()*self.geom_dim()),
                                         (py::ssize_t) self.number_of_grad_evaluations()});
    }, "every local probe's every gradient component at every gradient snapshot");
}

/*cppimport
<%
from dolfin.jit.jit import dolfin_pc
setup_pybind11(cfg)
cfg['sources'] = ['Probe.cpp', 'Probes.cpp', 'GradProbe.cpp', 'GradProbes.cpp']
# a header edited alone must rebuild the module too: cppimport checksums these
cfg['dependencies'] = ['Probe.h', 'Probes.h', 'GradProbe.h', 'GradProbes.h',
                       'ProbeCollection.h']
cfg['libraries'] = ['dolfin']
cfg['include_dirs'] = dolfin_pc['include_dirs']
cfg['library_dirs'] = dolfin_pc['library_dirs']
# dolfin's own definitions (HAS_MPI, HAS_PETSC, ...): its headers lay out its classes by
# them, so code built without them sees other classes than the library has
cfg['compiler_args'] = ['-D{}'.format(k) if v is None else '-D{}={}'.format(k, v)
                        for k, v in dict(dolfin_pc['define_macros']).items()]
%>
*/
