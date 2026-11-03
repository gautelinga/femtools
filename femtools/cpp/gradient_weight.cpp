/*
<%
from dolfin.jit.jit import dolfin_pc
setup_pybind11(cfg)
cfg['include_dirs'] = dolfin_pc['include_dirs']
cfg['library_dirs'] = dolfin_pc['library_dirs']
# dolfin's own definitions (HAS_MPI, HAS_PETSC, ...), which its headers lay its classes
# out by
cfg['compiler_args'] = ['-std=c++11'] + [
    '-D{}'.format(k) if v is None else '-D{}={}'.format(k, v)
    for k, v in dict(dolfin_pc['define_macros']).items()]
%>
*/

#include <pybind11/pybind11.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/common/MPI.h>

using namespace dolfin;

void compute_weight(Function& DG)
{
  // 1/volume on each cell, on the cells this rank owns: a DG0 dof is its one cell's, so a
  // ghost cell (ghost_mode other than "none") is its owner's to weigh, and counting it
  // here as well would double its weight
  std::shared_ptr<const FunctionSpace> V = DG.function_space();
  std::shared_ptr<const Mesh> mesh = V->mesh();
  std::shared_ptr<const GenericDofMap> dofmap = V->dofmap();
  std::vector<double> ws(V->element()->space_dimension());
  GenericVector& weight = *DG.vector();
  weight.zero();
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    auto dofs = dofmap->cell_dofs(cell->index());
    std::fill(ws.begin(), ws.end(), 1./cell->volume());
    weight.set_local(ws.data(), dofs.size(), dofs.data());
  }
  weight.apply("insert");
}

void compute_DG0_to_CG_weight_matrix(GenericMatrix& A, Function& DG)
{
  // A (target dofs x DG0 dofs) becomes the averaging matrix: row r's entries w_c^2 / sum
  // of the row's w_c, the cells' weights (compute_weight)
  compute_weight(DG);
  const MPI_Comm comm = DG.function_space()->mesh()->mpi_comm();
  const std::size_t size = dolfin::MPI::size(comm);
  const std::pair<std::size_t, std::size_t> rows = A.local_range(0);
  const std::size_t m = rows.second - rows.first;
  const GenericVector& weight = *DG.vector();
  const std::pair<std::size_t, std::size_t> mine = weight.local_range();
  std::vector<double> local;
  weight.get_local(local);

  // every rank's first weight, to find a dof's owner
  std::vector<std::size_t> starts;
  dolfin::MPI::all_gather(comm, mine.first, starts);
  auto owner = [&starts](std::size_t dof) {
    return std::size_t(std::upper_bound(starts.begin(), starts.end(), dof)
                       - starts.begin()) - 1;
  };
  auto off = [&mine](std::size_t dof) { return dof < mine.first || dof >= mine.second; };

  // the rows' columns, read once, and the other ranks' weights they need, asked once each
  std::vector<std::vector<std::size_t>> columns(m);
  std::vector<double> values;
  std::vector<std::vector<std::size_t>> needed(size);
  for (std::size_t row = 0; row < m; row++)
  {
    A.getrow(rows.first + row, columns[row], values);
    for (std::size_t dof : columns[row])
      if (off(dof))
        needed[owner(dof)].push_back(dof);
  }
  for (auto& d : needed)
  {
    std::sort(d.begin(), d.end());
    d.erase(std::unique(d.begin(), d.end()), d.end());
  }
  std::vector<std::vector<std::size_t>> asked;
  dolfin::MPI::all_to_all(comm, needed, asked);
  std::vector<std::vector<double>> answers(size), got;
  for (std::size_t p = 0; p < size; p++)
    for (std::size_t dof : asked[p])
      answers[p].push_back(local[dof - mine.first]);
  dolfin::MPI::all_to_all(comm, answers, got);
  std::unordered_map<std::size_t, double> remote;
  for (std::size_t p = 0; p < size; p++)
    for (std::size_t k = 0; k < needed[p].size(); k++)
      remote[needed[p][k]] = got[p][k];

  for (std::size_t row = 0; row < m; row++)
  {
    const std::vector<std::size_t>& cols = columns[row];
    values.resize(cols.size());
    double s = 0.0;
    for (std::size_t i = 0; i < cols.size(); i++)
    {
      values[i] = off(cols[i]) ? remote.at(cols[i]) : local[cols[i] - mine.first];
      s += values[i];
    }
    for (double& v : values)
      v = v * v / s;
    A.setrow(rows.first + row, cols, values);
  }
  A.apply("insert");
}

// A PETSc product as a dolfin matrix. The wrapper takes its own reference to the Mat,
// so the product's own one is dropped here: kept, every call would leak the product.
std::shared_ptr<GenericMatrix> wrapped(Mat C, PetscErrorCode ierr, const char* what)
{
  if (ierr)
    throw std::runtime_error(std::string(what) + " failed (PETSc error "
                             + std::to_string(ierr) + ")");
  auto out = std::make_shared<PETScMatrix>(C);
  MatDestroy(&C);
  return out;
}

std::shared_ptr<GenericMatrix> MatTransposeMatMult(const GenericMatrix& A, const GenericMatrix& B)
{
  Mat C;
  PetscErrorCode ierr = MatTransposeMatMult(as_type<const PETScMatrix>(A).mat(),
                                            as_type<const PETScMatrix>(B).mat(),
                                            MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  return wrapped(C, ierr, "MatTransposeMatMult");
}

std::shared_ptr<GenericMatrix> MatMatTransposeMult(const GenericMatrix& A, const GenericMatrix& B)
{
  Mat C;
  PetscErrorCode ierr = MatMatTransposeMult(as_type<const PETScMatrix>(A).mat(),
                                            as_type<const PETScMatrix>(B).mat(),
                                            MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  return wrapped(C, ierr, "MatMatTransposeMult");
}

std::shared_ptr<GenericMatrix> matMatMult(const GenericMatrix& A, const GenericMatrix& B)
{
  Mat C;
  PetscErrorCode ierr = MatMatMult(as_type<const PETScBaseMatrix>(A).mat(),
                                   as_type<const PETScBaseMatrix>(B).mat(),
                                   MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  return wrapped(C, ierr, "MatMatMult");
}

std::shared_ptr<GenericMatrix> compute_weighted_gradient_matrix(GenericMatrix& A, const GenericMatrix& dP, Function& DG)
{
  compute_DG0_to_CG_weight_matrix(A, DG);
  return matMatMult(A, dP);
}

namespace py = pybind11;

PYBIND11_MODULE(gradient_weight, m)
{
  m.def("compute_DG0_to_CG_weight_matrix", [](GenericMatrix& M, py::object v){
    auto _v = v.attr("_cpp_object").cast<Function*>();
    compute_DG0_to_CG_weight_matrix(M, *_v);
  });
  m.def("compute_weighted_gradient_matrix", [](GenericMatrix& A, const GenericMatrix& dP, py::object v){
    auto _v = v.attr("_cpp_object").cast<Function*>();
    return compute_weighted_gradient_matrix(A, dP, *_v);
  });
  m.def("matMatMult", [](const GenericMatrix& A, const GenericMatrix& B){
    return matMatMult(A, B);
  });
  m.def("MatMatTransposeMult", [](const GenericMatrix& A, const GenericMatrix& B){
    return MatMatTransposeMult(A, B);
  });
  m.def("MatTransposeMatMult", [](const GenericMatrix& A, const GenericMatrix& B){
    return MatTransposeMatMult(A, B);
  });
}
