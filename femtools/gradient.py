__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
import os
import cppimport
from dolfin import Function, FunctionSpace, assemble, TrialFunction, TestFunction, dx

compiled_gradient_module = cppimport.imp_from_filepath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "gradient_weight.cpp"),
    "femtools.cpp.gradient_weight")
compute_weighted_gradient_matrix = compiled_gradient_module.compute_weighted_gradient_matrix
compute_DG0_to_CG_weight_matrix = compiled_gradient_module.compute_DG0_to_CG_weight_matrix

def weighted_gradient_matrix(mesh, i, family='CG', degree=1, constrained_domain=None):
    """Compute weighted gradient matrix

    The matrix allows you to compute the gradient of a P1 Function
    through a simple matrix vector product

    CG family:
        p_ is the pressure solution on CG1
        dPdX = weighted_gradient_matrix(mesh, 0, 'CG', degree)
        V = FunctionSpace(mesh, 'CG', degree)
        dpdx = Function(V)
        dpdx.vector()[:] = dPdX * p_.vector()

        The space for dpdx must be continuous Lagrange of some order

    CR family:
        p_ is the pressure solution on CR
        dPdX = weighted_gradient_matrix(mesh, 0, 'CR', 1)
        V = FunctionSpace(mesh, 'CR', 1)
        dpdx = Function(V)
        dpdx.vector()[:] = dPdX * p_.vector()

    """

    DG = FunctionSpace(mesh, 'DG', 0)

    if family == 'CG':
        # Source and Target spaces are CG_1 and CG_degree
        S = FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
        T = FunctionSpace(mesh, 'CG', degree, constrained_domain=constrained_domain)
    elif family == 'CR':
        if degree != 1:
            print('\033[1;37;34m%s\033[0m' % 'Ignoring degree')

        # Source and Target spaces are CR
        S = FunctionSpace(mesh, 'CR', 1, constrained_domain=constrained_domain)
        T = S
    else:
        raise ValueError('Only CG and CR families are allowed.')

    # the averaging matrix (target dofs x cells) is the same for every direction: built
    # once, and multiplied into each direction's cell gradients
    A = assemble(TrialFunction(DG)*TestFunction(T)*dx)
    compiled_gradient_module.compute_DG0_to_CG_weight_matrix(A, Function(DG))

    def direction(ii):
        dP = assemble(TrialFunction(S).dx(ii)*TestFunction(DG)*dx)
        return compiled_gradient_module.matMatMult(A, dP)

    if isinstance(i, (tuple, list)):
        return [direction(ii) for ii in i]
    return direction(i)
