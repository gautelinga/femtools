"""WGM * p for a quadratic p on P1 over a UnitCubeMesh, written by rank zero as
`(dof coordinates, d/dx, d/dy, d/dz)` rows to argv[1]; any rank count, in dolfin's
`ghost_mode` argv[2] (default none).

    mpirun -n 2 python3 tests/gradient_worker.py OUT.npy [shared_facet]
"""
import sys

import dolfin as df
import numpy as np
from mpi4py import MPI

from femtools.gradient import compute_weighted_gradient_matrix

df.parameters["ghost_mode"] = sys.argv[2] if len(sys.argv) > 2 else "none"
mesh = df.UnitCubeMesh(4, 4, 4)
Q = df.FunctionSpace(mesh, "CG", 1)
DG = df.FunctionSpace(mesh, "DG", 0)
p = df.interpolate(df.Expression("x[0]*x[0] + x[1]*x[2] - 2*x[2]*x[2]*x[0]", degree=3), Q)
lo, hi = Q.dofmap().ownership_range()
x = Q.tabulate_dof_coordinates().reshape(-1, 3)[:hi - lo]
cols = [x]
for i in range(3):
    G = df.assemble(df.TrialFunction(DG) * df.TestFunction(Q) * df.dx)
    dP = df.assemble(df.TrialFunction(Q).dx(i) * df.TestFunction(DG) * df.dx)
    W = compute_weighted_gradient_matrix(G, dP, df.Function(DG))
    out = df.Function(Q)
    W.mult(p.vector(), out.vector())
    cols.append(out.vector().get_local()[:, None])
rows = MPI.COMM_WORLD.gather(np.hstack(cols), root=0)
if MPI.COMM_WORLD.rank == 0:
    np.save(sys.argv[1], np.vstack(rows))
