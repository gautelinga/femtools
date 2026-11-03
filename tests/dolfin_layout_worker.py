"""Worker for tests/test_dolfin_layout.py: under mpirun, write a mesh and two Functions
with dolfin's HDF5File and with `femtools.dolfin_layout`, and report on rank zero whether the
files hold the same datasets, dtypes, values and attributes, whether ours are compressed,
and whether dolfin reads ours back to the same values.

    mpirun -n R python3 tests/dolfin_layout_worker.py OUT_DIR
"""
import os
import sys

import dolfin as df
import h5py
import numpy as np

from femtools import dolfin_layout


class Periodic(df.SubDomain):
    """Periodic in x and y on the unit cube: a slave on an upper face maps to the lower."""

    def inside(self, x, on_boundary):
        return bool(on_boundary and (df.near(x[0], 0) or df.near(x[1], 0))
                    and not (df.near(x[0], 1) or df.near(x[1], 1)))

    def map(self, x, y):
        y[0] = x[0] - 1 if df.near(x[0], 1) else x[0]
        y[1] = x[1] - 1 if df.near(x[1], 1) else x[1]
        y[2] = x[2]


def main(out):
    comm = df.MPI.comm_world
    mesh = df.UnitCubeMesh(comm, 5, 4, 6)
    V1 = df.FunctionSpace(mesh, "P", 1)
    V2 = df.VectorFunctionSpace(mesh, "P", 2, constrained_domain=Periodic())
    p = df.interpolate(df.Expression("sin(3*x[0]) + x[1]*x[2]", degree=2), V1)
    u = df.interpolate(df.Expression(("cos(2*pi*x[0])", "x[2]*x[2]", "sin(2*pi*x[1])"),
                                     degree=2), V2)
    a, b = os.path.join(out, "dolfin.h5"), os.path.join(out, "ours.h5")
    with df.HDF5File(comm, a, "w") as h:
        h.write(mesh, "mesh")
        h.write(u, "u")
        h.write(p, "p")
    dolfin_layout.write_mesh(b, mesh)
    dolfin_layout.write_function(b, u, "u")
    dolfin_layout.write_function(b, p, "p")
    comm.Barrier()
    # dolfin reads ours back: the mesh, and the Functions into the same spaces
    m2 = df.Mesh(comm)
    with df.HDF5File(comm, b, "r") as h:
        h.read(m2, "mesh", False)
        u2, p2 = df.Function(V2), df.Function(V1)
        h.read(u2, "u")
        h.read(p2, "p")
    back = (m2.num_entities_global(3) == mesh.num_entities_global(3)
            and np.array_equal(u2.vector().get_local(), u.vector().get_local())
            and np.array_equal(p2.vector().get_local(), p.vector().get_local()))
    back = comm.allreduce(int(back), op=__import__("mpi4py").MPI.MIN)
    if comm.rank:
        return
    lines = ["read back by dolfin: {}".format("same" if back else "DIFFERENT")]
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        names = []
        fa.visititems(lambda n, o: names.append(n))
        mine = []
        fb.visititems(lambda n, o: mine.append(n))
        lines.append("names: {}".format("same" if sorted(names) == sorted(mine) else
                                        "DIFFERENT {} {}".format(names, mine)))
        for n in names:
            x, y = fa[n], fb[n]
            ax = {k: np.asarray(v).tobytes() for k, v in x.attrs.items()}
            ay = {k: np.asarray(v).tobytes() for k, v in y.attrs.items()}
            what = "same" if ax == ay else "DIFFERENT attrs {} {}".format(
                dict(x.attrs), dict(y.attrs))
            if isinstance(x, h5py.Dataset):
                if x.dtype != y.dtype or x.shape != y.shape:
                    what = "DIFFERENT {} {} {} {}".format(x.dtype, y.dtype, x.shape, y.shape)
                elif x[()].tobytes() != y[()].tobytes():
                    what = "DIFFERENT values"
                elif y.compression != "gzip" or not y.shuffle:
                    what = "NOT COMPRESSED"
            lines.append("{}: {}".format(n, what))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
