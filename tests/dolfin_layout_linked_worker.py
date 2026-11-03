"""Worker for tests/test_dolfin_layout.py: dof tables written once (`write_dofs`) and
linked from a Function's group (`write_function(..., dofs=...)`).

    mpirun -n R python3 tests/dolfin_layout_linked_worker.py OUT tables
        writes OUT/tables_<R>.h5, the tables of the vector space on R ranks
    mpirun -n R python3 tests/dolfin_layout_linked_worker.py OUT link [TABLES]
        writes the tables (or takes TABLES, another run's) and links them into
        OUT/linked.h5, and reports on rank zero whether dolfin reads the Function back --
        also from a copy of the two files in another folder -- and whether tables of
        another element, of another run's numbering, or in the Function's own file, are refused

Another run's numbering is another numbering even on as many ranks: dolfin partitions the
mesh anew, and differently, each run.

Rank zero prints which way h5py wrote (`mpio`: every rank, MPI-IO; `gather`: rank zero).
"""
import os
import shutil
import sys

import dolfin as df
import numpy as np

from femtools import dolfin_layout, hdf5_rows
from dolfin_layout_worker import Periodic


def spaces(comm):
    mesh = df.UnitCubeMesh(comm, 5, 4, 6)
    V1 = df.FunctionSpace(mesh, "P", 1)
    V2 = df.VectorFunctionSpace(mesh, "P", 2, constrained_domain=Periodic())
    return V1, V2


def refused(f):
    try:
        f()
    except ValueError as e:
        return str(e)
    return None


def main(out, mode, foreign=None):
    comm = df.MPI.comm_world
    V1, V2 = spaces(comm)
    lines = ["h5py: " + ("mpio" if hdf5_rows.mpio(comm) else "gather")]
    if mode == "tables":
        dolfin_layout.write_dofs(os.path.join(out, "tables_{}.h5".format(comm.size)), V2, "V")
    else:
        u = df.interpolate(df.Expression(("cos(2*pi*x[0])", "x[2]*x[2]", "sin(2*pi*x[1])"),
                                         degree=2), V2)
        p = df.interpolate(df.Expression("x[0]", degree=1), V1)
        linked = os.path.join(out, "linked.h5")
        tables = "tables.h5"
        dolfin_layout.write_dofs(os.path.join(out, tables), V2, "V")
        dolfin_layout.write_function(linked, u, "u", "w", dofs=(tables, "V"))
        dolfin_layout.write_function(linked, u, "previous", dofs=(tables, "V"))
        moved = os.path.join(out, "moved")
        if comm.rank == 0:
            os.makedirs(moved)
            for f in (linked, os.path.join(out, tables)):
                shutil.copy(f, moved)
        comm.Barrier()
        for name, where in (("read back", linked),
                            ("read back moved", os.path.join(moved, "linked.h5"))):
            same = True
            for group in ("u", "previous"):
                back = df.Function(V2)
                with df.HDF5File(comm, where, "r") as h:
                    h.read(back, group)
                same &= np.array_equal(back.vector().get_local(), u.vector().get_local())
            same = comm.allreduce(int(same), op=__import__("mpi4py").MPI.MIN)
            lines.append("{}: {}".format(name, "same" if same else "DIFFERENT"))
        checks = [("other element", "another element", os.path.join(out, "p.h5"), p, tables),
                  ("own file", "another file", os.path.join(out, tables), u, tables)]
        if foreign is not None:
            checks.append(("numbering", "another numbering", os.path.join(out, "f.h5"), u,
                           foreign))
        for label, words, path, f, given in checks:
            why = refused(lambda: dolfin_layout.write_function(path, f, f.name(), "w",
                                                               dofs=(given, "V")))
            lines.append("{}: {}".format(label, "refused" if why and words in why
                                         else str(why)))
    if comm.rank == 0:
        print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
