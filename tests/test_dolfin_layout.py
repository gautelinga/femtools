"""`femtools.dolfin_layout`: a mesh and Functions written compressed, in dolfin's own layout.

    python3 -m pytest tests/test_dolfin_layout.py

dolfin, mpi4py and `mpirun`. dolfin's HDF5File cannot compress, which is what
`dolfin_layout` is for. The worker writes a mesh, a P2 vector on a periodic space and a P1 scalar
both ways on 1 and 3 ranks: the files must hold the same datasets, dtypes, values and
attributes, ours compressed, and dolfin must read ours back to the same values.
"""
import os
import sys

import pytest

pytest.importorskip("dolfin")
pytest.importorskip("h5py")
pytestmark = [pytest.mark.dolfin]

from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dolfin_layout_worker.py")


#: h5py's builds, as Debian picks them: the MPI one writes through MPI-IO, every rank its
#: own block; the serial one through rank zero (the fallback)
H5PY = {"mpio": {"H5PY_ALWAYS_USE_MPI": "1"}, "gather": {"H5PY_NEVER_USE_MPI": "1"}}
LINKED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dolfin_layout_linked_worker.py")


@pytest.mark.mpi
@pytest.mark.parametrize("ranks,h5py_build", [(1, None), (3, "mpio"), (3, "gather")])
def test_the_same_file_as_dolfin_writes_but_compressed(ranks, h5py_build, tmp_path):
    """the file dolfin writes, compressed, on one rank and on three either way h5py
    writes, and dolfin reads it back"""
    proc = support.run([sys.executable, WORKER, str(tmp_path)], ranks, timeout=600,
                       env=H5PY.get(h5py_build))
    lines = support.lines(proc.stdout)
    assert lines and all(line.endswith(": same") for line in lines), proc.stdout
    assert len(lines) == 2 + 3 + 1 + 2 * 5, proc.stdout     # groups and datasets


@pytest.mark.mpi
def test_tables_written_once_and_linked(tmp_path):
    """`write_dofs` once, `write_function(..., dofs=...)` per vector: dolfin reads each
    group back exactly, from a copied folder too; tables of another element, of
    another number of ranks, or in the Function's own file, are refused. (Both ways
    h5py writes are `write_ordered`'s, tested in test_compress.)"""
    def run(ranks, *args):
        proc = support.run([sys.executable, LINKED, str(tmp_path)] + list(args), ranks,
                           timeout=600)
        return support.lines(proc.stdout)[1:]
    run(1, "tables")
    assert run(3, "link", "tables_1.h5") == [
        "read back: same", "read back moved: same", "other element: refused",
        "own file: refused", "numbering: refused"]


@pytest.mark.parametrize("dim", [2, 3])
def test_a_mesh_from_arrays_is_dolfins_serial_mesh(dim, tmp_path):
    """`write_mesh_arrays`: the datasets, dtypes, values and attributes dolfin's HDF5File
    writes for the same mesh on one rank, compressed, and dolfin reads it back."""
    import dolfin as df
    import h5py
    import numpy as np
    from mpi4py import MPI
    from femtools.dolfin_layout import write_mesh_arrays
    mesh = (df.UnitSquareMesh(MPI.COMM_SELF, 3, 2) if dim == 2
            else df.UnitCubeMesh(MPI.COMM_SELF, 2, 2, 1))
    a, b = str(tmp_path / "dolfin.h5"), str(tmp_path / "ours.h5")
    with df.HDF5File(MPI.COMM_SELF, a, "w") as h:
        h.write(mesh, "mesh")
    write_mesh_arrays(b, mesh.coordinates(), mesh.cells())
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        names = []
        fa.visit(names.append)
        assert sorted(names) == sorted(_names(fb))
        for n in names:
            x, y = fa[n], fb[n]
            assert {k: np.asarray(v).tobytes() for k, v in x.attrs.items()} == \
                {k: np.asarray(v).tobytes() for k, v in y.attrs.items()}, n
            if isinstance(x, h5py.Dataset):
                assert x.dtype == y.dtype and x[()].tobytes() == y[()].tobytes(), n
                assert y.compression == "gzip" and y.shuffle, n
    m2 = df.Mesh(MPI.COMM_SELF)
    with df.HDF5File(MPI.COMM_SELF, b, "r") as h:
        h.read(m2, "mesh", False)
    assert np.array_equal(m2.coordinates(), mesh.coordinates())
    assert np.array_equal(m2.cells(), mesh.cells())


def _names(h5):
    out = []
    h5.visit(out.append)
    return out
