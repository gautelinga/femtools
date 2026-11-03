"""`femtools.boundaries`: a boundary facet file written and read back.

    python3 -m pytest tests/test_boundaries.py

numpy and h5py; the three-rank write needs mpirun and mpi4py.
"""
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from femtools import boundaries  # noqa: E402
from tests import support  # noqa: E402


def case(dim=3, n=500, nf=800, seed=2):
    rng = np.random.default_rng(seed)
    return (rng.random((n, dim)), rng.integers(0, n, (nf, dim)),
            rng.integers(1, 40, nf))


@pytest.mark.parametrize("dim", [3, 2])
def test_written_is_read_back(tmp_path, dim):
    """Points, facets and markers come back as given, in meshio's layout: float64
    points, int32 facets, int64 markers, every dataset compressed; triangles in 3D,
    segments in 2D."""
    points, facets, markers = case(dim)
    path = str(tmp_path / "b.xdmf")
    assert boundaries.write(path, points, facets, markers) == (len(facets), len(points))
    got = boundaries.read(path)
    for a, b in zip(got, (points, facets, markers)):
        assert np.array_equal(a, b)
    assert [a.dtype for a in got] == [np.float64, np.int32, np.int64]
    with h5py.File(str(tmp_path / "b.h5"), "r") as h5:
        assert sorted(h5) == ["facets", "markers", "points"]
        assert all(h5[k].compression == "gzip" and h5[k].shuffle for k in h5)
    text = open(path).read()
    assert ('TopologyType="%s"' % ("Triangle" if dim == 3 else "Polyline")) in text
    assert '<Attribute Name="markers" AttributeType="Scalar" Center="Cell">' in text
    assert 'Dimensions="%d"' % len(markers) in text


def test_facets_and_markers_of_different_lengths_are_refused(tmp_path):
    points, facets, markers = case()
    with pytest.raises(ValueError, match="facets and"):
        boundaries.write(str(tmp_path / "b.xdmf"), points, facets, markers[:-1])


WORKER = """
import sys
import numpy as np
from mpi4py import MPI
from femtools import boundaries
from femtools import hdf5_rows
comm = MPI.COMM_WORLD
hdf5_rows.MESSAGE = 4 << 10
rng = np.random.default_rng(2)
n, nf = 500, 800
points, facets, markers = rng.random((n, 3)), rng.integers(0, n, (nf, 3)), rng.integers(1, 40, nf)
# uneven blocks, rank 1 holding no points and rank 0 no facets
p = [0, 200, 200, n][comm.rank:comm.rank + 2]
f = [0, 0, 300, nf][comm.rank:comm.rank + 2]
if sys.argv[2] == "empty":      # nothing marked anywhere
    p = f = [0, 0]
mine = points[p[0]:p[1]]
if comm.rank == 1 and sys.argv[2] == "blocks":
    mine = np.zeros(0)          # an empty block passed 1-D
boundaries.write(sys.argv[1], mine, facets[f[0]:f[1]], markers[f[0]:f[1]], comm)
"""


@pytest.mark.mpi
def test_three_ranks_write_their_blocks_in_rank_order(tmp_path):
    """Three ranks, each passing an uneven block of each array (one passing none, as a
    1-D empty array), write what one rank writes from the whole arrays."""
    pytest.importorskip("mpi4py")
    points, facets, markers = case()
    one, three = str(tmp_path / "one.xdmf"), str(tmp_path / "three.xdmf")
    boundaries.write(one, points, facets, markers)
    support.run([sys.executable, "-c", WORKER, three, "blocks"], 3, timeout=300,
                env=dict(PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    for a, b in zip(boundaries.read(one), boundaries.read(three)):
        assert np.array_equal(a, b) and a.dtype == b.dtype
    with h5py.File(str(tmp_path / "three.h5"), "r") as h5:
        assert all(h5[k].compression == "gzip" for k in h5)
    assert open(one).read() == open(three).read().replace("three.h5", "one.h5")


@pytest.mark.mpi
@pytest.mark.parametrize("h5py_build", ["mpio", "gather"])
def test_three_ranks_write_nothing(tmp_path, h5py_build):
    """Nothing marked on any of three ranks: the file is the empty one one rank writes,
    through MPI-IO (whose zero-size datasets have no storage to write to) or rank zero."""
    pytest.importorskip("mpi4py")
    one, three = str(tmp_path / "one.xdmf"), str(tmp_path / "three.xdmf")
    boundaries.write(one, np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64))
    env = {"mpio": {"H5PY_ALWAYS_USE_MPI": "1"}, "gather": {"H5PY_NEVER_USE_MPI": "1"}}
    support.run([sys.executable, "-c", WORKER, three, "empty"], 3, timeout=300,
                env=dict(env[h5py_build],
                         PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    for a, b in zip(boundaries.read(one), boundaries.read(three)):
        assert a.shape == b.shape and len(a) == 0 and a.dtype == b.dtype
    assert open(one).read() == open(three).read().replace("three.h5", "one.h5")
