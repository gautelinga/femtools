"""`femtools.boundaries.BoundaryMarkers`: a facet MeshFunction written as a boundary facet
file and read back, on one rank and three; partrac reads the file as its walls.

    python3 -m pytest tests/test_boundary_markers.py
    python3 -m pytest -m slow tests/test_boundary_markers.py

Needs dolfin; the three-rank test mpirun, the partrac test a built partrac.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = pytest.mark.dolfin

from femtools import boundaries  # noqa: E402
from femtools.boundaries import BoundaryMarkers  # noqa: E402
from tests import support  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTRAC = os.environ.get("PARTRAC_BIN", os.path.expanduser("~/git/partrac/build/bin"))

MARK = """
import dolfin as df
from femtools.boundaries import BoundaryMarkers


def marked(dim, n):
    '''A unit box, its faces marked 1..2*dim (x lo, x hi, ...), x = 0.5 marked 9 (interior
    facets, which are not written), the y = 0 face left 0 where x > 0.5.'''
    mesh = df.UnitSquareMesh(n, n) if dim == 2 else df.UnitCubeMesh(n, n, n)
    b = BoundaryMarkers(mesh)
    for k in range(2 * dim):
        b.mark(df.CompiledSubDomain("on_boundary && near(x[%d], %d)" % (k // 2, k % 2)), k + 1)
    b.mark(df.CompiledSubDomain("near(x[0], 0.5)"), 9)
    b.mark(df.CompiledSubDomain("on_boundary && near(x[1], 0) && x[0] > 0.5"), 0)
    return mesh, b


def expected(b):
    '''b's values on the exterior facets, 0 elsewhere.'''
    mesh = b.mesh()
    on = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    df.DomainBoundary().mark(on, 1)
    return b.array() * (on.array() == 1)


def bc_dofs(b, value):
    V = df.FunctionSpace(b.mesh(), "CG", 2)
    return sorted(df.DirichletBC(V, df.Constant(1.0), b, value).get_boundary_values())
"""
exec(MARK)


def by_position(path):
    """{sorted vertex coordinates of a facet: marker} of a facet file."""
    points, facets, markers = boundaries.read(path)
    return {tuple(sorted(map(tuple, points[f]))): int(m) for f, m in zip(facets, markers)}


@pytest.mark.parametrize("dim", [3, 2])
def test_written_is_read_back(tmp_path, dim):
    """The exterior facets with nonzero markers are written, on the vertices they use, in
    global order; read back they give those markers and 0 on every other facet. It is a
    MeshFunctionSizet throughout."""
    mesh, b = marked(dim, 4)
    path = str(tmp_path / "b.xdmf")
    nf, npts = b.write(path)
    want = expected(b)
    points, facets, markers = boundaries.read(path)
    assert nf == (want != 0).sum() == len(markers) and not (markers == 0).any()
    assert 9 not in markers and set(markers) == set(range(1, 2 * dim + 1))
    used = np.unique(facets)
    assert np.array_equal(used, np.arange(npts))
    x = mesh.coordinates()
    vertices = np.unique(np.array(mesh.topology()(dim - 1, 0)(), dtype=np.int64)
                         .reshape(-1, dim)[want != 0])
    assert np.array_equal(points, x[vertices])
    back = BoundaryMarkers.read(path, mesh)
    assert isinstance(back, df.cpp.mesh.MeshFunctionSizet) and back.dim() == dim - 1
    assert np.array_equal(back.array(), want)


def test_a_dirichlet_bc_and_ds_on_the_read_markers_are_the_originals(tmp_path):
    """A DirichletBC on a written-then-read marker constrains the same dofs, and ds of
    each marker measures the same area."""
    mesh, b = marked(3, 4)
    path = str(tmp_path / "b.xdmf")
    b.write(path)
    back = BoundaryMarkers.read(path, mesh)
    for value in (1, 3, 4):
        assert bc_dofs(back, value) == bc_dofs(b, value) != []
    ds0 = df.Measure("ds", domain=mesh, subdomain_data=b)
    ds1 = df.Measure("ds", domain=mesh, subdomain_data=back)
    one = df.Constant(1.0)
    for value in range(1, 7):
        assert df.assemble(one * ds1(value)) == pytest.approx(df.assemble(one * ds0(value)),
                                                              abs=1e-14)


def test_a_facet_off_the_mesh_boundary_is_refused(tmp_path):
    """A file facet that is no exterior facet of the mesh."""
    mesh, b = marked(3, 4)
    path = str(tmp_path / "b.xdmf")
    b.write(path)
    points, facets, markers = boundaries.read(path)
    boundaries.write(path, points * 0.5, facets, markers)
    with pytest.raises(ValueError, match="match an exterior facet"):
        BoundaryMarkers.read(path, mesh)


def test_partrac_reads_the_file_as_its_walls(tmp_path):
    """A P2 case at rest on its faces, `wall_facets` the written file: partrac's `interpol`
    matches every facet the markers select to an exterior facet of its mesh."""
    interpol = os.path.join(PARTRAC, "interpol")
    if not os.access(interpol, os.X_OK):
        pytest.skip("no partrac at " + PARTRAC + " (PARTRAC_BIN)")
    mesh, b = marked(3, 3)
    b.write(str(tmp_path / "walls.xdmf"))
    _, _, markers = boundaries.read(str(tmp_path / "walls.xdmf"))
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u = df.interpolate(df.Expression(("x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])", "0", "0"),
                                     degree=2), V)
    p = df.Function(df.FunctionSpace(mesh, "CG", 1))
    with df.HDF5File(mesh.mpi_comm(), str(tmp_path / "mesh.h5"), "w") as f:
        f.write(mesh, "mesh")
    with df.HDF5File(mesh.mpi_comm(), str(tmp_path / "up_0.h5"), "w") as f:
        f.write(u, "u")
        f.write(p, "p")
    (tmp_path / "timestamps.dat").write_text("0 up_0.h5\n1 up_0.h5\n")
    n_ext = 6 * 2 * 3 * 3
    for listed, n in (("1:6", len(markers)), ("5,6", np.isin(markers, (5, 6)).sum())):
        (tmp_path / "dolfin_params.dat").write_text(
            "timestamps=timestamps.dat\nvelocity_space=P2\npressure_space=P1\nmesh=mesh.h5\n"
            "periodic_x=false\nperiodic_y=false\nperiodic_z=false\nwall_upgrade=true\n"
            "wall_facets=walls.xdmf\nwall_markers=%s\n" % listed)
        proc = subprocess.run([interpol, "dolfin_params.dat", "mode=tet", "Nrw=10",
                               "int_order=1", "t0=0", "random=false", "seed=1"],
                              cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
                              env=dict(support.clean_env(), OMP_NUM_THREADS="2"))
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        assert "Wall facets: %d of %d exterior facets" % (n, n_ext) in proc.stdout


WORKER = MARK + """
import sys
import numpy as np
from femtools import boundaries
one, three = sys.argv[1:3]
for ghosts in ("none", "shared_vertex"):
    df.parameters["ghost_mode"] = ghosts
    mesh, b = marked(3, 5)
    if ghosts == "none":
        b.write(three)
    back = BoundaryMarkers.read(one, mesh)
    assert np.array_equal(back.array(), expected(b)), ghosts
    for value in (1, 4):
        assert bc_dofs(back, value) == bc_dofs(b, value), (ghosts, value)
"""


@pytest.mark.mpi
def test_three_ranks_write_and_read_the_one_rank_file(tmp_path):
    """Three ranks write the facets and points one rank writes (points in global order,
    facets in rank order) and read the one-rank file back to the markers and the
    DirichletBC dofs of the original, with and without ghost cells."""
    mesh, b = marked(3, 5)
    one, three = str(tmp_path / "one.xdmf"), str(tmp_path / "three.xdmf")
    b.write(one)
    support.run([sys.executable, "-c", WORKER, one, three], 3, timeout=600,
                env=dict(PYTHONPATH=ROOT))
    assert np.array_equal(boundaries.read(one)[0], boundaries.read(three)[0])
    assert by_position(one) == by_position(three)
    assert np.array_equal(BoundaryMarkers.read(three, mesh).array(), expected(b))
