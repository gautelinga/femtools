"""`femtools.probes`: Probes and GradProbes on a P2 field (scalar and vector) on a
UnitCubeMesh give the function's own value and gradient at the probe points, every
gradient component and every gradient snapshot; a Probes built from located GradProbes
gives the same values."""
import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytest.importorskip("cppimport")
pytestmark = [pytest.mark.dolfin]

from femtools.probes import GradProbes, Probes  # noqa: E402

POINTS = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.77, 0.13, 0.91], [0.33, 0.9, 0.05]])


def fields():
    mesh = df.UnitCubeMesh(4, 4, 4)
    u = df.interpolate(df.Expression("x[0]*x[0] + 2*x[1]*x[2] - x[2]", degree=2),
                       df.FunctionSpace(mesh, "CG", 2))
    v = df.interpolate(df.Expression(("x[0]*x[1]", "x[2]*x[2]", "x[0] - x[1]*x[1]"),
                                     degree=2), df.VectorFunctionSpace(mesh, "CG", 2))
    return mesh, u, v


def own_gradient(f, mesh, x):
    """f's gradient at x, from the DG1 projection of grad f on its cell (exact for P2)."""
    W = df.TensorFunctionSpace(mesh, "DG", 1) if f.value_rank() else \
        df.VectorFunctionSpace(mesh, "DG", 1)
    g = df.project(df.grad(f), W, solver_type="lu")
    return np.array(g(*x))


@pytest.mark.parametrize("which", ["scalar", "vector"])
def test_probes_match_the_function(which):
    mesh, u, v = fields()
    f = u if which == "scalar" else v
    p = Probes(POINTS.flatten(), f.function_space())
    p(f)
    p(f)
    got = p.array(N=1)
    want = np.array([np.atleast_1d(f(*x)) for x in POINTS]).squeeze()
    assert p.number_of_evaluations() == 2
    assert np.allclose(got, want, rtol=0, atol=1e-13), (got, want)


@pytest.mark.parametrize("which", ["scalar", "vector"])
def test_grad_probes_match_the_gradient(which):
    mesh, u, v = fields()
    f = u if which == "scalar" else v
    gp = GradProbes(POINTS.flatten(), f.function_space())
    gp(f)
    gp.grad(f)
    values = gp.array(N=0)
    grads = gp.array_grad(N=0)
    want_v = np.array([np.atleast_1d(f(*x)) for x in POINTS]).squeeze()
    want_g = np.array([own_gradient(f, mesh, x) for x in POINTS])
    assert np.allclose(values, want_v, rtol=0, atol=1e-13)
    assert np.allclose(grads, want_g, rtol=0, atol=1e-10), (grads, want_g)


def test_probes_from_located_grad_probes():
    mesh, u, v = fields()
    gp = GradProbes(POINTS.flatten(), u.function_space())
    p = Probes(gp, u.function_space())
    p(u)
    want = np.array([u(*x) for x in POINTS])
    assert np.allclose(p.array(N=0), want, rtol=0, atol=1e-13)


@pytest.mark.parametrize("which", ["scalar", "vector"])
def test_every_gradient_snapshot_holds_every_component(which):
    mesh, u, v = fields()
    f = u if which == "scalar" else v
    gp = GradProbes(POINTS.flatten(), f.function_space())
    gp(f)                       # a value-only evaluation between the gradient ones
    gp.grad(f)
    gp.grad(f)
    want = np.array([own_gradient(f, mesh, x) for x in POINTS]).reshape(len(POINTS), -1)
    assert gp.number_of_grad_evaluations() == 2
    grads = gp.array_grad()
    assert grads.shape == (len(POINTS), want.shape[1], 2)
    for n in range(2):
        assert np.allclose(grads[:, :, n], want, rtol=0, atol=1e-10), (n, grads[:, :, n], want)
    for i, (index, probe) in enumerate(gp):
        snap = np.array(probe.get_probe_grad_at_snapshot(1))
        assert snap.shape == (want.shape[1],)
        assert np.allclose(snap, want[index], rtol=0, atol=1e-10)


def square():
    mesh = df.UnitSquareMesh(8, 8)
    V = df.FunctionSpace(mesh, "CG", 1)
    return mesh, V, df.interpolate(df.Expression("x[0] + 2*x[1]", degree=1), V)


def test_a_restarted_snapshot_counts():
    """`restart_probe` is a snapshot of its own: the next evaluation is the next one."""
    from femtools.probes import Probe
    _, V, u = square()
    p = Probe(np.array([0.3, 0.4]), V)
    p.restart_probe(np.array([7.0]))
    p(u)
    assert p.number_of_evaluations() == 2
    assert np.allclose(p.get_probe_sub(0), [7.0, 1.1])


def test_one_component_is_that_component():
    mesh = df.UnitSquareMesh(4, 4)
    W = df.VectorFunctionSpace(mesh, "CG", 1)
    w = df.interpolate(df.Expression(("1.0", "2.0"), degree=0), W)
    q = Probes(np.array([0.3, 0.4, 0.6, 0.6]), W)
    q(w)
    assert np.allclose(q.array(N=0, component=1), [2.0, 2.0])
    assert np.allclose(q.array(component=0), [1.0, 1.0])


def test_points_from_a_strided_array_and_outside_the_mesh():
    """A column slice of an array is read as the points it holds; a point outside the
    mesh is held by no rank and reads 0."""
    _, V, u = square()
    pts = np.array([[0.1, 0.2, 99.0], [0.7, 0.8, 99.0], [2.0, 2.0, 99.0]])[:, :2]
    r = Probes(pts, V)
    r(u)
    assert r.local_size() == 2 and r.get_total_number_probes() == 3
    assert np.allclose(r.array(N=0), [0.5, 2.3, 0.0])


def test_out_of_range_is_refused():
    """A snapshot, component or local probe past the end raises; positions are not
    added once there are snapshots the new probes would lack."""
    _, V, u = square()
    p = Probes(np.array([0.3, 0.4]), V)
    p(u)
    p(u)
    with pytest.raises(IndexError):
        p.array(N=2)
    with pytest.raises(IndexError):
        p.array(component=1)
    for call in (lambda: p.get_probe(1), lambda: p.erase_snapshot(2),
                 lambda: p.get_probes_component_and_snapshot(0, 5),
                 lambda: p.add_positions(np.array([0.9, 0.9]), V)):
        with pytest.raises(RuntimeError):
            call()


def test_erasing_a_gradient_snapshot_keeps_the_histories_aligned():
    _, V, u = square()
    g = GradProbes(np.array([0.3, 0.4]), V)
    g.grad(u)
    g.grad(u)
    g.erase_snapshot(0)
    assert g.number_of_evaluations() == g.number_of_grad_evaluations() == 1
    assert g.array_grad(N=0).shape == (2,)
    g.clear()
    assert g.number_of_evaluations() == g.number_of_grad_evaluations() == 0


BOUNDARY = r"""
import numpy as np, dolfin as df
from mpi4py import MPI
from femtools.probes import Probes
comm = MPI.COMM_WORLD
mesh = df.UnitSquareMesh(8, 8)
V = df.FunctionSpace(mesh, "CG", 1)
u = df.interpolate(df.Expression("x[0] + 2*x[1]", degree=1), V)
pts = np.array([[i / 8, j / 8] for i in range(9) for j in range(9)])   # every vertex
p = Probes(pts.flatten(), V)
p(u)
n = comm.allreduce(p.local_size())
a = p.array(N=0)
if comm.rank == 0:
    print("HELD", n, "EXACT", bool(np.allclose(a, pts[:, 0] + 2 * pts[:, 1])))
"""


@pytest.mark.mpi
def test_a_point_on_a_partition_boundary_is_held_once():
    """Every vertex of the mesh as a point, on three ranks: each is held by one rank
    (the lowest finding it), and gathered to its value."""
    import os
    import sys
    from tests import support
    proc = support.run([sys.executable, "-c", BOUNDARY], 3,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert "HELD 81 EXACT True" in proc.stdout, proc.stdout[-2000:]


def test_an_array_before_any_evaluation_is_empty():
    _, V, _ = square()
    p = Probes(np.array([0.3, 0.4, 0.5, 0.5]), V)
    assert np.asarray(p.array()).size == 0
