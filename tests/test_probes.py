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
