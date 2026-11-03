"""`femtools.gradient`: the weighted gradient matrix gives the exact derivatives of a
linear p on P1 in 2D and 3D, weighs the cells around a node by 1/|K| (checked by hand
on unequal cells), and gives the same values on 2 ranks as on one."""
import os
import sys

import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytest.importorskip("cppimport")
pytestmark = [pytest.mark.dolfin]

from femtools.gradient import (compute_weighted_gradient_matrix,  # noqa: E402
                               weighted_gradient_matrix)
from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradient_worker.py")


def wgm(V, Q, i):
    DG = df.FunctionSpace(V.mesh(), "DG", 0)
    G = df.assemble(df.TrialFunction(DG) * df.TestFunction(V) * df.dx)
    dP = df.assemble(df.TrialFunction(Q).dx(i) * df.TestFunction(DG) * df.dx)
    return compute_weighted_gradient_matrix(G, dP, df.Function(DG))


def apply(W, p, V):
    out = df.Function(V)
    W.mult(p.vector(), out.vector())
    return out.vector().get_local()


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_p_exact_at_every_node(dim):
    mesh = df.UnitSquareMesh(5, 4) if dim == 2 else df.UnitCubeMesh(3, 4, 2)
    Q = df.FunctionSpace(mesh, "CG", 1)
    slope = [0.7, -1.3, 2.1][:dim]
    p = df.interpolate(df.Expression(" + ".join("{}*x[{}]".format(c, k)
                                                for k, c in enumerate(slope)) + " + 0.4",
                                     degree=1), Q)
    for i in range(dim):
        got = apply(wgm(Q, Q, i), p, Q)
        assert np.allclose(got, slope[i], rtol=0, atol=1e-12), (i, got)


def test_scalar_index_entry_point():
    mesh = df.UnitSquareMesh(4, 4)
    Q = df.FunctionSpace(mesh, "CG", 1)
    p = df.interpolate(df.Expression("3*x[0] - x[1]", degree=1), Q)
    assert np.allclose(apply(weighted_gradient_matrix(mesh, 1), p, Q), -1.0, atol=1e-12)


def test_one_over_volume_weighting_by_hand():
    mesh = df.UnitSquareMesh(2, 2)
    x = mesh.coordinates()
    centre = int(np.flatnonzero(np.all(np.isclose(x, 0.5), axis=1))[0])
    x[centre] = [0.38, 0.31]                      # unequal cells around it
    Q = df.FunctionSpace(mesh, "CG", 1)
    rng = np.random.default_rng(3)
    nodal = rng.random(mesh.num_vertices())
    p = df.Function(Q)
    p.vector()[:] = nodal[df.dof_to_vertex_map(Q)]
    got = [apply(wgm(Q, Q, i), p, Q)[df.vertex_to_dof_map(Q)[centre]] for i in range(2)]
    num, den = np.zeros(2), 0.0
    for cell in mesh.cells():
        if centre not in cell:
            continue
        a, b, c = x[cell]
        J = np.array([b - a, c - a]).T
        area = 0.5 * abs(np.linalg.det(J))
        grad = np.linalg.solve(J.T, nodal[cell[1:]] - nodal[cell[0]])
        num += grad / area
        den += 1.0 / area
    assert np.allclose(got, num / den, rtol=0, atol=1e-12), (got, num / den)
    # not the volume-weighted mean, which this mesh tells apart
    vol = [(0.5 * abs(np.linalg.det(np.array([x[c[1]] - x[c[0]], x[c[2]] - x[c[0]]]).T)),
            np.linalg.solve(np.array([x[c[1]] - x[c[0]], x[c[2]] - x[c[0]]]),
                            nodal[c[1:]] - nodal[c[0]])) for c in mesh.cells() if centre in c]
    weighted = sum(v * g for v, g in vol) / sum(v for v, _ in vol)
    assert not np.allclose(got, weighted, atol=1e-6)


@pytest.mark.mpi
def test_two_ranks_match_serial(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {"PYTHONPATH": root}
    support.run([sys.executable, WORKER, str(tmp_path / "one.npy")], 1, env=env, timeout=600)
    support.run([sys.executable, WORKER, str(tmp_path / "two.npy")], 2, env=env, timeout=600)
    a, b = np.load(tmp_path / "one.npy"), np.load(tmp_path / "two.npy")
    assert a.shape == b.shape
    a, b = a[np.lexsort(a[:, :3].T)], b[np.lexsort(b[:, :3].T)]
    assert np.allclose(a[:, :3], b[:, :3], rtol=0, atol=1e-14)
    assert np.allclose(a[:, 3:], b[:, 3:], rtol=0, atol=1e-12)
