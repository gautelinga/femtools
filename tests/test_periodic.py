"""`femtools.periodic`: the periodic box, as a Python SubDomain and compiled, and the
compiled bulk accessors.

    python3 -m pytest tests/test_periodic.py

dolfin, a compiler and `libdolfin-dev`. The masters are the points on a lower periodic
face and on no upper one, and `map` takes an upper face down by the box; the two classes
must give the same function spaces, dof for dof, on 2D and 3D boxes, at the origin or
not, periodic on some axes or all.
"""
import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from femtools.periodic import PeriodicBox, cell_dofs, connectivity, periodic_box  # noqa: E402


@pytest.mark.parametrize("make", [PeriodicBox, periodic_box])
def test_the_box_identifies_only_the_axes_it_is_given(make):
    """Walled in z: a point on a plate takes no part, so a point on it and on a lower x
    face is still a master and one on a plate alone is neither; a corner of the periodic
    axes maps to the lower corner in one step, z left alone."""
    box = make(np.ones(3), 1e-10, (True, True, False))
    checks = [
        ((0.0, 0.5, 0.5), True, "on the lower x face"),
        ((1.0, 0.5, 0.5), False, "on the upper x face"),
        ((0.0, 0.5, 0.0), True, "on the lower x face and on a plate"),
        ((0.5, 0.5, 0.0), False, "on a plate alone"),
        ((0.5, 0.5, 1.0), False, "on the other plate alone"),
        ((0.0, 1.0, 0.5), False, "on the lower x face and the upper y face"),
    ]
    for point, expected, what in checks:
        assert box.inside(np.array(point), True) == expected, (what, point)
    y = np.zeros(3)
    box.map(np.array([1.0, 1.0, 0.7]), y)
    assert np.allclose(y, [0.0, 0.0, 0.7]), y
    assert box.periodic == [True, True, False] and box.lower == [0.0, 0.0, 0.0]


def test_a_walled_axis_leaves_its_dofs_alone():
    """Counted independently: a P1 space on a box mesh periodic on some axes has one dof
    per vertex that is not on an upper periodic face."""
    n, L = 6, np.ones(3)
    mesh = df.BoxMesh(df.Point(0., 0., 0.), df.Point(*L), n, n, n)
    for periodic in ((True, True, False), (True, True, True), (False, False, False)):
        V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=periodic_box(L, 1e-10, periodic))
        x = mesh.coordinates()
        upper = np.zeros(len(x), bool)
        for a in range(3):
            if periodic[a]:
                upper |= np.abs(x[:, a] - L[a]) < 1e-10
        assert V.dim() == int((~upper).sum()), periodic


CASES = [
    # (mesh, lower, upper, periodic): a centred rectangle, a 2D box off the origin, a
    # 3D box off the origin periodic in x and y, and the unit cube periodic on all three
    (lambda: df.RectangleMesh(df.Point(-1.5, -0.5), df.Point(1.5, 0.5), 12, 5),
     [-1.5, -0.5], [1.5, 0.5], [True, False]),
    (lambda: df.RectangleMesh(df.Point(0.2, 0.1), df.Point(1.3, 0.9), 7, 6),
     [0.2, 0.1], [1.3, 0.9], [True, True]),
    (lambda: df.BoxMesh(df.Point(-0.5, 0.25, 0.), df.Point(0.5, 1.25, 2.), 4, 4, 6),
     [-0.5, 0.25, 0.], [0.5, 1.25, 2.], [True, True, False]),
    (lambda: df.UnitCubeMesh(4, 3, 5), [0., 0., 0.], [1., 1., 1.], [True, True, True]),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_the_python_and_the_compiled_box_give_the_same_spaces(case):
    """Same dimension and the same dof coordinates, in order, for P1 and vector P2, and
    fewer dofs than the unconstrained space."""
    make, lower, upper, periodic = CASES[case]
    mesh = make()
    for el in (("CG", 1), ("CG", 2)):
        spaces = [df.VectorFunctionSpace(mesh, *el, constrained_domain=box(
            upper, 1e-10, periodic, lower, map_tol=1e-10)) for box in (PeriodicBox, periodic_box)]
        a, b = spaces
        assert a.dim() == b.dim() < df.VectorFunctionSpace(mesh, *el).dim()
        assert np.array_equal(a.tabulate_dof_coordinates(), b.tabulate_dof_coordinates())
        assert np.array_equal(a.dofmap().dofs(), b.dofmap().dofs())


def test_cell_dofs_are_the_dofmaps():
    """`cell_dofs` is `dofmap.cell_dofs` of each listed cell, in one call."""
    mesh = df.UnitCubeMesh(3, 2, 2)
    V = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement("CG", mesh.ufl_cell(), 2), df.FiniteElement("CG", mesh.ufl_cell(), 1)]),
        constrained_domain=periodic_box(np.ones(3), 1e-10))
    cells = np.array([0, 5, 2, mesh.num_cells() - 1])
    got = cell_dofs(V.dofmap(), cells, 34)
    assert got.dtype == np.int64
    assert np.array_equal(got, np.array([V.dofmap().cell_dofs(int(c)) for c in cells]))


def test_connectivity_is_dolfins_as_an_array():
    """`connectivity` is dolfin's `topology()(d0, d1)()`, which Python hands back as a
    list, as one (entities, k) array; it builds what is not built yet."""
    for d0, d1 in ((3, 2), (2, 0), (2, 1), (1, 0), (3, 1), (3, 0)):
        mesh = df.UnitCubeMesh(3, 2, 2)
        a = connectivity(mesh, d0, d1)
        assert np.array_equal(a, np.asarray(mesh.topology()(d0, d1)()).reshape(a.shape))
    with pytest.raises(ValueError):
        connectivity(mesh, 0, 3)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_faces_congruent_on_a_2d_mesh(n):
    """2D coordinates are padded, not reshaped as 3D rows."""
    from femtools.periodic import faces_congruent
    assert faces_congruent(df.UnitSquareMesh(n, n), [1.0, 1.0], [0, 1], edges=True)[0]


def test_a_box_off_the_origin_needs_its_lower_corner():
    """A centred box is compared at its own faces with `lower`, and refused without it
    rather than compared with the plane through its middle."""
    from femtools.periodic import faces_congruent
    mesh = df.BoxMesh(df.Point(-.5, -1, -.5), df.Point(.5, 1, .5), 3, 4, 3)
    with pytest.raises(ValueError, match="lower corner"):
        faces_congruent(mesh, [.5, 1, .5], [0, 1, 2])
    assert faces_congruent(mesh, [.5, 1, .5], [0, 1, 2], edges=True, lower=[-.5, -1, -.5])[0]
    # a face moved off its partner is seen
    x = mesh.coordinates()
    x[np.abs(x[:, 0] + .5) < 1e-12, 1] += 1e-3
    assert not faces_congruent(mesh, [.5, 1, .5], [0], lower=[-.5, -1, -.5])[0]


@pytest.mark.parametrize("dim", [2, 3])
def test_refining_a_centred_box_keeps_its_faces_paired(dim):
    """Cells marked near one face only of a centred box: its partner face is bisected
    alike, and the guard compares the box's own faces."""
    from femtools.refine_periodic import refine_congruent
    if dim == 2:
        mesh = df.RectangleMesh(df.Point(-.5, -.5), df.Point(.5, .5), 6, 6)
    else:
        mesh = df.BoxMesh(df.Point(-.5, -.5, -.5), df.Point(.5, .5, .5), 4, 4, 4)
    marked = df.MeshFunction("bool", mesh, dim, False)
    for c in df.cells(mesh):
        marked[c] = c.midpoint().x() < -0.3
    new, _, _ = refine_congruent(mesh, marked, [.5] * dim, list(range(dim)),
                                 lower=[-.5] * dim)
    x = new.coordinates()
    assert new.num_cells() > mesh.num_cells()
    assert np.sum(np.abs(x[:, 0] + .5) < 1e-9) == np.sum(np.abs(x[:, 0] - .5) < 1e-9)
