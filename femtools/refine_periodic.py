"""Refinement of a mesh with periodic faces, checked before it is trusted.

    refine_congruent(mesh, markers, box, axes)   periodic Plaza, then the congruence guard
    congruent(mesh, box, axes)                    whether identified faces carry the same nodes
    check_congruent(mesh, box, axes)              the same, or exit with the refusal
    (each takes `lower=`, the box's lower corner when it is not the origin)
    compute_box(mesh)                             the box's upper corner, over the mesh's ranks
    dorfler_threshold(indicator, theta)           the bulk-chasing threshold of a DG0 indicator

`constrained_domain` pairs periodic dofs by coordinate. If a face is refined and the face
it is identified with is not, the slave dofs with no master are simply left
unconstrained: dolfin builds the space, the solve runs, and periodicity is quietly gone
along that face. `refine_congruent` refines through `femtools.plaza_periodic`, which
bisects identified faces alike, and checks the faces are congruent after it. There is no
fallback: marking every cell on the identified faces also keeps them congruent, but
refines that layer uniformly on every pass, and the cost compounds.

Runs under `mpirun`: `congruent` gathers the face nodes across ranks and deduplicates them
by position before comparing (a rank-boundary vertex is ghosted onto more than one rank).
"""

import dolfin as df

TOL = 1e-9


def _say(mesh, text):
    if mesh.mpi_comm().rank == 0:
        print(text, flush=True)


def congruent(mesh, box, axes, exempt=None, lower=None):
    """Whether the identified faces still carry the same nodes, edges included
    (`femtools.periodic.faces_congruent`). `exempt` is a bool per vertex then per edge
    of nodes allowed no partner; `lower` the box's lower corner (the origin when None,
    which a mesh elsewhere is refused)."""
    from femtools.periodic import faces_congruent
    ok, _, _, _ = faces_congruent(mesh, box, axes, TOL, edges=True, exempt=exempt,
                                  lower=lower)
    return ok


def check_congruent(mesh, box, axes, exempt=None, lower=None):
    """`congruent`, or exit with the refinement's refusal: the guard on a refined mesh
    (`refine_congruent`, or its caller after `deferred=True`)."""
    if not congruent(mesh, box, axes, exempt, lower):
        raise SystemExit(
            "the identified faces came out of the refinement not matching. Do "
            "not solve on this mesh: a periodic dof with no partner is left "
            "unconstrained and dolfin will not say so.")


def compute_box(mesh):
    """The periodic box's upper corner off the mesh itself, over the mesh's ranks
    (`periodic.extent`)."""
    from femtools.periodic import extent
    return extent(mesh)[1]


def dorfler_threshold(indicator_, theta, comm=None, iterations=60):
    """The largest t with sum of the indicator over {eta_K > t} at least theta of all.

    Bulk chasing (Doerfler 1996): refine the smallest set of cells carrying a fixed
    share of the total. Scale free, so it does not care what units the indicator is in,
    and it responds to how concentrated the error is rather than to how large. For an
    indicator carrying a sign it sums |eta|: a share of the total contribution, not of
    the estimate.

    Found by bisection on the threshold rather than by sorting, so nothing is gathered
    and the cost is a fixed number of small reductions. `df.MPI.sum/max` are
    allreduces, so every rank leaves with the same answer; a bisection where the ranks
    disagree about the branch does not terminate together.
    """
    comm = comm or df.MPI.comm_world
    a = abs(indicator_.vector().get_local())
    total = df.MPI.sum(comm, float(a.sum()))
    lo, hi = 0.0, df.MPI.max(comm, float(a.max()) if len(a) else 0.0)
    if total <= 0.0:
        return 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        got = df.MPI.sum(comm, float(a[a > mid].sum()))
        lo, hi = (mid, hi) if got >= theta * total else (lo, mid)
    return lo


def refine_congruent(mesh, cell_markers, box, axes, redistribute=False, max_cells=None,
                     exempt=None, max_rank_cells=None, frozen=None, report=None,
                     deferred=False, lower=None):
    """The periodic-safe refinement, with its output checked before it is trusted.

    Returns `(new_mesh, npair, parent_cell)`; `parent_cell` is None after a
    redistribution (`plaza_periodic.refine_periodic`) and for the no-periodic-axes
    `df.refine` path. The congruence guard lives here so no caller can skip it, with one
    exception asked for by name: `deferred=True` leaves it to the caller, which must run
    `check_congruent` on the mesh it goes on with before solving on it or refining it
    again (on a rebalanced mesh, say: the positions it compares are the same on any
    partition).

    `max_cells`: raise `plaza_periodic.TooManyCells` rather than refine past it,
    counted exactly before anything is built; `max_rank_cells` the same for the most
    loaded rank. Periodic axes only: `df.refine` cannot count first.

    `exempt` is a callable `(new_mesh, parent_cell) ->` a bool per vertex then per edge
    of the new mesh: nodes allowed no periodic partner (a slab's inlet and outlet, say).
    The guard forgives those and nothing else.

    `frozen` and `report`: edges never to bisect, and what freezing them left unmarked
    (`plaza_periodic.refine_periodic`). Periodic axes only.

    `lower`: the box's lower corner, when it is not the origin.
    """
    _say(mesh, "refining")
    if axes:
        from femtools.plaza_periodic import refine_periodic as plaza_periodic
        new_mesh, npair, parent_cell = plaza_periodic(
            mesh, cell_markers, box, axes, redistribute=redistribute,
            max_cells=max_cells, max_rank_cells=max_rank_cells, frozen=frozen,
            report=report, lower=lower)
        _say(mesh, "periodic Plaza: {} identified edge pairs".format(npair))
        if deferred:
            if exempt is not None:
                raise ValueError("refine_congruent: a deferred guard takes its exempt "
                                 "nodes at check_congruent")
        else:
            check_congruent(new_mesh, box, axes,
                            None if exempt is None else exempt(new_mesh, parent_cell),
                            lower)
    else:
        if max_cells or max_rank_cells or frozen is not None:
            raise ValueError("max_cells and frozen need periodic axes: df.refine can "
                             "neither count nor freeze before it refines")
        new_mesh = df.refine(mesh, cell_markers)
        npair, parent_cell = 0, None
    return new_mesh, npair, parent_cell
