"""Exactness of the search-free transfer, against fields with known answers:

- `transfer_p1_field` matches `df.interpolate` of a P1 field onto its periodic refinement
  to 1e-13;
- through two refinements, a stored quadratic P2 field read by `DistributedCellSource`
  is reproduced at every P2 node of the refined mesh, midsides included, to 1e-12, and
  survives `as_p2_function`;
- global cell ids round-trip through HDF5: a mesh refined and written on 3 ranks, read
  back on 1, 2 and 3, names the same tetrahedra by them;
- every case passes again under `mpirun -n 3`.
"""
import os
import shutil
import sys
import tempfile

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

import numpy as np  # noqa: E402

import dolfin as df  # noqa: E402

from tests import support  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set in every mpirun child: a child that spawned again would do so on each rank
MPI_CHILD = "FEMTOOLS_TEST_MPI_CHILD"
OWN_SCRATCH = "FEMTOOLS_TEST_SCRATCH" not in os.environ
SCRATCH = os.environ.get("FEMTOOLS_TEST_SCRATCH") or tempfile.mkdtemp(prefix="htransfer_")
CHILD = {"FEMTOOLS_TEST_SCRATCH": SCRATCH, MPI_CHILD: "1"}
BOX = np.array([1.0, 1.0, 2.0])

df.set_log_level(40)


@pytest.fixture(scope="module", autouse=True)
def scratch():
    yield SCRATCH
    if OWN_SCRATCH:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def _not_in_a_child():
    if os.environ.get(MPI_CHILD):
        pytest.skip("already an mpirun child")


def quadratic_exact(x):
    u = np.empty((len(x), 3))
    u[:, 0] = 0.1 + 0.5*x[:, 0] - 0.25*x[:, 1] + x[:, 0]**2 - 0.5*x[:, 1]*x[:, 2]
    u[:, 1] = -0.2 + x[:, 2] + 0.75*x[:, 1]**2 - 0.3*x[:, 0]*x[:, 1]
    u[:, 2] = 1.0 + 0.25*x[:, 0] - 0.5*x[:, 2] + 0.4*x[:, 2]**2 + 0.2*x[:, 0]*x[:, 2]
    return u


QUADRATIC = ("0.1 + 0.5*x[0] - 0.25*x[1] + x[0]*x[0] - 0.5*x[1]*x[2]",
             "-0.2 + x[2] + 0.75*x[1]*x[1] - 0.3*x[0]*x[1]",
             "1.0 + 0.25*x[0] - 0.5*x[2] + 0.4*x[2]*x[2] + 0.2*x[0]*x[2]")


def quadratic_files():
    """mesh.h5 and u.h5 (the quadratic on P2) of the box, written serially once."""
    out = os.path.join(SCRATCH, "quadratic")
    comm = df.MPI.comm_world
    if df.MPI.rank(comm) == 0 and not os.path.exists(os.path.join(out, "u.h5")):
        os.makedirs(out, exist_ok=True)
        mesh = df.BoxMesh(df.MPI.comm_self, df.Point(0, 0, 0), df.Point(*BOX), 8, 8, 16)
        u = df.interpolate(df.Expression(QUADRATIC, degree=2),
                           df.VectorFunctionSpace(mesh, "Lagrange", 2))
        with df.HDF5File(df.MPI.comm_self, os.path.join(out, "mesh.h5"), "w") as h5:
            h5.write(mesh, "mesh")
        with df.HDF5File(df.MPI.comm_self, os.path.join(out, "u.h5"), "w") as h5:
            h5.write(u, "u")
    df.MPI.barrier(comm)
    return os.path.join(out, "mesh.h5"), os.path.join(out, "u.h5")


def read_mesh(path):
    mesh = df.Mesh()
    with df.HDF5File(df.MPI.comm_world, path, "r") as h5:
        h5.read(mesh, "mesh", False)
    return mesh


def refined_twice(mesh, box, axes):
    """Two periodic-safe passes with different markings; the meshes and the composed
    ancestor map."""
    from femtools.plaza_periodic import refine_periodic
    from femtools.hierarchical_transfer import compose
    anc = np.arange(mesh.num_cells(), dtype=np.int64)
    meshes = [mesh]
    for cut in (0.35, 0.6):
        m = meshes[-1]
        cm = df.MeshFunction("bool", m, 3)
        cm.array()[:] = np.array([c.midpoint().z() for c in df.cells(m)]) < cut
        new, _, parent = refine_periodic(m, cm, box, axes, redistribute=False)
        assert parent is not None
        anc = compose(anc, parent)
        meshes.append(new)
    return meshes, anc


def test_p1_exact():
    """transfer_p1_field is exact against df.interpolate"""
    from femtools.plaza_periodic import refine_periodic
    from femtools.hierarchical_transfer import transfer_p1_field
    comm = df.MPI.comm_world
    mesh = df.UnitCubeMesh(comm, 6, 6, 6)
    f = df.Function(df.FunctionSpace(mesh, "Lagrange", 1))
    f.interpolate(df.Expression("sin(3*x[0]) + x[1]*x[2] - 0.5*x[2]", degree=3))
    cm = df.MeshFunction("bool", mesh, 3)
    cm.array()[:] = np.array([c.midpoint().x() for c in df.cells(mesh)]) < 0.4
    new, _, parent = refine_periodic(mesh, cm, [1.0, 1.0, 1.0], [0, 1], redistribute=False)
    fast = transfer_p1_field(mesh, f.compute_vertex_values(mesh), new, parent)
    ref = df.interpolate(f, df.FunctionSpace(new, "Lagrange", 1)).compute_vertex_values(new)
    worst = df.MPI.max(comm, float(np.abs(fast - ref).max()))
    assert worst < 1e-13, "warm-start transfer off by {:.2e}".format(worst)


def test_p2_transfer_exact():
    """every P2 node of a twice-refined mesh, read through DistributedCellSource"""
    from femtools.hierarchical_transfer import (DistributedCellSource, as_p2_function,
                                                local_to_global_cells, p2_entity_nodes,
                                                velocity_on_mesh_p2)
    mesh_h5, u_h5 = quadratic_files()
    comm = df.MPI.comm_world
    mesh0 = read_mesh(mesh_h5)
    source = DistributedCellSource(mesh_h5, u_h5, "u", degree=2, components=3, comm=comm)
    gcell0 = local_to_global_cells(mesh0)
    meshes, anc = refined_twice(mesh0, BOX, [0, 1])

    worst = 0.0
    for mesh, ancestors in ((mesh0, np.arange(mesh0.num_cells())), (meshes[-1], anc)):
        vals = velocity_on_mesh_p2(source, mesh, np.asarray(ancestors), gcell0)
        for (points, _), v in zip(p2_entity_nodes(mesh), vals):
            worst = max(worst, float(np.abs(v - quadratic_exact(points)).max()))
        # read the dofs back by their entity
        f = as_p2_function(mesh, vals)
        owned = len(f.vector().get_local())
        space = f.function_space()
        for dim, (points, _) in enumerate(p2_entity_nodes(mesh)):
            block = np.asarray(space.dofmap().entity_dofs(mesh, dim)).reshape(-1, 3)
            keep = (block[:, 0] >= 0) & (block[:, 0] < owned)
            back = f.vector().get_local()[block[keep]]
            worst = max(worst, float(np.abs(back - quadratic_exact(points[keep])).max()))
    worst = df.MPI.max(comm, worst)
    assert worst < 1e-12, "P2 nodes reproduced only to {:.2e}".format(worst)


def _write_refined(out_dir):
    """The box refined once on the ranks given and written."""
    from femtools.refine_periodic import refine_congruent
    from femtools.hierarchical_transfer import local_to_global_cells
    comm = df.MPI.comm_world
    mesh = read_mesh(quadratic_files()[0])
    # marked by global id parity, so the marked set does not depend on the partition
    cm = df.MeshFunction("bool", mesh, 3)
    cm.array()[:] = local_to_global_cells(mesh) % 2 == 0
    new, _, parent = refine_congruent(mesh, cm, BOX, [0, 1], redistribute=False)
    assert parent is not None
    os.makedirs(out_dir, exist_ok=True)
    with df.HDF5File(comm, os.path.join(out_dir, "refined_mesh.h5"), "w") as h5:
        h5.write(new, "mesh")


def _check_refined(out_dir):
    """Read back on the ranks given: each local cell's global id names, in
    `mesh/cell_indices`, the row holding its four global vertices, and `mesh/coordinates`
    is indexed by global vertex id."""
    import h5py
    from femtools.hierarchical_transfer import local_to_global_cells
    path = os.path.join(out_dir, "refined_mesh.h5")
    mesh = read_mesh(path)
    with h5py.File(path, "r") as handle:
        topology = np.asarray(handle["mesh/topology"], dtype=np.int64)
        cell_indices = np.asarray(handle["mesh/cell_indices"], dtype=np.int64)
        coordinates = np.asarray(handle["mesh/coordinates"], dtype=np.float64)
    row_of_global = np.full(int(cell_indices.max()) + 1, -1, dtype=np.int64)
    row_of_global[cell_indices] = np.arange(len(cell_indices), dtype=np.int64)
    gvert = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    rows = row_of_global[local_to_global_cells(mesh)]
    assert (rows >= 0).all(), "a local cell names a global id the file does not hold"
    mine = np.sort(gvert[mesh.cells()], axis=1)
    bad = int((mine != np.sort(topology[rows], axis=1)).any(axis=1).sum())
    assert bad == 0, "{} of {} cells land on the wrong row".format(bad, len(mine))
    dx = (np.abs(coordinates[gvert] - mesh.coordinates()).max() if mesh.num_vertices()
          else 0.0)
    assert dx == 0.0, "coordinates are not indexed by global vertex id ({:.2e})".format(dx)
    print("all rows agree")


def _spawn(nranks, *args):
    proc = support.run([sys.executable, "-m", "tests.test_hierarchical_transfer"]
                       + list(args), nranks, env=CHILD, timeout=600, cwd=ROOT)
    return proc.stdout + proc.stderr


@pytest.mark.mpi
def test_cell_id_roundtrip():
    """written on 3 ranks, read back on 1, 2 and 3"""
    _not_in_a_child()
    out = os.path.join(SCRATCH, "roundtrip")
    _spawn(3, "--write-refined", out)
    for n in (1, 2, 3):
        assert _spawn(n, "--check-refined", out).count("all rows agree") == n


@pytest.mark.mpi
def test_every_case_again_on_three_ranks():
    """the cases above under mpirun -n 3, every rank running every case"""
    _not_in_a_child()
    proc = support.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        os.path.abspath(__file__)], 3, env=CHILD, timeout=900, cwd=ROOT)
    out = proc.stdout + proc.stderr
    assert out.count("passed") >= 3, out[-2000:]      # one tally per rank


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-refined":
        _write_refined(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--check-refined":
        _check_refined(sys.argv[2])
    else:
        raise SystemExit("a pytest module; run as a script only by its own mpirun children")
