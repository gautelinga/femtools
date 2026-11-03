"""dolfin 2019's parallel periodic dofmap breaks the pairing on some partitions; the
partition repair prevents it and the check catches it.

    python3 -m pytest tests/test_periodic_partition.py

**System python3** (dolfin), and `mpirun`: the defect needs two ranks. The worker,
`tests/periodic_partition_worker.py`, crafts the failing case on purpose -- a cell
touching a periodic face at one vertex, alone on a rank -- so this is deterministic, not
a matter of which partition ParMETIS happens to pick.
"""
import os
import sys

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'periodic_partition_worker.py')


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp('periodic_partition'))
    proc = support.run([sys.executable, WORKER, tmp], 2, timeout=900)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            _, name, value = line.split(None, 2)
            out[name] = value
    return out


@pytest.mark.mpi
def test_the_crafted_partition_breaks_the_pairing_and_the_check_says_so(results):
    """The defect, reproduced on purpose: one endangered vertex, and a space whose
    periodic pairing is broken -- which `check_pairing` refuses."""
    assert results['endangered_crafted'] == '1'
    assert results['crafted_space_ok'] == 'False'


@pytest.mark.mpi
def test_the_repair_moves_the_cell_and_the_pairing_holds(results):
    assert results['moved'] == '1'
    assert results['endangered_repaired'] == '0'
    assert results['repaired_space_ok'] == 'True'


@pytest.mark.mpi
def test_a_distributed_mesh_needing_a_move_is_repaired_without_a_whole_read(results):
    """The crafted partition, read as written, goes through the layer gather and the
    in-memory rebuild with the move (`rebuild`): one cell moved, every
    cell and every global cell id kept, and a correctly paired space."""
    assert results['repaired_log'] == '1'
    assert results['repaired_cells'] == '1296'
    assert results['repaired_cell_ids'] == '1'
    assert results['repaired_distributed_space_ok'] == 'True'


@pytest.mark.mpi
def test_read_mesh_gives_a_correctly_paired_space_from_an_xdmf_file(results):
    assert results['read_mesh_cells'] == '1296'
    assert results['read_mesh_space_ok'] == 'True'


@pytest.mark.mpi
def test_a_read_only_mesh_directory_is_only_read(results):
    """The mesh's directory is only read: the repair writes nothing anywhere."""
    assert results['read_only_mesh_dir_cells'] == '1296'


@pytest.mark.mpi
def test_the_repair_writes_nothing(results):
    """A partition that needs a move, repaired in a working directory no rank can write:
    rebuilt in memory, every cell with its global id, vertices and coordinates, and the
    result needs no further move."""
    assert results['unwritable_cwd_repaired_in_memory'] == 'True'




WORKER_2D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'periodic_partition_2d_worker.py')


@pytest.fixture(scope='module')
def results_2d(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp('periodic_partition_2d'))
    proc = support.run([sys.executable, WORKER_2D, tmp], 2, timeout=900)
    return dict(line.split(None, 2)[1:] for line in proc.stdout.splitlines()
                if line.startswith('RESULT '))


@pytest.mark.mpi
def test_on_triangles_the_crafted_partition_breaks_the_pairing_and_the_check_says_so(
        results_2d):
    """The defect on a triangle mesh: a vertex of a periodic edge held by a rank only
    through a triangle touching the edge there."""
    assert results_2d['endangered_crafted'] == '1'
    assert results_2d['crafted_space_ok'] == 'False'


@pytest.mark.mpi
def test_on_triangles_the_repair_moves_the_cell_and_the_pairing_holds(results_2d):
    assert results_2d['moved'] == '1'
    assert results_2d['endangered_repaired'] == '0'
    assert results_2d['repaired_log'] == '1'
    assert results_2d['repaired_cells'] == '576'
    assert results_2d['repaired_cell_ids'] == '1'
    assert results_2d['repaired_space_ok'] == 'True'


@pytest.mark.mpi
def test_on_triangles_read_mesh_gives_a_correctly_paired_space(results_2d):
    assert results_2d['read_mesh_cells'] == '576'
    assert results_2d['read_mesh_space_ok'] == 'True'


@pytest.mark.mpi
def test_a_box_off_the_origin_is_checked_and_repaired(results_2d):
    """A box centred at the origin: the check, given the lower corner, still sees the
    broken pairing, and the repair (which needs the upper corner alone) fixes it."""
    assert results_2d['centred_crafted_space_ok'] == 'False'
    assert results_2d['centred_repaired_space_ok'] == 'True'


@pytest.mark.parametrize('dim', [2, 3])
def test_the_check_takes_the_lower_corner(dim):
    """Serially, on a box centred at the origin: a periodic space passes the check given
    the lower corner, and fails it taken from the origin, where the upper faces' images
    land on the middle plane."""
    import dolfin as df
    from mpi4py import MPI
    from femtools import periodic_partition as pp
    from femtools.periodic import periodic_box
    lo, hi = [-0.5] * dim, [0.5] * dim
    mesh = (df.RectangleMesh(MPI.COMM_SELF, df.Point(*lo), df.Point(*hi), 6, 6) if dim == 2
            else df.BoxMesh(MPI.COMM_SELF, df.Point(*lo), df.Point(*hi), 4, 4, 4))
    periodic = [True] * dim
    V = df.FunctionSpace(mesh, 'CG', 2, constrained_domain=periodic_box(hi, 1e-10, periodic, lo))
    pp.check_space_pairing(V, hi, periodic, MPI.COMM_SELF, lower=lo)
    with pytest.raises(pp.PairingError):
        pp.check_space_pairing(V, hi, periodic, MPI.COMM_SELF)

def _brute_dual(cells):
    """Every pair of cells sharing a face, both ways, by a dict of sorted triples."""
    seen, pairs = {}, set()
    for c, t in enumerate(cells):
        for drop in range(4):
            f = tuple(sorted(int(v) for j, v in enumerate(t) if j != drop))
            if f in seen:
                pairs |= {(seen[f], c), (c, seen[f])}
            else:
                seen[f] = c
    return pairs


def test_the_dual_graph_is_every_shared_face():
    """`dual_graph_of` on one rank -- the local pairing and the exchange both run -- is
    the brute-force dual graph, edge for edge."""
    import dolfin as df
    import numpy as np
    from mpi4py import MPI
    from femtools.periodic_partition import dual_graph_of
    mesh = df.UnitCubeMesh(MPI.COMM_SELF, 4, 3, 2)
    cells = np.asarray(mesh.cells(), np.int64)
    rows, cols, offset = dual_graph_of(cells, MPI.COMM_SELF)
    assert offset == 0
    got = set(zip(rows.tolist(), cols.tolist()))
    assert len(got) == len(rows)                      # no edge twice
    assert got == _brute_dual(cells)


def test_a_face_of_three_cells_is_refused():
    """Three cells on one face would put overlapping pairs in the graph: refused, on
    every rank (`agreed`), rather than handed to ParMETIS."""
    import numpy as np
    from mpi4py import MPI
    from femtools.periodic_partition import dual_graph_of
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], np.int64)
    with pytest.raises(RuntimeError, match='three or more cells'):
        dual_graph_of(cells, MPI.COMM_SELF)


def test_a_graph_past_idx_t_is_refused():
    """With a 32-bit idx_t, a cell count past 2^31 would wrap in vtxdist and adjncy:
    refused before ParMETIS is called. (A 64-bit build holds any count here.)"""
    import dolfin as df
    import numpy as np
    from mpi4py import MPI
    from femtools import periodic_partition as pp
    module = pp._graph_module()
    if module.IDX_BITS != 32:
        pytest.skip('ParMETIS here has a {}-bit idx_t'.format(module.IDX_BITS))
    like = df.Mesh(MPI.COMM_SELF)
    z = np.zeros(1, np.int64)
    with pytest.raises(RuntimeError, match='IDXTYPEWIDTH 64'):
        module.partition(like, np.array([0, 2 ** 31], np.int64), z, z, False, z, False, z,
                         False, pp.ITR, pp.IMBALANCE)


NO_FACE = r"""
import sys
from mpi4py import MPI
import dolfin as df
import numpy as np
from femtools import periodic_partition as pp
from femtools.periodic import PeriodicBox
comm = MPI.COMM_WORLD
path, L, periodic = sys.argv[1], [6.0, 1.0, 1.0], (True, False, False)
if comm.rank == 0:
    m = df.BoxMesh(MPI.COMM_SELF, df.Point(0, 0, 0), df.Point(*L), 24, 2, 2)
    f = df.HDF5File(MPI.COMM_SELF, path, "w")
    f.write(m, "mesh")
    f.close()
comm.Barrier()
mesh = pp.read_mesh(path, None, periodic, comm)
x = mesh.coordinates()
touches = bool(((np.abs(x[:, 0]) < 1e-8) | (np.abs(x[:, 0] - L[0]) < 1e-8)).any())
V = df.FunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBox(L, 1e-8, periodic))
pp.check_space_pairing(V, L, periodic, comm)
none = comm.allreduce(int(not touches), op=MPI.SUM)
if comm.rank == 0:
    print("RANKS WITH NO PERIODIC FACE", none)
"""


@pytest.mark.mpi
def test_a_rank_touching_no_periodic_face_checks_its_none(tmp_path):
    """A bar periodic along its length only, on six ranks: the middle ones touch no
    periodic face and hold no face node, and the pairing check passes over them."""
    proc = support.run([sys.executable, "-c", NO_FACE, str(tmp_path / "bar.h5")], 6,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    none = [int(l.split()[-1]) for l in proc.stdout.splitlines()
            if l.startswith("RANKS WITH NO PERIODIC FACE")]
    assert none and none[0] > 0, proc.stdout[-2000:]


FAMILIES = r"""
import sys
from mpi4py import MPI
import dolfin as df
import numpy as np
from femtools import periodic_partition as pp
comm = MPI.COMM_WORLD
path = sys.argv[1]
if comm.rank == 0:
    m = df.BoxMesh(MPI.COMM_SELF, df.Point(0, 0, 0), df.Point(1, 1, 1), 6, 6, 6)
    f = df.HDF5File(MPI.COMM_SELF, path, "w")
    f.write(m, "mesh")
    f.close()
comm.Barrier()
mesh = pp.read_mesh(path, None, (False, False, False), comm)
anc = np.asarray(mesh.topology().global_indices(3), np.int64)
# a refinement in one corner: there, a coarse cell has 40 descendants
c = mesh.coordinates()[mesh.cells()].mean(1)
k = np.where((c < 0.35).all(1), 40, 1)
fine = np.repeat(anc, k)
part = pp.Families(path, comm).partition(mesh, fine)
rows = comm.gather(np.stack([fine, part], 1), root=0)
if comm.rank == 0:
    t = np.concatenate(rows)
    pairs = np.unique(t, axis=0)                 # (coarse cell, part): one per family
    print("FAMILIES SPLIT", len(pairs) - len(np.unique(pairs[:, 0])))
    load = np.bincount(t[:, 1], minlength=comm.size)
    print("FAMILIES LOAD", load.max() / load.mean(), load.min())
"""


@pytest.mark.mpi
def test_a_refined_mesh_is_partitioned_by_family(tmp_path):
    """`Families`: the coarse cells partitioned by ParMETIS, each weighed by its
    descendants, and every refined cell to its coarse cell's part. On four ranks, a corner
    refined forty-fold: no family is split across ranks, and the refined cells are
    balanced."""
    proc = support.run([sys.executable, "-c", FAMILIES, str(tmp_path / "coarse.h5")], 4,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lines = dict(l.split(None, 2)[1:] for l in proc.stdout.splitlines()
                 if l.startswith("FAMILIES "))
    assert lines.get("SPLIT") == "0", proc.stdout[-2000:]
    ratio, least = lines["LOAD"].split()
    assert float(ratio) < 1.1 and int(least) > 0, lines


def test_rows_of_nothing_are_rows():
    """`as_rows` is `reshape(len, -1)` with the width from the shape, so no rows is a
    (0, width) array, not numpy's error; and the class split of no face nodes is none."""
    import numpy as np
    from mpi4py import MPI
    from femtools.hdf5_rows import as_rows
    from femtools.periodic_partition import _split_classes
    assert as_rows(np.zeros((0, 2, 3))).shape == (0, 6)
    assert as_rows(np.zeros(0)).shape == (0, 1)
    assert as_rows(np.arange(6).reshape(3, 2)).tolist() == [[0, 1], [2, 3], [4, 5]]
    assert _split_classes(MPI.COMM_WORLD, np.zeros((0, 3)), np.zeros((0, 2), np.int64),
                          [1.0, 1.0, 1.0], [0], 1e-8, None) == 0


def test_unique_rows_is_numpys():
    """The repair's `_unique_rows` is `np.unique(axis=0)`, the same rows in the same order,
    on rows that pack into one key (vertex numbers, -1 for no second end, ranks) and on
    rows whose ranges do not."""
    import numpy as np
    from femtools.periodic_partition import _unique_rows
    rng = np.random.default_rng(2)
    for top in (5, 1 << 20, 1 << 40):
        rows = np.stack([rng.integers(0, 2, 5000), rng.integers(0, top, 5000),
                         rng.integers(-1, top, 5000), rng.integers(0, 300, 5000)], 1)
        rows = np.concatenate([rows, rows[::3]]).astype(np.int64)
        assert np.array_equal(_unique_rows(rows), np.unique(rows, axis=0))
    assert _unique_rows(np.zeros((0, 4), np.int64)).shape == (0, 4)


def test_the_csr_is_the_dual_graph_sorted():
    """`dual_csr` packs each edge into one key: the CSR of the edge list lexsorted by row
    then column, as ParMETIS was given it before, entry for entry."""
    import dolfin as df
    import numpy as np
    from mpi4py import MPI
    from femtools.periodic_partition import dual_csr, dual_graph_of
    mesh = df.UnitCubeMesh(MPI.COMM_SELF, 4, 3, 2)
    cells = np.asarray(mesh.cells(), np.int64)
    rows, cols, _ = dual_graph_of(cells, MPI.COMM_SELF)
    order = np.lexsort((cols, rows))
    xadj, adjncy = dual_csr(cells, MPI.COMM_SELF)
    assert xadj.dtype == adjncy.dtype == np.int64
    assert np.array_equal(xadj, np.r_[0, np.cumsum(np.bincount(rows, minlength=len(cells)))])
    assert np.array_equal(adjncy, cols[order])
    xadj, adjncy = dual_csr(np.zeros((0, 4), np.int64), MPI.COMM_SELF)
    assert xadj.tolist() == [0] and len(adjncy) == 0


def test_a_rebuild_on_one_rank_is_the_same_mesh():
    """`Rebuild` hands dolfin its data filled first (`local_data`) and builds from that:
    on one rank the same cells, coordinates and global ids, and the carry by id."""
    import dolfin as df
    import numpy as np
    from mpi4py import MPI
    from femtools.periodic_partition import Rebuild
    mesh = df.UnitCubeMesh(MPI.COMM_SELF, 4, 3, 2)
    gv, gc = (np.asarray(mesh.topology().global_indices(d), np.int64) for d in (0, 3))
    plan = Rebuild(mesh, np.zeros(mesh.num_cells(), np.int64))
    new = plan.build()
    nv, nc = (np.asarray(new.topology().global_indices(d), np.int64) for d in (0, 3))
    assert plan.data is None and new.num_cells() == mesh.num_cells()
    assert {tuple(c) for c in np.sort(gv[mesh.cells()], 1)} == \
        {tuple(c) for c in np.sort(nv[new.cells()], 1)}
    assert np.array_equal(mesh.coordinates()[np.argsort(gv)], new.coordinates()[np.argsort(nv)])
    (cells,), (verts,) = plan.carry(new, [gc], [gv])
    assert np.array_equal(cells, nc) and np.array_equal(verts, nv)
