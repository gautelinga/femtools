"""Periodic Plaza refinement keeps the identified faces paired edge by edge, on a sheared
box made of exact longest-edge ties, refined three times, serially and on 3 ranks; frozen
edges are never bisected, and 3 ranks give the serial numbers."""
import os
import shutil
import sys

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "plaza_flip_worker.py")
FREEZE_WORKER = os.path.join(HERE, "plaza_freeze_worker.py")


@pytest.mark.parametrize("ranks", [pytest.param(1, marks=pytest.mark.slow), pytest.param(3, marks=pytest.mark.mpi)])
def test_faces_stay_paired_edge_by_edge(ranks, tmp_path):
    proc = support.run([sys.executable, WORKER, str(tmp_path / "sheared.h5")], ranks,
                       timeout=600)
    results = [line.split() for line in proc.stdout.splitlines() if line.startswith("RESULT")]
    assert len(results) == 3, proc.stdout[-2000:]
    assert all(r[3] == "ok" for r in results), results
    # the case tested the tie-break: some pass had exact ties to decide
    assert sum(int(r[4]) for r in results) > 0, results


def freeze_results(ranks):
    proc = support.run([sys.executable, FREEZE_WORKER], ranks, timeout=600)
    rows = [line.split()[1:] for line in proc.stdout.splitlines()
            if line.startswith("RESULT")]
    return {r[0]: [int(v) for v in r[1:]] for r in rows}


@pytest.mark.mpi
def test_frozen_edges_are_never_bisected():
    """The frozen face keeps its facets, which the same marking without the freeze
    bisects; some marked cells are left whole; the closure reaches past the cells holding
    a face edge; and 3 ranks give the serial numbers."""
    serial = freeze_results(1)
    cells, nz, marked, unmarked, frozen_cells, whole = serial["frozen"]
    nz0 = serial["free"][1]
    assert nz0 > 128 and nz == 128, (nz0, nz)
    assert 0 < unmarked < marked and 0 < whole <= unmarked
    assert serial["free"][3:] == [0, 0, 0]
    assert frozen_cells > 256, frozen_cells       # 256: the cells holding a face edge
    if shutil.which("mpirun"):
        assert freeze_results(3) == serial
