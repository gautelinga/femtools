"""`gather_rows`, the raw-byte gather behind `gather_global`, is exact in rounds.

    python3 -m pytest tests/test_gather_rows.py

dolfin, mpi4py and `mpirun`. mpi4py's pickling `gather` fails past 2 GiB delivered to
rank zero; `gather_rows` splits the gather into `Gatherv` rounds; the worker,
`tests/gather_rows_worker.py`, shrinks the round size so the rounds, uneven counts, an
empty rank, dtypes and trailing shapes are all exercised on small arrays.
"""
import os
import sys

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin, pytest.mark.mpi]

from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gather_rows_worker.py')
CASES = ('int64_rows', 'float_rows_3', 'bool_rows', 'empty_rank_default_dtype',
         'empty_root_default_dtype', 'disagreeing_dtypes_refused', 'gather_global',
         'gather_global_empty_root')


@pytest.fixture(scope='module')
def results():
    proc = support.run([sys.executable, WORKER], 3, timeout=300)
    return dict(line.split()[1:3] for line in proc.stdout.splitlines()
                if line.startswith('RESULT '))


@pytest.mark.parametrize('case', CASES)
def test_gathered_exactly(results, case):
    assert results.get(case) == 'ok', results
