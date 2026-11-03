"""Run under `mpirun -n 3` by tests/test_gather_rows.py; prints one RESULT line per case.

`GATHER_ROUND_BYTES` is shrunk so that small arrays go through many `Gatherv` rounds:
the rounds, the uneven row counts (one rank sends nothing), the dtypes and the trailing
shapes are what the >2 GiB case exercises, at a size a test can afford.
"""
import numpy as np
from mpi4py import MPI

from femtools import xdmf as ex

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ex.GATHER_ROUND_BYTES = 40                  # a few rows a round: dozens of rounds below


def result(name, ok):
    if rank == 0:
        print('RESULT', name, 'ok' if ok else 'FAIL', flush=True)


counts = [7, 0, 13]                          # rank 1 has nothing to send
start = sum(counts[:rank])

a = np.arange(start, start + counts[rank], dtype=np.int64)
got = ex.gather_rows(comm, a)
result('int64_rows', rank != 0 or np.array_equal(got, np.arange(sum(counts))))

f = np.arange(3 * counts[rank], dtype=np.float64).reshape(-1, 3) + 1000.0 * rank
got = ex.gather_rows(comm, f)
want = np.vstack([np.arange(3 * c, dtype=np.float64).reshape(-1, 3) + 1000.0 * r
                  for r, c in enumerate(counts)])
result('float_rows_3', rank != 0 or (got.shape == want.shape and np.array_equal(got, want)))

b = (np.arange(counts[rank]) + rank) % 2 == 0
got = ex.gather_rows(comm, b)
want = np.concatenate([(np.arange(c) + r) % 2 == 0 for r, c in enumerate(counts)])
result('bool_rows', rank != 0 or (got.dtype == bool and np.array_equal(got, want)))

# the empty rank sends a default-typed, 1-D empty array while the others send (n, 2)
# int32 rows: rank zero must read everyone's bytes as the others' type, not its own --
# so here rank 1 (empty) is the odd one, and in the next case rank 0 is
e = (np.zeros(0) if counts[rank] == 0
     else np.arange(2 * counts[rank], dtype=np.int32).reshape(-1, 2) + 100 * rank)
got = ex.gather_rows(comm, e)
want = np.vstack([np.arange(2 * c, dtype=np.int32).reshape(-1, 2) + 100 * r
                  for r, c in enumerate(counts) if c])
result('empty_rank_default_dtype', rank != 0 or (got.dtype == np.int32 and np.array_equal(got, want)))

rows0 = [0, 5, 4]                           # now rank 0 is the empty one
e = (np.zeros(0) if rows0[rank] == 0
     else np.arange(rows0[rank], dtype=np.int64) + 10 * rank)
got = ex.gather_rows(comm, e)
want = np.concatenate([np.arange(c, dtype=np.int64) + 10 * r for r, c in enumerate(rows0) if c])
result('empty_root_default_dtype', rank != 0 or (got.dtype == np.int64 and np.array_equal(got, want)))

# rows that genuinely disagree are refused on every rank, not decoded wrongly
try:
    ex.gather_rows(comm, np.zeros(3, dtype=np.float32 if rank == 2 else np.float64))
    refused = False
except ValueError:
    refused = True
result('disagreeing_dtypes_refused', bool(comm.allreduce(int(refused), op=MPI.MIN)))

# gather_global end to end: 30 global rows, every rank also carries a neighbour's rows
# as ghosts (equal values), and every row is covered
n = 30
mine = np.arange(10 * rank, 10 * rank + 10)
ghost = (mine[-1] + 1 + np.arange(3)) % n
index = np.concatenate([mine, ghost])
payload = np.column_stack([index * 2.0, index * 3.0])
got = ex.gather_global(comm, index, payload, n)
want = np.column_stack([np.arange(n) * 2.0, np.arange(n) * 3.0])
result('gather_global', rank != 0 or np.array_equal(got, want))

# gather_global with rank zero owning nothing and passing default-typed empties, while
# the others send (n, 2) int64 rows: its output must take the gathered type and shape
n = 20
mine = np.arange(10 * (rank - 1), 10 * rank) if rank else np.zeros(0)
payload = (np.column_stack([mine, -mine]).astype(np.int64) if rank else np.zeros(0))
got = ex.gather_global(comm, mine, payload, n)
want = np.column_stack([np.arange(n), -np.arange(n)]).astype(np.int64)
result('gather_global_empty_root', rank != 0 or (got.dtype == np.int64 and np.array_equal(got, want)))
