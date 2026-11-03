"""Worker for tests/test_compress.py: `hdf5_rows.write_ordered` on the ranks mpirun gives,
with messages capped small, so a rank's block goes to rank zero in several batches; and
rows already one run a rank in rank order (the path that sends no index), whose stored
chunks are the scattered path's to the byte.

    mpirun -n 3 python3 tests/compress_worker.py OUT_DIR
"""
import os
import sys

import h5py
import numpy as np
from mpi4py import MPI

from femtools import hdf5_rows


def main(out):
    comm = MPI.COMM_WORLD
    hdf5_rows.MESSAGE = 64 << 10
    rng = np.random.default_rng(3)                  # the same on every rank
    whole = {"i": rng.integers(0, 1 << 40, (700_001, 4)), "f": rng.random((500_003, 3)),
             "b": rng.random(300_000) > .5}
    path = os.path.join(out, "o.h5")
    for k, (name, a) in enumerate(whole.items()):
        mine = np.arange(len(a))[comm.rank::comm.size]
        extra = rng.integers(0, len(a), 1000)       # repeats, the same bits from every rank
        rows = np.r_[mine, extra]
        hdf5_rows.write_ordered(path, name, rows, a[rows], len(a), comm,
                                mode="w" if k == 0 else "a")
    # one run a rank, rank 1 holding none, the others uneven
    a = whole["i"]
    cut = np.r_[0, np.sort(rng.integers(0, len(a), comm.size - 1)), len(a)]
    if comm.size > 2:
        cut[2] = cut[1]
    runs = np.arange(cut[comm.rank], cut[comm.rank + 1])
    hdf5_rows.write_ordered(path, "runs", runs, a[runs], len(a), comm)
    assert hdf5_rows._runs(comm, runs, len(a)) is not None
    assert hdf5_rows._runs(comm, runs[::-1].copy(), len(a)) is None   # some rank holds two
    if comm.rank == 0:
        with h5py.File(path, "r") as h:
            for name, a in whole.items():
                assert np.array_equal(h[name][()], a), name
                assert h[name].compression == "gzip" and h[name].shuffle, name
            d, e = h["runs"], h["i"]
            assert np.array_equal(d[()], whole["i"])
            for k in range(0, d.shape[0], d.chunks[0]):
                assert d.id.read_direct_chunk((k, 0)) == e.id.read_direct_chunk((k, 0)), k
        print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
