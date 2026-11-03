"""Worker for tests/test_compress.py: `compress.rewrite(path, comm)` on the ranks mpirun
gives, of a file rank zero writes with plain, filtered, resizable, scalar and empty
datasets, attributes, and soft and external links (and, with `vlen`, a variable-length string, which parallel HDF5
cannot write: rank zero rewrites that file alone). Rank zero prints which way h5py wrote,
the result, and `ok` when everything is kept and the plain datasets are compressed.

    mpirun -n 3 python3 tests/rewrite_worker.py OUT [vlen]
"""
import os
import sys

import h5py
import numpy as np
from mpi4py import MPI

from femtools import compress, hdf5_rows


def main(out, vlen=False):
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(5)
    a = {"topology": rng.integers(0, 1 << 30, (300_001, 4)), "coordinates": rng.random((200_003, 3)),
         "lzf": rng.integers(0, 9, (100_000, 4))}
    p = os.path.join(out, "f.h5")
    lzf = None
    if comm.rank == 0:
        with h5py.File(p, "w") as h:
            h.attrs["eps"] = np.nan
            g = h.create_group("mesh")
            g.attrs["partition"] = np.array([0, 5], np.uint64)
            g["topology"] = a["topology"]
            g["coordinates"] = a["coordinates"]
            h.create_dataset("growing", data=a["coordinates"], maxshape=(None, 3),
                             chunks=(4096, 3), fillvalue=-7.0)
            h.create_dataset("lzf", data=a["lzf"], chunks=(8192, 4), compression="lzf")
            h["scalar"] = 3.5
            h["none"] = np.zeros(0)
            h["mesh/again"] = h5py.SoftLink("/mesh/topology")
            h["elsewhere"] = h5py.ExternalLink("other.h5", "/x")      # need not exist
            if vlen:
                h["name"] = "tetrahedron"
        with h5py.File(p, "r") as h:
            lzf = [(m, b) for _, m, b in compress._chunks(h["lzf"])]
        os.utime(p, (1e9, 1e9))
    comm.Barrier()
    code, what = compress.rewrite(p, comm)
    assert comm.allgather((code, what)) == [(code, what)] * comm.size
    if comm.rank:
        return
    print("mpio" if hdf5_rows.mpio(comm) else "gather")
    print(code)
    with h5py.File(p, "r") as h:
        assert np.isnan(h.attrs["eps"]) and list(h["mesh"].attrs["partition"]) == [0, 5]
        for n in ("topology", "coordinates"):
            d = h["mesh/" + n]
            assert np.array_equal(d[()], a[n]) and d.compression == "gzip" and d.shuffle, n
        g = h["growing"]
        assert g.maxshape == (None, 3) and g.fillvalue == -7.0 and g.compression == "gzip"
        assert np.array_equal(g[()], a["coordinates"])
        assert h["lzf"].compression == "lzf" and np.array_equal(h["lzf"][()], a["lzf"])
        assert [(m, b) for _, m, b in compress._chunks(h["lzf"])] == lzf     # as stored
        assert h.get("mesh/again", getlink=True).path == "/mesh/topology"
        e = h.get("elsewhere", getlink=True)
        assert (e.filename, e.path) == ("other.h5", "/x")
        assert h["scalar"][()] == 3.5 and h["none"].shape == (0,)
        if vlen:
            assert h["name"][()] == b"tetrahedron"
    assert os.path.getmtime(p) == 1e9 and not os.path.exists(p + compress.SUFFIX)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2)
