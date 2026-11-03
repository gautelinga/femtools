"""Reading XDMF + HDF5 files fast enough for billions of tetrahedra.

    it = items("mesh.xdmf")                      # {'geometry', 'topology', attributes...}
    p = pool(8)
    X = read(open_item(it, "geometry"), p)       # the whole dataset
    for first, T in row_blocks(open_item(it, "topology"), p, ahead=2):
        ...                                      # a chunk-row at a time

meshio reads an HDF5 dataset through h5py, which decompresses its gzip chunks one after
another on one thread; at billions of tetrahedra that takes hours. Here the chunks are read
raw and inflated on a thread pool (zlib releases the GIL), and the connectivity can be
walked a chunk-row at a time, the next row being read while the current one is used.

Only the layout meshio and mesh-sphere-packing write is taken this way: rows chunked, one
column a chunk, gzip and nothing else. Any other dataset is read by h5py as before, so the
answers never depend on which path was taken.
"""
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np


def items(xdmf):
    """{'geometry': (h5 path, dataset), 'topology': ..., <attribute name>: ...} of an XDMF
    file whose data are in HDF5 (its first grid, `femtools.xdmf.grids`). Raises ValueError
    for data held inline in the XML."""
    from .xdmf import grids
    g = grids(xdmf)[0]
    out = {k: g[k] for k in ('geometry', 'topology') if g[k] is not None}
    out.update(g['attributes'])
    return out


def _raw_ok(dset):
    """Whether `dset` is in the layout read chunk by chunk here."""
    if dset.chunks is None or dset.compression != 'gzip' or dset.shuffle:
        return False
    if dset.id.get_create_plist().get_nfilters() != 1:
        return False
    return dset.ndim == 1 or dset.chunks[1] == 1


def _inflate(dset, offset, rows):
    mask, raw = dset.id.read_direct_chunk(offset)
    if mask:   # a filter skipped for this chunk: h5py would undo it, this cannot
        raise ValueError(f'{dset.name}: chunk {offset} has a filter mask')
    return np.frombuffer(zlib.decompress(raw), dtype=dset.dtype)[:rows]


def read(dset, pool):
    """The whole dataset, its chunks inflated on `pool`."""
    if not _raw_ok(dset):
        return dset[()]
    n = dset.shape[0]
    cols = 1 if dset.ndim == 1 else dset.shape[1]
    out = np.empty(dset.shape, dtype=dset.dtype)
    flat = out.reshape(n, cols)
    step = dset.chunks[0]

    def one(job):
        r, c = job
        k = min(step, n - r)
        flat[r:r + k, c] = _inflate(dset, (r,) if dset.ndim == 1 else (r, c), k)

    list(pool.map(one, [(r, c) for r in range(0, n, step) for c in range(cols)]))
    return out


def row_blocks(dset, pool, ahead):
    """(first row, block) over a 2-D dataset, a chunk-row at a time, with `ahead` blocks
    read in advance on `pool`."""
    n = dset.shape[0]
    if not _raw_ok(dset):
        step = dset.chunks[0] if dset.chunks else 1 << 22
        for s in range(0, n, step):
            yield s, dset[s:s + step]
        return
    step = dset.chunks[0]
    starts = list(range(0, n, step))

    def get(s):
        k = min(step, n - s)
        block = np.empty((k, dset.shape[1]), dtype=dset.dtype)
        for c in range(dset.shape[1]):
            block[:, c] = _inflate(dset, (s, c), k)
        return s, block

    pending = [pool.submit(get, s) for s in starts[:ahead]]
    nxt = len(pending)
    while pending:
        f = pending.pop(0)
        if nxt < len(starts):
            pending.append(pool.submit(get, starts[nxt]))
            nxt += 1
        yield f.result()


def pool(threads=None):
    """A thread pool for the reads: $SLURM_CPUS_PER_TASK, else every core."""
    n = threads or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) or os.cpu_count()
    return ThreadPoolExecutor(n)


def open_item(xdmf_items, key):
    """The h5py dataset of one XDMF item (the file stays open with it)."""
    h5, name = xdmf_items[key]
    return h5py.File(h5, 'r')[name]
