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

The layouts meshio, mesh-sphere-packing and femtools write are taken this way: chunked, one
or two dimensions, gzip alone or shuffle then gzip (`femtools.compress`). Any other
dataset is read by h5py as before, so the answers never depend on which path was taken.
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


def _filters(dset):
    """The filter ids of `dset`'s pipeline, in order."""
    plist = dset.id.get_create_plist()
    return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]


def _raw_ok(dset):
    """Whether `dset` is in a layout read chunk by chunk here: chunked, gzip alone or
    shuffle then gzip (`femtools.compress`'s), one or two dimensions."""
    if dset.chunks is None or dset.ndim not in (1, 2):
        return False
    return _filters(dset) in ([h5py.h5z.FILTER_DEFLATE],
                              [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE])


def _inflate(dset, offset):
    """The chunk at `offset`, inflated (and unshuffled) to its full chunk shape: a
    dataset's last chunks are stored at full size, padded."""
    mask, raw = dset.id.read_direct_chunk(offset)
    if mask:   # a filter skipped for this chunk: h5py would undo it, this cannot
        raise ValueError(f'{dset.name}: chunk {offset} has a filter mask')
    data = np.frombuffer(zlib.decompress(raw), dtype=np.uint8)
    size = dset.dtype.itemsize
    if h5py.h5z.FILTER_SHUFFLE in _filters(dset) and size > 1:
        # HDF5's shuffle: every value's first byte, then every second, and so on
        data = np.ascontiguousarray(data.reshape(size, -1).T)
    return data.view(dset.dtype).reshape(dset.chunks)


def _chunk_into(dset, out, r, c):
    """The chunk at row r (column c) of `dset` into `out` (the dataset's shape, or its
    rows from r on), trimmed to the dataset."""
    n = dset.shape[0]
    chunk = _inflate(dset, (r,) if dset.ndim == 1 else (r, c))
    k = min(dset.chunks[0], n - r)
    if dset.ndim == 1:
        out[:k] = chunk[:k]
    else:
        w = min(dset.chunks[1], dset.shape[1] - c)
        out[:k, c:c + w] = chunk[:k, :w]


def read(dset, pool):
    """The whole dataset, its chunks inflated on `pool`."""
    if not _raw_ok(dset):
        return dset[()]
    out = np.empty(dset.shape, dtype=dset.dtype)
    n = dset.shape[0]
    cols = [0] if dset.ndim == 1 else range(0, dset.shape[1], dset.chunks[1])
    jobs = [(r, c) for r in range(0, n, dset.chunks[0]) for c in cols]
    list(pool.map(lambda job: _chunk_into(dset, out[job[0]:], *job), jobs))
    return out


def row_blocks(dset, pool, ahead):
    """(first row, block) over a 2-D dataset, a chunk-row at a time, with `ahead` blocks
    read in advance on `pool` (at least the one being read)."""
    n = dset.shape[0]
    if not _raw_ok(dset):
        step = dset.chunks[0] if dset.chunks else 1 << 22
        for s in range(0, n, step):
            yield s, dset[s:s + step]
        return
    step = dset.chunks[0]
    starts = list(range(0, n, step))

    def get(s):
        block = np.empty((min(step, n - s), dset.shape[1]), dtype=dset.dtype)
        for c in range(0, dset.shape[1], dset.chunks[1]):
            _chunk_into(dset, block, s, c)
        return s, block

    pending = [pool.submit(get, s) for s in starts[:max(int(ahead), 1)]]
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
