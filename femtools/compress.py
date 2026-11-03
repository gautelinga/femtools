"""Every HDF5 dataset written through femtools is shuffled and gzipped, level 1.

    compress.put(h5, name, a)                                          # any serial writer
    hdf5_rows.write_ordered(...)                                       # rows from many ranks
    compress.rewrite(path)          # a file other code wrote, compressed in place

numpy and h5py only. Readers need nothing: h5py, dolfin's HDF5File and ParaView decode
gzip in the HDF5 library itself.

Under MPI the compression is spread over the ranks: `encode` turns a rank's block of whole
chunks into the bytes HDF5 would store (shuffle, then a zlib stream), and rank zero writes
them as they are (`write_direct_chunk`), so it only moves bytes. A block must start on a
chunk boundary (`aligned_edges`); HDF5 stores a dataset's last, partial chunk at full size,
so it is padded with zeros, the fill value.
"""
import zlib

import numpy as np

LEVEL = 1
CHUNK = 1 << 20            # values a chunk: ~8 MB of float64


def chunk_rows(shape):
    """Rows a chunk of a dataset of this shape holds (whole rows, at least one)."""
    width = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    return max(1, min(int(shape[0]), CHUNK // max(width, 1)))


def options(shape):
    """create_dataset keywords for a dataset of this shape: chunked, shuffled, gzipped.
    Nothing for a scalar or an empty dataset, which HDF5 cannot chunk."""
    shape = tuple(int(s) for s in shape)
    if not shape or shape[0] == 0 or 0 in shape:
        return {}
    return {"chunks": (chunk_rows(shape),) + shape[1:], "compression": "gzip",
            "compression_opts": LEVEL, "shuffle": True}


def put(group, name, data):
    """`group[name] = data`, compressed: the one line every serial writer uses."""
    a = np.asarray(data)
    return group.create_dataset(name, data=a, **(options(a.shape) if a.dtype.kind not in
                                                  "OUS" else {}))


def aligned_edges(n, size, rows):
    """`size + 1` block edges over n rows, each on a multiple of `rows` (the last at n)."""
    k = -(-int(n) // int(rows))                     # chunks
    return np.array([min(int(n), (k * r // size) * rows) for r in range(size + 1)],
                    dtype=np.int64)


def encode(block, rows):
    """The stored bytes of each chunk of `block`, whose first row starts a chunk of `rows`
    rows: a list, the last chunk padded to full size. Shuffle is HDF5's: every element's
    first byte, then every second byte, and so on, over the whole chunk."""
    block = np.ascontiguousarray(block)
    out = []
    for lo in range(0, len(block), rows):
        c = block[lo:lo + rows]
        if len(c) < rows:
            c = np.concatenate([c, np.zeros((rows - len(c),) + c.shape[1:], c.dtype)])
        b = c.reshape(-1).view(np.uint8).reshape(-1, c.dtype.itemsize)
        out.append(zlib.compress(np.ascontiguousarray(b.T).tobytes(), LEVEL))
    return out


# -- an existing file, rewritten in place ---------------------------------------------------

SKIPPED = 2
SUFFIX = ".compress-tmp"
BLOCK = 1 << 24            # values copied and compared at a time (128 MB of float64)
# a file is rewritten only when its plain datasets are at least this much of it: a slice
# store's part, gzipped but for a few offsets, would be read and written whole for nothing
WORTH_BYTES = 1 << 20
WORTH_SHARE = 0.01


def _walk(g, out):
    """Every group and dataset under g, in order; refuses links and anything else."""
    import h5py
    for k, v in g.items():
        if not isinstance(g.get(k, getlink=True), h5py.HardLink):
            raise ValueError("{}/{} is a link".format(g.name, k))
        if isinstance(v, h5py.Group):
            out.append(("g", v.name))
            _walk(v, out)
        elif isinstance(v, h5py.Dataset):
            out.append(("d", v.name))
        else:
            raise ValueError("{} is a {}".format(v.name, type(v).__name__))
    return out


def _filtered(d):
    return d.id.get_create_plist().get_nfilters() > 0


def _plain(d):
    """A dataset to compress: data, no filter of any kind (one a plugin applies reads as
    no `compression` in h5py, so the creation property list is asked)."""
    return d.ndim >= 1 and d.size > 0 and d.dtype.kind not in "OUS" and not _filtered(d)


def _raw(d):
    """Copied as its stored chunks: filtered, and HDF5 can list its chunks."""
    return _filtered(d) and d.chunks is not None and hasattr(d.id, "get_num_chunks")


def _rows(d):
    width = int(np.prod(d.shape[1:])) if d.ndim > 1 else 1
    step = max(1, BLOCK // max(width, 1))
    for lo in range(0, d.shape[0], step):
        yield slice(lo, min(d.shape[0], lo + step))


def _chunks(d):
    """`(offset, filter mask, stored bytes)` of every stored chunk, in storage order."""
    for i in range(d.id.get_num_chunks()):
        at = d.id.get_chunk_info(i).chunk_offset
        mask, data = d.id.read_direct_chunk(at)
        yield at, mask, data


def _same_attrs(a, b):
    """By the bytes, so a NaN equals itself."""
    if sorted(a.keys()) != sorted(b.keys()):
        return False
    for k in a.keys():
        x, y = a[k], b[k]
        if type(x) is not type(y):
            return False
        if isinstance(x, (str, bytes)):
            if x != y:
                return False
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def _copy(src, dst):
    """`(names, plain bytes)`; dst written only when the plain bytes are worth it."""
    import os
    import h5py
    with h5py.File(src, "r") as fi:
        names = _walk(fi, [])
        plain = sum(fi[n].size * fi[n].dtype.itemsize for kind, n in names
                    if kind == "d" and _plain(fi[n]))
        if plain < max(WORTH_BYTES, WORTH_SHARE * os.path.getsize(src)):
            return None, plain
        with h5py.File(dst, "w", rdcc_nbytes=64 << 20, rdcc_nslots=10007) as fo:
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            for kind, n in names:
                d = fi[n]
                if kind == "g":
                    e = fo.create_group(n)
                elif _plain(d):
                    e = fo.create_dataset(n, shape=d.shape, dtype=d.dtype, maxshape=d.maxshape,
                                          fillvalue=d.fillvalue, **options(d.shape))
                    for s in _rows(d):
                        e[s] = d[s]
                else:       # as it was created: layout, filters, fill value, maxshape
                    e = h5py.Dataset(h5py.h5d.create(fo.id, n.encode(), d.id.get_type(),
                                                     d.id.get_space(),
                                                     dcpl=d.id.get_create_plist()))
                    if _raw(d):
                        for at, mask, data in _chunks(d):
                            e.id.write_direct_chunk(at, data, mask)
                    elif d.size and d.ndim:
                        for s in _rows(d):
                            e[s] = d[s]
                    elif d.size:
                        e[()] = d[()]
                for k, v in d.attrs.items():
                    e.attrs[k] = v
    return names, plain


def _check(src, dst, names):
    import h5py
    with h5py.File(src, "r") as fi, h5py.File(dst, "r") as fo:
        if _walk(fo, []) != names or not _same_attrs(fi.attrs, fo.attrs):
            return "the structure or the file's attributes differ"
        for kind, n in names:
            d, e = fi[n], fo[n]
            if not _same_attrs(d.attrs, e.attrs):
                return "{}'s attributes differ".format(n)
            if kind == "g":
                continue
            if d.dtype != e.dtype or d.shape != e.shape or d.maxshape != e.maxshape:
                return "{}: dtype, shape or maxshape differs".format(n)
            if _raw(d):
                # chunk by chunk: holding both datasets' stored bytes doubled the peak
                if d.id.get_num_chunks() != e.id.get_num_chunks() or any(
                        a != b for a, b in zip(_chunks(d), _chunks(e))):
                    return "{}: a stored chunk differs".format(n)
                continue
            if d.ndim == 0 or d.size == 0:
                if np.asarray(d[()]).tobytes() != np.asarray(e[()]).tobytes():
                    return "{} differs".format(n)
                continue
            for s in _rows(d):
                if np.ascontiguousarray(d[s]).tobytes() != np.ascontiguousarray(e[s]).tobytes():
                    return "{} differs in rows {}..{}".format(n, s.start, s.stop)
    return None


def rewrite(path):
    """`path` rewritten with every dataset not yet compressed compressed (`options`), in
    place: written beside it, every dataset and attribute read back from both and compared
    bit for bit, the original's mtime and mode put on the copy, then `os.replace`, so a
    reader meets one file or the other. A plain dataset keeps its maxshape and fill value;
    a filtered one is created as it was and copied as its stored chunks, compared stored
    byte for stored byte. Skips a symlink, a file with other hard links (the old copy
    would stay alive), and a file whose plain datasets are under `WORTH_BYTES` or
    `WORTH_SHARE` of it, so a rerun does nothing. Returns `(code, message)`: 0 rewritten, 1 failed (the original kept),
    SKIPPED. For files other code writes, and for files written before compression."""
    import os
    import time
    t = time.time()
    tmp = path + SUFFIX
    if os.path.islink(path):
        return SKIPPED, "a symlink"
    st = os.stat(path)
    if st.st_nlink > 1:
        return SKIPPED, "{} hard links".format(st.st_nlink)
    try:
        names, plain = _copy(path, tmp)
        if names is None:
            return SKIPPED, ("nothing to compress" if plain == 0 else
                             "nothing worth compressing ({:.0f} KiB plain)".format(plain / 1024))
        bad = _check(path, tmp, names)
        if bad:
            os.remove(tmp)
            return 1, "NOT REPLACED: " + bad
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.chmod(tmp, st.st_mode & 0o7777)
        after = os.path.getsize(tmp)
        os.replace(tmp, path)
    except Exception as e:                  # noqa: BLE001 -- reported; the original kept
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1, "NOT REPLACED: {}: {}".format(type(e).__name__, e)
    return 0, "{:.2f} -> {:.2f} GiB, x{:.2f}, {:.0f} s".format(
        st.st_size / 2**30, after / 2**30, st.st_size / after, time.time() - t)


def _one(path):
    return (path,) + rewrite(path)


def main(argv=None):
    """`compress-h5 FILE... [--list F] [--jobs P] [--dry]`: `rewrite` each, largest first."""
    import argparse
    import os
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(
        prog="compress-h5", description="Rewrite HDF5 files in place with every dataset "
        "compressed, each checked bit for bit before it replaces the original.")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--list", help="a file naming more files, one a line")
    ap.add_argument("--jobs", type=int, default=1, help="files at a time")
    ap.add_argument("--dry", action="store_true", help="list the files, largest first")
    a = ap.parse_args(argv)
    named = list(a.files)
    if a.list:
        with open(a.list) as f:
            named += [line.strip() for line in f if line.strip()]
    if not named:
        ap.error("no files")
    files = sorted(set(named), key=lambda p: -os.path.getsize(p))
    total = sum(os.path.getsize(p) for p in files)
    print("{} files, {:.1f} GiB".format(len(files), total / 2**30), flush=True)
    if a.dry:
        for p in files:
            print("  {:8.2f} GiB  {}".format(os.path.getsize(p) / 2**30, p))
        return 0
    failed = 0
    with Pool(a.jobs) as pool:
        for path, code, what in pool.imap_unordered(_one, files):
            failed += code == 1
            print("  {} {}: {}".format({0: "ok  ", 1: "FAIL", SKIPPED: "skip"}[code], path,
                                       what), flush=True)
    after = sum(os.path.getsize(p) for p in files)
    print("{:.1f} -> {:.1f} GiB; {} failed".format(total / 2**30, after / 2**30, failed))
    return 1 if failed else 0
