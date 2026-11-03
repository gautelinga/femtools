"""Every HDF5 dataset written through femtools is shuffled and gzipped, level 1.

    compress.put(h5, name, a)                                          # any serial writer
    hdf5_rows.write_ordered(...)                                       # rows from many ranks
    compress.rewrite(path[, comm])  # a file other code wrote, compressed in place

numpy and h5py only. Readers need nothing: h5py, dolfin's HDF5File and ParaView decode
gzip in the HDF5 library itself.

Under MPI every rank writes its own block of whole chunks (`aligned_edges`), collectively
through MPI-IO, and HDF5 compresses each rank's chunks. Without h5py's MPI build the
compression is still spread over the ranks: `encode` turns a rank's block into the bytes
HDF5 would store (shuffle, then a zlib stream), and rank zero writes them as they are
(`write_direct_chunk`), so it only moves bytes. HDF5 stores a dataset's last, partial chunk
at full size, so it is padded with zeros, the fill value.
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
    """Every group, dataset and link under g, in order: `("g", name)`, `("d", name)`, or
    `("l", name, target)` for a soft or external link, kept as it is and not followed
    (`target` is `("soft", path)` or `("external", file, path)`). Refuses anything else."""
    import h5py
    for k in g.keys():
        name = g.name.rstrip("/") + "/" + k
        link = g.get(k, getlink=True)
        if isinstance(link, h5py.SoftLink):
            out.append(("l", name, ("soft", link.path)))
            continue
        if isinstance(link, h5py.ExternalLink):
            out.append(("l", name, ("external", link.filename, link.path)))
            continue
        v = g[k]
        if isinstance(v, h5py.Group):
            out.append(("g", name))
            _walk(v, out)
        elif isinstance(v, h5py.Dataset):
            out.append(("d", name))
        else:
            raise ValueError("{} is a {}".format(name, type(v).__name__))
    return out


def _link(target):
    import h5py
    return h5py.SoftLink(target[1]) if target[0] == "soft" else h5py.ExternalLink(*target[1:])


def _filtered(d):
    return d.id.get_create_plist().get_nfilters() > 0


def _plain(d):
    """A dataset to compress: data, no filter of any kind (one a plugin applies reads as
    no `compression` in h5py, so the creation property list is asked)."""
    return d.ndim >= 1 and d.size > 0 and d.dtype.kind not in "OUS" and not _filtered(d)


def _raw(d):
    """Copied as its stored chunks: filtered, and HDF5 can list its chunks."""
    return _filtered(d) and d.chunks is not None and hasattr(d.id, "get_num_chunks")


def _pieces(d, comm=None, pad=False):
    """Row slices of d of at most about `BLOCK` values: all of its rows serially, under
    MPI this rank's block of whole chunks (`aligned_edges`). With `pad`, as many on every
    rank (empty ones added), since a collective write takes every rank once a slice."""
    width = int(np.prod(d.shape[1:])) if d.ndim > 1 else 1
    rows = d.chunks[0] if d.chunks else 1
    step = max(rows, (BLOCK // max(width, 1)) // rows * rows)
    lo, hi = 0, d.shape[0]
    if comm is not None:
        edges = aligned_edges(d.shape[0], comm.Get_size(), rows)
        lo, hi = int(edges[comm.Get_rank()]), int(edges[comm.Get_rank() + 1])
    mine = [slice(a, min(hi, a + step)) for a in range(lo, hi, step)]
    if pad and comm is not None:
        from femtools.hdf5_rows import allreduce
        mine += [slice(hi, hi)] * (allreduce(comm, len(mine), "max") - len(mine))
    return mine


def _chunk(d, i):
    """`(offset, filter mask, stored bytes)` of the i-th stored chunk."""
    at = d.id.get_chunk_info(i).chunk_offset
    mask, data = d.id.read_direct_chunk(at)
    return at, mask, data


def _chunks(d):
    """`_chunk` of every stored chunk, in storage order."""
    for i in range(d.id.get_num_chunks()):
        yield _chunk(d, i)


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


def _survey(src):
    """`(names, plain bytes, variable-length data)` of src, read only."""
    import h5py
    with h5py.File(src, "r") as fi:
        names = _walk(fi, [])
        datasets = [fi[n] for kind, n, *_ in names if kind == "d"]
        plain = sum(d.size * d.dtype.itemsize for d in datasets if _plain(d))
        vlen = any(d.dtype.kind == "O" for d in datasets)
    return names, plain, vlen


def _create(fo, n, d):
    """Dataset n of fo for src's dataset d: compressed (`options`) if plain, else created
    as it was -- layout, filters, fill value, maxshape."""
    import h5py
    if _plain(d):
        return fo.create_dataset(n, shape=d.shape, dtype=d.dtype, maxshape=d.maxshape,
                                 fillvalue=d.fillvalue, **options(d.shape))
    return h5py.Dataset(h5py.h5d.create(fo.id, n.encode(), d.id.get_type(), d.id.get_space(),
                                        dcpl=d.id.get_create_plist()))


def _copy_chunks(fo, fi, names):
    """The filtered datasets `names` of fi into fo as their stored chunks, with their
    attributes."""
    for n in names:
        d = fi[n]
        e = _create(fo, n, d)
        for at, mask, data in _chunks(d):
            e.id.write_direct_chunk(at, data, mask)
        for k, v in d.attrs.items():
            e.attrs[k] = v


def _copy(src, dst, names, comm=None):
    """dst as src, every plain dataset compressed, links kept as links. A filtered
    dataset HDF5 can list the chunks of is copied as its stored chunks.

    With comm (MPI-IO, every rank): each rank writes its own block of every other
    dataset's rows, collectively, and HDF5 compresses the chunks. The stored chunks are
    then copied by rank zero alone, into the closed file reopened serially: HDF5 takes
    no stored chunks from several ranks at once (it hangs at close)."""
    import contextlib
    import os
    import h5py
    from femtools.abort import aborting
    from femtools.hdf5_rows import collective_write
    rank0 = comm is None or comm.Get_rank() == 0
    raw = []
    mpi = {} if comm is None else {"driver": "mpio", "comm": comm}

    def drop():             # by whichever rank failed: the others are held in the write
        if os.path.exists(dst):
            os.remove(dst)

    # under MPI-IO a failure aborts the job, the copy removed first: the abort inside
    # the file, whose collective close would otherwise hold the failing rank
    collective = (lambda: aborting(comm, drop)) if comm is not None else contextlib.nullcontext
    with h5py.File(src, "r") as fi:
        with h5py.File(dst, "w", rdcc_nbytes=64 << 20, rdcc_nslots=10007, **mpi) as fo, \
                collective():
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            for kind, n, *target in names:
                if kind == "l":
                    fo[n] = _link(target[0])
                    continue
                d = fi[n]
                if kind == "g":
                    e = fo.create_group(n)
                elif _raw(d):
                    raw.append(n)
                    if comm is not None:
                        continue
                    _copy_chunks(fo, fi, [n])
                    continue
                else:
                    e = _create(fo, n, d)
                    if d.size and d.ndim:
                        for s in _pieces(e, comm, pad=True):
                            if comm is None:
                                e[s] = d[s]
                            else:
                                collective_write(e, s.start, d[s])
                    elif d.size:
                        e[()] = d[()]
                for k, v in d.attrs.items():
                    e.attrs[k] = v
        if comm is not None and raw:
            with collective():
                if rank0:
                    with h5py.File(dst, "r+") as fo:
                        _copy_chunks(fo, fi, raw)
                comm.Barrier()


def _check(src, dst, names, comm=None):
    """The first difference between src and dst, or None: structure, links, attributes,
    and every dataset's values, a stored-chunk dataset's stored bytes. With comm, each
    rank compares its own share of the rows and chunks, and every rank gets the answer."""
    import h5py
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    bad = None
    with h5py.File(src, "r") as fi, h5py.File(dst, "r") as fo:
        if _walk(fo, []) != names or not _same_attrs(fi.attrs, fo.attrs):
            bad = "the structure or the file's attributes differ"
        for kind, n, *_ in ([] if bad else names):
            if kind == "l":
                continue
            d, e = fi[n], fo[n]
            if not _same_attrs(d.attrs, e.attrs):
                bad = "{}'s attributes differ".format(n)
            elif kind == "g":
                continue
            elif d.dtype != e.dtype or d.shape != e.shape or d.maxshape != e.maxshape:
                bad = "{}: dtype, shape or maxshape differs".format(n)
            elif _raw(d):
                if d.id.get_num_chunks() != e.id.get_num_chunks():
                    bad = "{}: the number of stored chunks differs".format(n)
                elif any(_chunk(d, i) != _chunk(e, i)
                         for i in range(rank, d.id.get_num_chunks(), size)):
                    bad = "{}: a stored chunk differs".format(n)
            elif d.ndim == 0 or d.size == 0:
                if np.asarray(d[()]).tobytes() != np.asarray(e[()]).tobytes():
                    bad = "{} differs".format(n)
            else:
                for s in _pieces(e, comm):
                    if (np.ascontiguousarray(d[s]).tobytes()
                            != np.ascontiguousarray(e[s]).tobytes()):
                        bad = "{} differs in rows {}..{}".format(n, s.start, s.stop)
                        break
            if bad:
                break
    if comm is None:
        return bad
    found = [b for b in comm.allgather(bad) if b]
    return found[0] if found else None


def rewrite(path, comm=None, share=None):
    """`path` rewritten with every dataset not yet compressed compressed (`options`), in
    place: written beside it, every dataset and attribute read back from both and compared
    bit for bit, the original's mtime and mode put on the copy, then `os.replace`, so a
    reader meets one file or the other. A plain dataset keeps its maxshape and fill value;
    a filtered one is created as it was and copied as its stored chunks, compared stored
    byte for stored byte; a soft or external link is kept as the link. Skips a symlink, a
    file with other hard links (the old copy would stay alive), and a file whose plain
    datasets are under `WORTH_BYTES` or `WORTH_SHARE` of it, so a rerun does nothing.
    Returns `(code, message)`: 0 rewritten, 1 failed (the original kept), SKIPPED. For
    files other code writes, and for files written before compression. `share` replaces
    `WORTH_SHARE`: a file appended to and rewritten after each append (an XDMF checkpoint
    at each close) is copied whole each time, so a larger share -- a rewrite once the
    plain part has grown that much -- keeps the copying a few times the final size.

    With `comm`, collective, and every rank gets the result: through MPI-IO when h5py
    has it (`hdf5_rows.mpio`), every rank copying and comparing its own share of each
    dataset; otherwise, or for a file with variable-length data (which parallel HDF5
    cannot write), by rank zero. A failure while the ranks write together aborts the job
    (`abort.aborting`), the copy removed."""
    from femtools.hdf5_rows import mpio
    share = WORTH_SHARE if share is None else float(share)
    if mpio(comm):
        return _rewrite(path, comm, share)
    if comm is None or comm.Get_size() == 1:
        return _rewrite(path, None, share)
    return comm.bcast(_rewrite(path, None, share) if comm.Get_rank() == 0 else None, root=0)


def _rewrite(path, comm, share=WORTH_SHARE):
    """`rewrite` by this process alone (comm None), or by every rank of comm (MPI-IO)."""
    import os
    import time
    from femtools.abort import aborting
    t = time.time()
    tmp = path + SUFFIX
    rank0 = comm is None or comm.Get_rank() == 0
    if os.path.islink(path):
        return SKIPPED, "a symlink"
    st = os.stat(path)
    if st.st_nlink > 1:
        return SKIPPED, "{} hard links".format(st.st_nlink)
    # read only, and alike on every rank: what is refused here is answered, not raised
    try:
        names, plain, vlen = _survey(path)
    except Exception as e:                  # noqa: BLE001 -- reported; the original kept
        return 1, "NOT REPLACED: {}: {}".format(type(e).__name__, e)
    if plain < max(WORTH_BYTES, share * st.st_size):
        return SKIPPED, ("nothing to compress" if plain == 0 else
                         "nothing worth compressing ({:.0f} KiB plain)".format(plain / 1024))
    if comm is not None and vlen:
        return comm.bcast(_rewrite(path, None, share) if rank0 else None, root=0)

    def replace():
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.chmod(tmp, st.st_mode & 0o7777)
        after = os.path.getsize(tmp)
        os.replace(tmp, path)
        return after

    if comm is None:
        try:
            _copy(path, tmp, names)
            bad = _check(path, tmp, names)
            after = None if bad else replace()
        except Exception as e:              # noqa: BLE001 -- reported; the original kept
            bad = "{}: {}".format(type(e).__name__, e)
        if bad and os.path.exists(tmp):
            os.remove(tmp)
    else:
        _copy(path, tmp, names, comm)       # aborts by itself, inside its MPI-IO file

        def drop():
            if os.path.exists(tmp):
                os.remove(tmp)

        with aborting(comm, drop):
            bad = _check(path, tmp, names, comm)
            after = None if bad or not rank0 else replace()
            after = comm.bcast(after, root=0)
        if bad and rank0:
            os.remove(tmp)
    if bad:
        return 1, "NOT REPLACED: " + bad
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
