"""`femtools.xdmfread`: datasets read with their gzip chunks inflated on a thread pool,
whole and a chunk-row at a time, give what meshio wrote."""
import numpy as np
import pytest


def test_xdmfread_reads_what_meshio_reads(tmp_path):
    meshio = pytest.importorskip("meshio")

    from femtools import xdmfread
    rng = np.random.default_rng(4)
    pts, cells = rng.random((5000, 3)), rng.integers(0, 5000, (20000, 4)).astype(np.int32)
    meshio.write(tmp_path / 'm.xdmf', meshio.Mesh(pts, [('tetra', cells)]))
    it = xdmfread.items(tmp_path / 'm.xdmf')
    pool = xdmfread.pool(4)
    tets = xdmfread.open_item(it, 'topology')
    assert np.array_equal(xdmfread.read(xdmfread.open_item(it, 'geometry'), pool), pts)
    assert np.array_equal(xdmfread.read(tets, pool), cells)
    blocks = [b for _, b in xdmfread.row_blocks(tets, pool, ahead=3)]
    assert np.array_equal(np.concatenate(blocks), cells)


@pytest.mark.parametrize("layout", ["femtools", "meshio"])
@pytest.mark.parametrize("shape,dtype", [((100_003,), "f8"), ((70_001, 3), "f8"),
                                         ((50_000, 4), "i8"), ((9_999, 10), "i4")])
def test_the_fast_path_reads_what_h5py_reads(tmp_path, layout, shape, dtype):
    """Shuffled and gzipped by `femtools.compress` (whole rows a chunk), or gzipped a
    column a chunk as meshio writes: the chunks inflated here are h5py's arrays, read
    whole or a chunk-row at a time however far ahead -- 0 included."""
    import h5py
    from femtools import compress, xdmfread
    rng = np.random.default_rng(7)
    a = (rng.random(shape) * 1000).astype(dtype)
    path = str(tmp_path / "f.h5")
    with h5py.File(path, "w") as h:
        if layout == "femtools":
            compress.put(h, "x", a)
        else:
            chunks = (4096,) if a.ndim == 1 else (4096, 1)
            h.create_dataset("x", data=a, chunks=chunks, compression="gzip")
    pool = xdmfread.pool(2)
    with h5py.File(path, "r") as h:
        d = h["x"]
        assert xdmfread._raw_ok(d)
        assert np.array_equal(xdmfread.read(d, pool), a)
        if a.ndim == 2:
            for ahead in (0, 2):
                blocks = list(xdmfread.row_blocks(d, pool, ahead))
                assert np.array_equal(np.vstack([b for _, b in blocks]), a)
                assert [s for s, _ in blocks] == list(range(0, len(a), d.chunks[0]))
