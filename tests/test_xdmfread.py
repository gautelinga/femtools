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
