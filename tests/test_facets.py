"""`femtools.facets`: a facet file cut to the selected markers keeps exactly those
facets, renumbers them onto the points they use, and carries their markers."""
import itertools
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from femtools.facets import cut, main, read, selector  # noqa: E402
from femtools.xdmf import XdmfDocument  # noqa: E402


def cube_facets():
    """The facets of the unit cube in six tetrahedra: the boundary ones marked 1..6 by
    face (x=0, x=1, y=0, y=1, z=0, z=1), the interior ones 0."""
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    tets = []
    for perm in itertools.permutations(range(3)):
        x, path = np.zeros(3), [0]
        for axis in perm:
            x[axis] = 1
            path.append(int(x[0] * 4 + x[1] * 2 + x[2]))
        tets.append(path)
    count = {}
    for t in tets:
        for f in itertools.combinations(sorted(t), 3):
            count[f] = count.get(f, 0) + 1
    facets = np.array(sorted(count), dtype=np.int64)
    markers = np.zeros(len(facets), dtype=np.uint64)
    for i, f in enumerate(facets):
        if count[tuple(f)] == 1:
            p = corners[f]
            axis = int(np.flatnonzero(np.ptp(p, axis=0) == 0)[0])
            markers[i] = 1 + 2 * axis + int(p[0, axis])
    return corners, facets, markers


def write_input(path, corners, facets, markers, name="f"):
    doc = XdmfDocument(path)
    grid = doc.grid("mesh")
    doc.topology(grid, facets, "Triangle", ref=doc.dataset("Mesh/0/topology", facets))
    doc.geometry(grid, corners, ref=doc.dataset("Mesh/0/geometry", corners))
    doc.attribute(grid, name, markers, "Cell", ref=doc.dataset("MeshFunction/0/values",
                                                                  markers))
    doc.close()


def test_selector():
    m = np.arange(12)
    assert np.flatnonzero(selector("3, 5,7:9")(m)).tolist() == [3, 5, 7, 8, 9]
    assert np.flatnonzero(selector("2,9:")(m)).tolist() == [2, 9, 10, 11]
    assert np.flatnonzero(selector("1:")(np.arange(12, dtype=np.uint64))).tolist() == \
        list(range(1, 12))


def test_an_open_range_cuts_a_full_file_to_its_nonzero_markers(tmp_path):
    """`--markers 1:` on a full facet MeshFunction keeps every facet marked nonzero, on
    the points they use, with their markers."""
    corners, facets, markers = cube_facets()
    src, out = str(tmp_path / "full.xdmf"), str(tmp_path / "boundary.xdmf")
    write_input(src, corners, facets, markers)
    assert main([src, out, "--markers", "1:"]) == 0
    points, got, kept = read(out)
    want = markers != 0
    assert len(got) == want.sum() == 12 and set(kept.tolist()) == set(range(1, 7))
    assert np.array_equal(np.unique(got), np.arange(len(points)))
    assert len(points) == len(np.unique(facets[want])) == 8
    before = {tuple(sorted(map(tuple, corners[f]))): int(m)
              for f, m in zip(facets[want], markers[want])}
    assert before == {tuple(sorted(map(tuple, points[f]))): int(m) for f, m in zip(got, kept)}


def test_cut_keeps_the_selected_facets_on_their_points(tmp_path):
    corners, facets, markers = cube_facets()
    assert (markers == 0).sum() > 0 and set(markers.tolist()) == set(range(7))
    src, out = str(tmp_path / "facets.xdmf"), str(tmp_path / "walls.xdmf")
    write_input(src, corners, facets, markers)
    assert main([src, out, "--markers", "1,5:6"]) == 0
    assert os.path.exists(str(tmp_path / "walls.h5"))
    points, got, kept = read(out)
    want = np.isin(markers, [1, 5, 6])
    assert len(got) == want.sum() == 6 and kept.dtype == np.int64
    assert sorted(kept.tolist()) == sorted(markers[want].tolist())
    # compacted: every point used, none twice
    assert np.array_equal(np.unique(got), np.arange(len(points)))
    assert len(np.unique(points, axis=0)) == len(points)
    # the same triangles, by position, with their markers
    before = {tuple(sorted(map(tuple, corners[f]))): int(m)
              for f, m in zip(facets[want], markers[want])}
    after = {tuple(sorted(map(tuple, points[f]))): int(m) for f, m in zip(got, kept)}
    assert before == after
    # only the points the kept facets use
    assert len(points) == len(np.unique(facets[want]))


def test_markers_by_name_and_refusals(tmp_path):
    corners, facets, markers = cube_facets()
    src = str(tmp_path / "named.xdmf")
    write_input(src, corners, facets, markers, name="markers")
    assert np.array_equal(read(src)[2], markers)
    with pytest.raises(ValueError, match="no facet carries"):
        cut(src, str(tmp_path / "none.xdmf"), "40")
