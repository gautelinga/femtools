"""A boundary facet file: facets, each with an integer marker, on the points they use.

    write(path, points, facets, markers, comm=None)
    points, facets, markers = read(path)

`path` (.xdmf) and the .h5 beside it, meshio's layout: one uniform grid, `points`
(float64, gdim a row), `facets` (indices into points, int32 where they fit, else int64:
triangles in 3D, segments in 2D) and `markers` (int64) as a 1-D cell attribute named
`markers`, every dataset compressed (`femtools.compress`). The layout partrac's
`wall_facets` reads, and `femtools-cut-facets` writes. numpy and h5py; mpi4py only where
a communicator is passed, dolfin never.
"""
import os

import numpy as np

XDMF = """<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Grid" GridType="Uniform">
      <Topology TopologyType="{top}" NumberOfElements="{nf}" NodesPerElement="{nv}">
        <DataItem Dimensions="{nf} {nv}" NumberType="Int" Precision="{ip}" Format="HDF">{h5}:/facets</DataItem>
      </Topology>
      <Geometry GeometryType="{geo}">
        <DataItem Dimensions="{np} {gd}" NumberType="Float" Precision="8" Format="HDF">{h5}:/points</DataItem>
      </Geometry>
      <Attribute Name="markers" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{nf}" NumberType="Int" Precision="8" Format="HDF">{h5}:/markers</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
TOPOLOGY = {3: "Triangle", 2: "Polyline"}
GEOMETRY = {3: "XYZ", 2: "XY"}


def h5_path(path):
    """The .h5 beside an .xdmf."""
    return os.path.splitext(path)[0] + ".h5"


def write(path, points, facets, markers, comm=None):
    """`path` and its .h5 from `points`, `facets` and `markers`. Under MPI each rank
    passes a block of rows of each array (any of them empty), and the file holds the
    blocks in rank order; `facets` index the whole points array. Collective; returns
    `(facets, points)` counts."""
    import h5py
    from femtools import compress, hdf5_rows
    points = np.asarray(points, dtype=np.float64)
    facets, markers = np.asarray(facets), np.asarray(markers, dtype=np.int64).ravel()
    size = 1 if comm is None else comm.Get_size()
    mine = (len(points), len(facets), facets.shape[1], points.shape[1])
    shares = np.array([mine] if size == 1 else comm.allgather(mine), dtype=np.int64)
    n_points, n_facets = (int(v) for v in shares[:, :2].sum(axis=0))
    nv, gd = (int(v) for v in shares[:, 2:].max(axis=0))
    # each rank's first row of points and of facets
    first = np.cumsum(shares[:, :2], axis=0) - shares[:, :2]
    bad = len(facets) != len(markers) or nv not in TOPOLOGY or gd not in GEOMETRY
    if hdf5_rows.agreed(comm, bad):
        raise ValueError("{}: {} facets and {} markers, facets of {} points in {}D"
                         .format(path, len(facets), len(markers), nv, gd))
    wide = n_points > np.iinfo(np.int32).max
    facets = facets.astype(np.int64 if wide else np.int32)
    out = h5_path(path)
    if size == 1:
        with h5py.File(out, "w") as h5:
            compress.put(h5, "points", points)
            compress.put(h5, "facets", facets)
            compress.put(h5, "markers", markers)
    else:
        rank = comm.Get_rank()
        for k, (name, a, n, at) in enumerate((("points", points, n_points, first[rank, 0]),
                                              ("facets", facets, n_facets, first[rank, 1]),
                                              ("markers", markers, n_facets, first[rank, 1]))):
            hdf5_rows.write_ordered(out, name, at + np.arange(len(a)), a, n, comm,
                                    mode="w" if k == 0 else "a")
    if comm is None or comm.Get_rank() == 0:
        with open(path, "w") as f:
            f.write(XDMF.format(top=TOPOLOGY[nv], nf=n_facets, nv=nv, ip=8 if wide else 4,
                                geo=GEOMETRY[gd], np=n_points, gd=gd,
                                h5=os.path.basename(out)))
    if size > 1:
        comm.Barrier()
    return n_facets, n_points


def read(xdmf):
    """`(points, facets, markers)` of the first grid of a facet file; `markers` the cell
    attribute named so, else the only cell attribute, else None. Reads this module's
    files and any facet XDMF over HDF5, such as a facet MeshFunction dolfin writes."""
    import h5py
    from femtools.xdmf import grids
    found = grids(xdmf)
    if not found:
        raise ValueError(xdmf + ": no grid")
    grid = found[0]
    cells = [n for n in grid["attributes"] if (grid["centers"][n] or "Cell") == "Cell"]
    if "markers" in cells:
        name = "markers"
    elif len(cells) == 1:
        name = cells[0]
    elif cells:
        raise ValueError("{} cell attributes, none named markers".format(len(cells)))
    else:
        name = None

    def load(where):
        with h5py.File(where[0], "r") as h5:
            return np.asarray(h5[where[1].lstrip("/")])
    points, facets = load(grid["geometry"]), load(grid["topology"])
    markers = None if name is None else load(grid["attributes"][name]).ravel()
    return points, facets, markers

