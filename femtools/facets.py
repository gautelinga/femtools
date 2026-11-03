"""Cut a facet file down to the selected facets.

    femtools-cut-facets <in>.xdmf <out>.xdmf --markers 3,5,7:106

writes <out>.xdmf and <out>.h5 in meshio's layout: the facets whose marker the list
selects (values and inclusive ranges), the points they use, renumbered, and their
markers as a 1-D cell attribute `markers`. The input is any facet XDMF with HDF5 data,
such as a facet MeshFunction dolfin writes, which holds every facet of the mesh."""
import argparse
import os

import numpy as np

from femtools import compress
from femtools.xdmf import grids


def selector(spec):
    """A predicate on an array of markers: `3,5,7:106` keeps 3, 5 and 7..106."""
    ranges = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        lo, _, hi = part.partition(":")
        ranges.append((int(lo), int(hi or lo)))
    if not ranges:
        raise ValueError("no markers in {!r}".format(spec))
    return lambda m: np.logical_or.reduce([(m >= a) & (m <= b) for a, b in ranges])


def read(xdmf):
    """`(points, facets, markers)` of the first grid of a facet file; `markers` the cell
    attribute named so, else the only cell attribute, else None."""
    import h5py
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


XDMF = """<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Grid" GridType="Uniform">
      <Topology TopologyType="{top}" NumberOfElements="{nf}" NodesPerElement="{nv}">
        <DataItem Dimensions="{nf} {nv}" NumberType="Int" Precision="4" Format="HDF">{h5}:/facets</DataItem>
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


def cut(infile, outfile, spec):
    """Writes the facets of `infile` that `spec` selects to `outfile` and its .h5;
    returns `(facets, points, markers)` as written."""
    import h5py
    points, facets, markers = read(infile)
    if markers is None:
        raise ValueError(infile + ": no markers to select by")
    keep = selector(spec)(markers)
    facets, markers = facets[keep], markers[keep]
    if not len(facets):
        raise ValueError("no facet carries the markers " + spec)
    # only the points the kept facets use
    used, inverse = np.unique(facets, return_inverse=True)
    facets, points = inverse.reshape(facets.shape), points[used]
    nv, gd = facets.shape[1], points.shape[1]
    h5 = os.path.splitext(outfile)[0] + ".h5"
    with h5py.File(h5, "w") as f:
        compress.put(f, "points", points.astype(np.float64))
        compress.put(f, "facets", facets.astype(np.int32))
        compress.put(f, "markers", markers.astype(np.int64))
    with open(outfile, "w") as f:
        f.write(XDMF.format(top={3: "Triangle", 2: "Polyline"}[nv], nf=len(facets), nv=nv,
                            geo={3: "XYZ", 2: "XY"}[gd], np=len(points), gd=gd,
                            h5=os.path.basename(h5)))
    return facets, points, markers


def main(argv=None):
    p = argparse.ArgumentParser(prog="femtools-cut-facets",
                                description=__doc__.splitlines()[0])
    p.add_argument("infile", help="facet file (.xdmf)")
    p.add_argument("outfile", help="cut facet file (.xdmf); its .h5 beside it")
    p.add_argument("--markers", required=True, help="markers kept, as 3,5,7:106")
    a = p.parse_args(argv)
    try:
        facets, points, markers = cut(a.infile, a.outfile, a.markers)
    except ValueError as e:
        raise SystemExit(str(e))
    print("%d facets, %d points, markers %s -> %s" % (
        len(facets), len(points), ",".join(map(str, np.unique(markers))), a.outfile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
