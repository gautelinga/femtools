"""Cut a facet file down to the selected facets.

    femtools-cut-facets <in>.xdmf <out>.xdmf --markers 3,5,7:106
    femtools-cut-facets <in>.xdmf <out>.xdmf --markers 1:

writes <out>.xdmf and <out>.h5 in `femtools.boundaries`' layout: the facets whose marker
the list selects (values, inclusive ranges and open ranges `a:`, every marker from a), the
points they use, renumbered, and their
markers as a 1-D cell attribute `markers`. The input is any facet XDMF with HDF5 data,
such as a facet MeshFunction dolfin writes, which holds every facet of the mesh."""
import argparse

import numpy as np

from femtools import boundaries
from femtools.boundaries import read


def selector(spec):
    """A predicate on an array of markers: `3,5,7:106` keeps 3, 5 and 7..106, `1:` every
    marker from 1."""
    ranges = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        lo, colon, hi = part.partition(":")
        hi = hi.strip()
        ranges.append((int(lo), int(hi) if hi else None if colon else int(lo)))
    if not ranges:
        raise ValueError("no markers in {!r}".format(spec))
    return lambda m: np.logical_or.reduce([(m >= a) & ((m <= b) if b is not None else True)
                                           for a, b in ranges])


def cut(infile, outfile, spec):
    """Writes the facets of `infile` that `spec` selects to `outfile` and its .h5;
    returns `(facets, points, markers)` as written."""
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
    boundaries.write(outfile, points, facets, markers)
    return facets, points, markers


def main(argv=None):
    p = argparse.ArgumentParser(prog="femtools-cut-facets",
                                description=__doc__.splitlines()[0])
    p.add_argument("infile", help="facet file (.xdmf)")
    p.add_argument("outfile", help="cut facet file (.xdmf); its .h5 beside it")
    p.add_argument("--markers", required=True, help="markers kept, as 3,5,7:106; 1: every marker from 1")
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
