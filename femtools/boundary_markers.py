"""A facet MeshFunction that is written and read as a boundary facet file.

    markers = BoundaryMarkers(mesh)
    markers.mark(subdomain, 3)
    bc = DirichletBC(V, g, markers, 3)
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    markers.write("boundaries.xdmf")
    markers = BoundaryMarkers.read("boundaries.xdmf", mesh)

`BoundaryMarkers` is a `dolfin.cpp.mesh.MeshFunctionSizet` on the facets, so it goes
wherever one does. The file (`femtools.boundaries`) holds the exterior facets with a
nonzero marker, each written by the rank whose cell holds it, on their vertices, one row
a global vertex, in global order. Reading matches its facets to the mesh's exterior
facets by vertex coordinates, on any number of ranks. Imports dolfin.
"""
import numpy as np

import dolfin as df

from femtools import boundaries, hdf5_rows

# rows a read
BLOCK = 1 << 20
# a point matches a vertex within this much of the shortest exterior edge here
REL_TOL = 1e-3


def _comm(mesh):
    """The mesh's communicator, None on one rank."""
    comm = mesh.mpi_comm()
    return None if comm.Get_size() == 1 else comm


def _exterior(mesh):
    """`(facets, vertices, owned)`: the local exterior facets, ghosts included, their
    local vertices (a row each) and whether a regular cell here holds each."""
    D = mesh.topology().dim()
    mesh.init(D - 1, D)
    mesh.init(D - 1, 0)
    on = df.MeshFunction("size_t", mesh, D - 1, 0)
    df.DomainBoundary().mark(on, 1)
    facets = np.flatnonzero(on.array())
    vertices = np.array(mesh.topology()(D - 1, 0)(), dtype=np.int64).reshape(-1, D)[facets]
    cell_facets = np.array(mesh.topology()(D, D - 1)(), dtype=np.int64).reshape(-1, D + 1)
    regular = np.zeros(mesh.num_facets(), dtype=bool)
    regular[cell_facets[:mesh.topology().ghost_offset(D)].ravel()] = True
    return facets, vertices, regular[facets]


def _row_keys(rows):
    """Each row's sorted entries as one structured value, for sorting and matching."""
    rows = np.ascontiguousarray(np.sort(rows, axis=1).astype(np.int64))
    return rows.view([("", np.int64)] * rows.shape[1]).ravel()


class BoundaryMarkers(df.cpp.mesh.MeshFunctionSizet):
    """A facet `MeshFunction('size_t')` of `mesh`, every facet `value`."""

    def __init__(self, mesh, value=0):
        super().__init__(mesh, mesh.topology().dim() - 1, value)

    def mark(self, subdomain, value):
        """The facets `subdomain` (a SubDomain or CompiledSubDomain) holds set to `value`."""
        subdomain.mark(self, value)
        return self

    def write(self, path):
        """`path` and its .h5: the exterior facets with a nonzero marker. Collective;
        returns `(facets, points)` counts."""
        mesh = self.mesh()
        comm = _comm(mesh)
        facets, vertices, owned = _exterior(mesh)
        values = self.array()[facets]
        keep = owned & (values != 0)
        vertices, values = vertices[keep], values[keep].astype(np.int64)
        used, inverse = np.unique(vertices, return_inverse=True)
        glob = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)[used]
        x = mesh.coordinates()[used]
        if comm is None:
            order = np.argsort(glob)
            points, index = x[order], np.empty(len(glob), np.int64)
            index[order] = np.arange(len(glob))
        else:
            # each vertex once, at the rank whose block of global indices holds it
            edges = hdf5_rows.block_edges(mesh.num_entities_global(0), comm.Get_size())
            routed = hdf5_rows.route(comm, hdf5_rows.owner_of(edges, glob, comm), glob, x)
            got, got_x = routed.received
            mine, first, where = np.unique(got, return_index=True, return_inverse=True)
            points = got_x[first]
            start = comm.exscan(len(mine)) or 0
            (index,) = hdf5_rows.reply(comm, routed, start + where.astype(np.int64))
        return boundaries.write(path, points, index[inverse].reshape(vertices.shape),
                                values, comm)

    @classmethod
    def read(cls, path, mesh, block=BLOCK):
        """The markers of `path`'s facets on `mesh`'s exterior facets, matched by vertex
        coordinates; every other facet 0. Each rank streams the file and keeps what lies
        in its bounding box. Refuses a facet that matches no exterior facet. Collective."""
        import h5py
        from scipy.spatial import cKDTree
        self = cls(mesh)
        comm = _comm(mesh)
        D, x = mesh.topology().dim(), mesh.coordinates()
        facets, vertices, owned = _exterior(mesh)
        near, ids, kept, marks = np.inf, [], [], []
        if len(facets):
            pairs = [(0, 1)] if D == 2 else [(0, 1), (1, 2), (0, 2)]
            near = min(np.linalg.norm(x[vertices[:, a]] - x[vertices[:, b]], axis=1).min()
                       for a, b in pairs)
        tol = REL_TOL * near
        local = np.unique(vertices)
        where = boundaries.datasets(path)
        if where[2] is None:
            raise ValueError(path + ": no markers")
        with h5py.File(where[0][0], "r") as h5:
            n_points = h5[where[0][1].lstrip("/")].shape[0]
        with h5py.File(where[1][0], "r") as h5:
            n_facets, nv = h5[where[1][1].lstrip("/")].shape
        bad = nv != D
        if hdf5_rows.agreed(comm, bad):
            raise ValueError("{}: facets of {} points, the mesh's of {}".format(path, nv, D))
        if len(local):
            lo, hi = x[local].min(axis=0) - tol, x[local].max(axis=0) + tol
            tree = cKDTree(x[local])
            with h5py.File(where[0][0], "r") as h5:
                p = h5[where[0][1].lstrip("/")]
                for start in range(0, n_points, block):
                    b = np.asarray(p[start:start + block])
                    inside = np.flatnonzero(((b >= lo) & (b <= hi)).all(axis=1))
                    dist, k = tree.query(b[inside], distance_upper_bound=tol)
                    hit = np.isfinite(dist)
                    ids.append(start + inside[hit])
                    kept.append(local[k[hit]])
        ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        kept = np.concatenate(kept) if kept else np.zeros(0, np.int64)
        rows, marks = [], []
        if len(ids):
            with h5py.File(where[1][0], "r") as fh, h5py.File(where[2][0], "r") as mh:
                f, m = fh[where[1][1].lstrip("/")], mh[where[2][1].lstrip("/")]
                for start in range(0, n_facets, block):
                    b = np.asarray(f[start:start + block], dtype=np.int64)
                    pos = np.minimum(np.searchsorted(ids, b), len(ids) - 1)
                    full = (ids[pos] == b).all(axis=1)
                    rows.append(kept[pos[full]])
                    marks.append(np.asarray(m[start:start + block]).ravel()[full])
        rows = np.concatenate(rows) if rows else np.zeros((0, D), np.int64)
        marks = np.concatenate(marks).astype(np.int64) if marks else np.zeros(0, np.int64)
        mine = _row_keys(vertices)
        order = np.argsort(mine)
        theirs = _row_keys(rows)
        pos = np.minimum(np.searchsorted(mine[order], theirs), max(len(mine) - 1, 0))
        hit = (mine[order][pos] == theirs) if len(mine) else np.zeros(len(rows), bool)
        at = order[pos[hit]]
        bad = bool((marks < 0).any())
        if hdf5_rows.agreed(comm, bad):
            raise ValueError(path + ": a negative marker")
        values = self.array()
        values[facets[at]] = marks[hit]
        matched = int(owned[at].sum())
        if comm is not None:
            matched = comm.allreduce(matched)
        if matched != n_facets:
            raise ValueError("{}: {} of its {} facets match an exterior facet of the mesh"
                             .format(path, matched, n_facets))
        return self
