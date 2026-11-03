"""Search-free field transfer between hierarchically refined meshes.

    first_incident_cell(cells, n)                one incident cell per vertex, deterministic
    compose(ancestors, parent)                   a parent-cell chain, one pass longer
    p1_eval, p2_eval, p1_eval_cellwise, ...      a P1 or P2 field at points in known cells
    DistributedCellSource(mesh_h5, field_h5, field, degree, components, comm)
                                                 a stored Function, fetched cell by cell
    eval_through_chain, eval_p2_through_chain    a source at a refined mesh's P1 or P2 nodes
    transfer_p1_field(old, values, new, parent)  a P1 field onto its one-pass refinement
    as_p1_function, as_p2_function               node values as a Function, ghost-guarded

`femtools.plaza_periodic.refine_periodic` returns the parent-cell map, and composing those
maps across passes gives every cell's ancestor in the original mesh. Evaluation is then
local algebra -- barycentric coordinates in the known ancestor, dotted with the P2 (or P1)
basis -- with no search at all, where `df.LagrangeInterpolator` locates every point by a
bounding-box search.

Everything works through cells, never through vertex identity across meshes: dolfin
renumbers vertices during refinement. Each new vertex is evaluated in the ancestor of one
of its incident cells; for hierarchical points the answer is the same whichever is picked
(the field is continuous), and P1 evaluation of a P1 field is exact at bisection points.
A P2 connectivity's columns are the 4 vertices, then the 6 midsides in XDMF order (e01 e12
e02 e03 e13 e23), `P2_EDGE_PAIRS`.

A source is anything with `eval_at(points, global_cells)`. `DistributedCellSource` reads a
Function as `femtools.dolfin_layout` (or `dolfin.HDF5File`) wrote it, each rank fetching
only the rows its cells name; nothing holds the whole field or numbers its nodes.

numpy alone at import; dolfin in the functions that take a Mesh, h5py in
`DistributedCellSource`, whose `comm` is an mpi4py communicator or None.
"""
import numpy as np

# midside columns 4..9 of the 10-node connectivity, as (vertex, vertex) pairs, in the
# XDMF order (`femtools.xdmf.XDMF_ORDER` puts dolfin's cell dofs in it)
P2_EDGE_PAIRS = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


def compose(ancestors, parent):
    """`ancestors` maps mesh_k cells to mesh_0; `parent` maps mesh_{k+1} to mesh_k."""
    return ancestors[parent]


def first_incident_cell(cells, n_vertices):
    """One incident cell per vertex -- any one; the fields evaluated are continuous.

    Deterministic (the lowest-indexed incident cell) so reruns are bit-identical.
    """
    owner = np.full(n_vertices, -1, dtype=np.int64)
    flat_v = cells.ravel()
    flat_c = np.repeat(np.arange(len(cells), dtype=np.int64), cells.shape[1])
    # reversed assignment: the FIRST occurrence wins after the dust settles
    owner[flat_v[::-1]] = flat_c[::-1]
    if (owner < 0).any():
        raise ValueError("{} vertices have no incident cell; a non-ghosted dolfin "
                         "mesh should not produce that".format(int((owner < 0).sum())))
    return owner


def barycentric_from_vertices(v, points):
    """Barycentric coordinates of each point in the tetrahedron whose four vertex
    coordinates are given as `v`, shape (n, 4, 3). Batched 3x3 solves, no search."""
    v0 = v[:, 0]
    T = np.stack([v[:, k] - v0 for k in (1, 2, 3)], axis=2)
    lam123 = np.linalg.solve(T, (np.asarray(points) - v0)[:, :, None])[:, :, 0]
    return np.column_stack([1.0 - lam123.sum(axis=1), lam123])


# Points are evaluated in blocks of this many. Every routine below is a per-row gather
# followed by a per-row contraction, so a block is bit-identical to the whole array --
# what blocking bounds is the *transient*, and the transient is what runs out. Evaluating
# a whole mesh at once expands the source's per-cell rows to one row per point
# (`nodal[conn10[cell_rows]]` is (n, 10, 3) for the velocity, 240 bytes a point) and lays
# the barycentric solve, the basis and the einsum on top of that.
#
# Measured, P2 velocity at 3M points over a 200k-cell source, peak RSS of the whole
# process (the floor -- sources, points and result -- is 295 MB):
#
#   chunk     whole      1M      500k     250k     100k      50k      25k
#   time     3.98 s   3.60 s   3.65 s   2.92 s   2.64 s   2.77 s   2.68 s
#   peak     1398 M    663 M    468 M    373 M    305 M    296 M    296 M
#
# So blocking is smaller *and* faster -- the working set fits in cache -- and 100k is
# where the memory reaches its floor at the best time. Below that nothing more is bought.
# This matters here and not elsewhere because it is the one evaluation whose point count
# is the *refined* mesh's: `DistributedCellSource` goes to some trouble to keep the row
# fetch bounded by the source mesh, and expanding to per-point in one array immediately
# afterwards handed that back.
EVAL_CHUNK = 100_000


def _in_blocks(n, evaluate, chunk=None):
    """`evaluate(slice)` over `n` rows a block at a time, as one array.

    Short-circuits below one block, so everything small -- every test, every fixture --
    takes exactly the unblocked path it always did. `chunk` defaults to `EVAL_CHUNK`
    read *here* rather than bound as a default argument, which would freeze it at import
    and make the module constant untunable -- including by the measurement that set it.
    """
    chunk = EVAL_CHUNK if chunk is None else chunk
    if n <= chunk:
        return evaluate(slice(0, n))
    first = evaluate(slice(0, chunk))
    out = np.empty((n,) + first.shape[1:], dtype=first.dtype)
    out[:chunk] = first
    for start in range(chunk, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = evaluate(slice(start, stop))
    return out


def barycentric(coords, cells, cell_ids, points):
    """Barycentric coordinates of each point in its assigned tetrahedron.

    `cells` may have more than 4 columns (a P2 connectivity); the first four are the
    vertices. Returns (n, 4) in vertex order. Callers already know the containing
    (ancestor) cell, so there is no search here.
    """
    return barycentric_from_vertices(coords[cells[cell_ids, :4]], points)


def p1_eval(coords, cells, vertex_values, cell_ids, points):
    """A P1 field at points with known containing cells: sum lambda_i * value_i.

    Exact -- not approximate -- wherever the points are vertices or edge midpoints of
    a refinement of the source mesh, which is precisely the warm-start case.
    """
    cell_ids, points = np.asarray(cell_ids), np.asarray(points)
    vertex_values = np.asarray(vertex_values)

    def block(sl):
        lam = barycentric(coords, cells, cell_ids[sl], points[sl])
        vals = vertex_values[cells[cell_ids[sl], :4]]
        if vals.ndim == 2:                                # scalar field
            return (lam * vals).sum(axis=1)
        return np.einsum("nk,nkc->nc", lam, vals)
    return _in_blocks(len(points), block)


def p2_basis(lam):
    """The 10 tetrahedral P2 basis functions at barycentric lam, XDMF node order."""
    basis = np.empty((len(lam), 10))
    for i in range(4):
        basis[:, i] = lam[:, i] * (2.0 * lam[:, i] - 1.0)
    for k, (a, b) in enumerate(P2_EDGE_PAIRS):
        basis[:, 4 + k] = 4.0 * lam[:, a] * lam[:, b]
    return basis


def p2_eval(coords, conn10, nodal, cell_rows, points):
    """A P2 field at points with known containing cells (rows of `conn10`).

    Exact for any field the P2 space represents -- the stored velocity in
    particular, which is why this reproduces `LagrangeInterpolator`'s vertex values
    to machine precision while doing no search.
    """
    cell_rows, points = np.asarray(cell_rows), np.asarray(points)
    nodal = np.asarray(nodal)

    def block(sl):
        basis = p2_basis(barycentric(coords, conn10, cell_rows[sl], points[sl]))
        vals = nodal[conn10[cell_rows[sl]]]               # (n, 10) or (n, 10, c)
        if vals.ndim == 2:
            return (basis * vals).sum(axis=1)
        return np.einsum("nk,nkc->nc", basis, vals)
    return _in_blocks(len(points), block)


def p2_eval_cellwise(vertices, nodal, points):
    """A P2 field where each point's own cell is already in hand, row by row.

    `vertices` is (n, 4, 3) and `nodal` is (n, 10) or (n, 10, c), both in XDMF node
    order, one row per point. The difference from `p2_eval` is that nothing is indexed
    through a node numbering -- which is the whole point, because a node numbering is
    global and a rank that holds only part of the mesh cannot build one.
    """
    basis = p2_basis(barycentric_from_vertices(vertices, points))
    nodal = np.asarray(nodal)
    if nodal.ndim == 2:
        return (basis * nodal).sum(axis=1)
    return np.einsum("nk,nkc->nc", basis, nodal)


def p1_eval_cellwise(vertices, nodal, points):
    """A P1 field where each point's own cell is already in hand, row by row.

    `vertices` is (n, 4, 3) and `nodal` is (n, 4) or (n, 4, c). Exact wherever the
    points are vertices or edge midpoints of a refinement of the source mesh, which is
    the warm-start case, and free of any node numbering, which is what lets it run on a
    rank holding only part of the mesh.
    """
    lam = barycentric_from_vertices(vertices, points)
    nodal = np.asarray(nodal)
    if nodal.ndim == 2:
        return (lam * nodal).sum(axis=1)
    return np.einsum("nk,nkc->nc", lam, nodal)


class DistributedCellSource:
    """A Lagrange field fetched cell by cell, with no rank holding the whole of it.

    It never builds a node numbering. That construction is a `np.unique` over six edges
    a cell, inherently global, and it is also unnecessary: the stored dofmap is already
    per cell, so a cell's nodal values can be fetched by row and used directly. Per
    *distinct* cell asked for it takes the four vertices from `mesh/topology`, their
    positions from `mesh/coordinates`, the dofs from `<field>/cell_dofs` and their
    values from `<field>/vector_0`, all through `hdf5_rows.fetch_rows`. dolfin
    partitions by geometry, so a rank's cells are a spatially local subset even though
    their global ids are scattered, and it holds about its own share.

    Node order: for degree 2 `cell_dofs` is in dolfin's order and `XDMF_ORDER` permutes
    it to the one `p2_basis` expects, corners first. For degree 1 the four dofs are the
    four vertices already, in the order `mesh/topology` lists them.

    The row a global cell occupies is resolved through `hdf5_rows.rows_of_labels`, which
    inverts `mesh/cell_indices` and `<field>/cells` without anyone holding them: a
    Function written under MPI has its rows in the writer's order. The offsets must be
    uniform, and that is a **raise**: a ragged `x_cell_dofs` would reshape into the
    right shape and mix cells together.

    `vertex_values`: the name of a dataset under `<field>` that, where present, holds a
    P1 scalar's values in the mesh file's vertex order; a cell's four values are then
    fetched by its own topology row, with no dofmap to resolve.

    **`cache=True`** is for a caller that evaluates in many collective rounds: every
    `eval_at` otherwise re-reads each rank's block of every dataset and redoes both label
    inversions. Cached, each block -- and each inverted label block -- is read on first
    use and kept until `release()` (or the end of a `with` block), and the lookups are
    the same functions on the same arrays, so the values are the same bits. It is opt-in
    because it holds all six blocks at once, ~1/P of the files a rank, where uncached
    holds one at a time. Serially (no comm, one rank) it is ignored.
    """

    def __init__(self, mesh_path, field_path, field, degree, components, comm=None,
                 cache=False, vertex_values=None):
        import h5py
        from femtools.xdmf import XDMF_ORDER
        from femtools.hdf5_rows import constant_diff
        self.comm = comm
        self.cache = cache and comm is not None and comm.Get_size() > 1
        self._blocks = {}
        self.field, self.degree, self.components = field, degree, components
        self.nodes = 10 if degree == 2 else 4
        self.order = np.asarray(XDMF_ORDER if degree == 2 else [0, 1, 2, 3])
        self.mesh_path, self.field_path = mesh_path, field_path
        self.values_dataset = vertex_values
        with h5py.File(mesh_path, "r") as h5:
            self.n_cells = int(h5["mesh/topology"].shape[0])
            n_vertices = int(h5["mesh/coordinates"].shape[0])
        with h5py.File(field_path, "r") as h5:
            self.vertex_values = vertex_values is not None and vertex_values in h5[field]
            if self.vertex_values:
                n_values = int(h5[field][vertex_values].shape[0])
        if self.vertex_values:
            if (degree, components) != (1, 1) or n_values != n_vertices:
                raise ValueError(
                    "{}:{} holds {} vertex values, a P1 scalar; asked for degree {} with "
                    "{} components on {} vertices".format(
                        field_path, field, n_values, degree, components, n_vertices))
            return
        with h5py.File(field_path, "r") as h5:
            offsets = h5[field + "/x_cell_dofs"]
            self.stride = int(offsets[1] - offsets[0])
            n_dofs = int(h5[field + "/cell_dofs"].shape[0])
        want = components * self.nodes
        if self.stride != want or n_dofs != self.stride * self.n_cells:
            raise ValueError(
                "{} is stored at {} dofs per cell over {} cells, not the {} x {} a "
                "degree-{} field with {} components has".format(
                    field, self.stride, self.n_cells, components, self.nodes,
                    degree, components))
        if not constant_diff(field_path, field + "/x_cell_dofs", self.stride, comm):
            raise ValueError(
                "{}:{}/x_cell_dofs is ragged, so a fixed stride would mix cells "
                "together".format(field_path, field))

    def release(self):
        """Drop the cached blocks; the next `eval_at` reads them again."""
        self._blocks = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def _rows(self, path, dataset, wanted, labels=False, group=1):
        """`fetch_rows` (or `rows_of_labels`), out of the cached block when caching.
        Cached, `group` > 1 takes a flat dataset as rows of `group` (`read_block`)."""
        from femtools import hdf5_rows
        if not self.cache:
            fetch = hdf5_rows.rows_of_labels if labels else hdf5_rows.fetch_rows
            return fetch(path, dataset, wanted, self.comm)
        # as `fetch_rows` and `rows_of_labels` do: a uint64 `wanted` (a topology row)
        # would make `lookup`'s offsets float64 under numpy 2
        wanted = np.asarray(wanted, dtype=np.int64)
        key = (path, dataset, labels, group)
        if key not in self._blocks:
            if labels:
                self._blocks[key] = hdf5_rows.invert_labels(path, dataset, self.comm)
            else:
                self._blocks[key] = hdf5_rows.read_block(path, dataset, self.comm, group)
        edges, block = self._blocks[key]
        hdf5_rows.check_range(path, dataset, wanted, int(edges[-1]), self.comm)
        return hdf5_rows.lookup(self.comm, edges, block, wanted)

    def eval_at(self, points, global_cells):
        """The field at points whose containing cells are known by global id."""
        from femtools.hdf5_rows import check_range, unique_inverse
        cells = np.asarray(global_cells, dtype=np.int64)
        check_range(self.mesh_path, "mesh/topology", cells, self.n_cells, self.comm)
        # one fetch per *distinct* cell, so the traffic is bounded by the source mesh
        # and not by the mesh being evaluated on, however many points share a cell
        uniq, inverse = unique_inverse(cells)
        # the two files order their rows differently, and neither order need be the
        # global one, so each global cell is resolved against its own file's labelling
        mesh_rows = self._rows(self.mesh_path, "mesh/cell_indices", uniq, labels=True)
        topo = self._rows(self.mesh_path, "mesh/topology", mesh_rows)
        xyz = self._rows(self.mesh_path, "mesh/coordinates",
                         topo.ravel()).reshape(len(uniq), 4, 3)
        if self.vertex_values:
            vals = self._rows(self.field_path, self.field + "/" + self.values_dataset,
                              topo.ravel()).reshape(len(uniq), 1, 4)
        else:
            field_rows = self._rows(self.field_path, self.field + "/cells", uniq,
                                    labels=True)
            if self.cache:
                # a cell's row of `stride` dofs at once: the same entries, with one
                # request a cell rather than one a dof to deduplicate and send
                dofs = self._rows(self.field_path, self.field + "/cell_dofs", field_rows,
                                  group=self.stride)
            else:
                rows = (field_rows[:, None] * self.stride
                        + np.arange(self.stride)).ravel()
                dofs = self._rows(self.field_path, self.field + "/cell_dofs", rows)
            vals = self._rows(self.field_path, self.field + "/vector_0",
                              dofs.ravel()).reshape(len(uniq), self.components, self.nodes)
        nodal = vals[:, :, self.order]
        if self.components > 1:
            nodal = np.transpose(nodal, (0, 2, 1))
        else:
            nodal = nodal[:, 0, :]
        evaluate = p2_eval_cellwise if self.degree == 2 else p1_eval_cellwise
        # blocked, because `xyz[inverse]` and `nodal[inverse]` are the expansion the
        # `np.unique` above exists to avoid -- one row per *point* rather than per
        # distinct cell -- and doing it whole gives back the memory the row fetch saved
        points = np.asarray(points)
        return _in_blocks(len(points),
                          lambda sl: evaluate(xyz[inverse[sl]], nodal[inverse[sl]],
                                              points[sl]))


# --- the thin dolfin-facing layer -------------------------------------------------

def local_to_global_cells(mesh):
    """Each local cell's global id; identity where the mesh has no global numbering."""
    topology = mesh.topology()
    if topology.have_global_indices(topology.dim()):
        return np.asarray(topology.global_indices(topology.dim()), dtype=np.int64)
    return np.arange(mesh.num_cells(), dtype=np.int64)


def as_p1_function(mesh, values, width=3, name="u"):
    """Per-vertex values as a P1 Function on `mesh`; vector where `width` > 1.

    The scatter is ghost-guarded (`femtools.xdmf.scatter_vertex_values`), and getting
    that wrong is silent under MPI.
    """
    import dolfin as df
    from femtools.xdmf import scatter_vertex_values
    space = (df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=width) if width > 1
             else df.FunctionSpace(mesh, "Lagrange", 1))
    return scatter_vertex_values(df.Function(space, name=name), values, width=width)


def _lowered(points, cell_of_point, z_offset):
    """Points moved down by their cell's z offset, to be evaluated in the ancestor a
    period or more below."""
    if z_offset is None:
        return points
    out = np.array(points, dtype=float, copy=True)
    out[:, 2] -= np.asarray(z_offset)[cell_of_point]
    return out


def eval_through_chain(source, mesh, global_cells, z_offset=None):
    """A source's field at every local vertex of `mesh`, via known containing cells.

    `global_cells` is one global cell id per *local cell* of `mesh` -- the chain read
    for this rank's own rows -- and each vertex is evaluated in the cell of the first
    incident one. The fields carried this way are continuous, so which incident cell
    is picked does not matter; that it is picked deterministically does, for reruns.
    `z_offset`, one per local cell, is subtracted from z first (None, the default: no
    shift).
    """
    fc = first_incident_cell(mesh.cells(), len(mesh.coordinates()))
    return source.eval_at(_lowered(mesh.coordinates(), fc, z_offset),
                          np.asarray(global_cells)[fc])


def velocity_on_mesh(source, mesh, ancestors, gcell0):
    """u evaluated at every local vertex of `mesh`, via ancestor cells. No search.

    `ancestors` maps this mesh's local cells to mesh-0 *local* cells (the composed
    parent chain -- rank-local throughout, since refinement without redistribution
    keeps children on their parent's rank); `gcell0` maps mesh-0 local cells to the
    global ids the velocity file is keyed by.
    """
    return eval_through_chain(source, mesh, gcell0[np.asarray(ancestors)])


# --- carrying the velocity at P2 -------------------------------------------------

def p2_entity_nodes(mesh):
    """The P2 nodes of `mesh` by entity dimension: `[(points, cell), (points, cell)]`.

    A P2 node sits on a vertex or on an edge, so the node set is exactly the mesh's
    vertices followed by its edge midpoints, each in that entity's own local ordering --
    which is the ordering `dofmap().entity_dofs(mesh, d)` speaks, so the two line up
    without anything being matched by coordinate. `cell` is one local cell incident to
    the entity; which one does not matter, the fields carried this way being continuous,
    but that it is chosen deterministically does, for reruns.

    Nothing here goes through `tabulate_dof_coordinates`: under MPI that returns the
    *owned* dofs only, while `entity_dofs` covers this rank's ghost entities too, so the
    two are not the same list and pairing them by position would be wrong on every rank
    but one. `tests/test_hierarchical_transfer.py` checks at 1 and 3 ranks that dof zero
    of each entity's block lands on the point computed here.
    """
    from femtools.periodic import connectivity
    x = mesh.coordinates()
    cells = mesh.cells()
    edge_v = connectivity(mesh, 1, 0)
    cell_e = connectivity(mesh, 3, 1)
    return [(x, first_incident_cell(cells, len(x))),
            (0.5 * (x[edge_v[:, 0]] + x[edge_v[:, 1]]),
             first_incident_cell(cell_e, len(edge_v)))]


def eval_p2_through_chain(source, mesh, global_cells, z_offset=None):
    """A source's field at every P2 node of `mesh`, as one array per entity dimension.

    The P2 twin of `eval_through_chain`, and the reason the velocity can be carried onto
    a refined mesh *at P2* rather than projected to P1 on the way: the stored field is
    evaluated exactly in each node's ancestor cell, midside nodes included, which is
    what the P1 step throws away. `global_cells` is one global cell id per local cell,
    exactly as `eval_through_chain` takes it, and `z_offset` as there.
    """
    g = np.asarray(global_cells)
    parts = p2_entity_nodes(mesh)
    # one evaluation for both dimensions: a distributed source reads the flow file's
    # rows once per call, so two calls read it twice for nearly the same cells
    points = np.concatenate([_lowered(p, c, z_offset) for p, c in parts])
    cells = np.concatenate([g[c] for _, c in parts])
    values = source.eval_at(points, cells)
    ends = np.cumsum([len(c) for _, c in parts])[:-1]
    return np.split(values, ends)


def as_p2_function(mesh, values_by_dim, width=3, name="u"):
    """Per-entity P2 node values as a P2 Function on `mesh`; vector where `width` > 1.

    The P2 counterpart of `as_p1_function`, and it needs the same guard for the same
    reason `femtools.xdmf.scatter_vertex_values` does: `entity_dofs` covers this rank's
    **ghost** entities, whose dof indices run past the end of `get_local()`, which is
    over the ownership range alone. Writing through them would raise, or with a negative
    index put a value on some other node's dof.

    A blocked dofmap keeps a node's `width` components consecutive -- checked here rather
    than assumed, since everything below indexes on that -- so the ownership of the whole
    block is settled by its first component, as the P1 scatter settles it.
    """
    import dolfin as df
    space = (df.VectorFunctionSpace(mesh, "Lagrange", 2, dim=width) if width > 1
             else df.FunctionSpace(mesh, "Lagrange", 2))
    f = df.Function(space, name=name)
    values = f.vector().get_local()
    for dim, vals in enumerate(values_by_dim):
        block = np.asarray(space.dofmap().entity_dofs(mesh, dim)).reshape(-1, width)
        if width > 1 and len(block) and not np.array_equal(
                block[:, 1:], block[:, :1] + np.arange(1, width)):
            raise ValueError(
                "the P2 dofmap is not blocked by node, so a node's components are not "
                "consecutive and the ownership test on the first one does not cover "
                "them. This whole transfer indexes on that.")
        owned = (block[:, 0] >= 0) & (block[:, 0] < len(values))
        vals = np.asarray(vals).reshape(len(block), width)
        values[block[owned]] = vals[owned]
    f.vector().set_local(values)
    f.vector().apply("insert")
    return f


def velocity_on_mesh_p2(source, mesh, ancestors, gcell0):
    """u at every P2 node of `mesh`, via ancestor cells. The P2 twin of
    `velocity_on_mesh`, taking the same rank-local chain and relabelling it the same way.
    """
    return eval_p2_through_chain(source, mesh, gcell0[np.asarray(ancestors)])


def transfer_p1_field(mesh_old, vertex_values_old, mesh_new, parent):
    """The previous pass's P1 field at every vertex of the refined mesh. Exact.

    `parent` is `plaza_periodic.refine_periodic`'s map; evaluation happens in the
    one-level parent cell, so this needs no ancestor chain and no vertex identity.
    """
    cells_new = mesh_new.cells()
    x_new = mesh_new.coordinates()
    fc = first_incident_cell(cells_new, len(x_new))
    return p1_eval(mesh_old.coordinates(), mesh_old.cells(), vertex_values_old,
                   parent[fc], x_new)
