"""A dolfin Mesh or Function written in the layout `dolfin.HDF5File.write` gives it,
compressed (`compress`; dolfin's own writer has no compression).

    write_mesh(path, mesh)                 # <name>/coordinates, topology, cell_indices
    write_function(path, u, "u")           # <name>/vector_0, cell_dofs, x_cell_dofs, cells
    write_dofs(path, V, "V")               # <name>/cell_dofs, x_cell_dofs, cells, once
    write_function(path, u, "u", dofs=("tables.h5", "V"))   # vector_0, the tables linked
    write_mesh_arrays(path, x, cells)      # the mesh from arrays, serially, without dolfin

The same datasets, dtypes, values and attributes as dolfin writes on the same partition
(`tests/test_dolfin_layout.py` holds the two to every bit on 1 and 3 ranks), so every
reader -- dolfin's `HDF5File.read`, partrac, h5py -- reads either. What dolfin writes, and
so this:

- `coordinates` by global vertex index; `topology` each rank's own cells in local order,
  rank after rank, as global vertex indices, with `celltype` and `partition` (each rank's
  first row); `cell_indices` those cells' global ids.
- `x_cell_dofs` (uint64) each own cell's first entry in `cell_dofs` (the dofmap's
  integer type) plus the total; `cells` (uint64) the cells' global ids; `vector_0` the
  vector's local ranges with `partition` (each rank's first row); the group's `signature`,
  the element's.

Every dataset goes through `hdf5_rows.write_ordered`: no rank holds one whole, and every
rank writes its own block (MPI-IO). dolfin and mpi4py are imported where they are used.

The dof tables are most of a Function's file, and the same for every Function on a space
until the mesh is partitioned anew; `write_dofs` writes them once, and
`write_function(..., dofs=...)` links them (an HDF5 external link a reader follows) next
to each vector -- a checkpoint then costs its vectors alone.
"""
import numpy as np

from femtools.hdf5_rows import write_ordered


def _offsets(comm, n):
    """(this rank's first row, every rank's first row) for n rows a rank, rank order."""
    from mpi4py import MPI
    first = comm.exscan(int(n), op=MPI.SUM) if comm.Get_size() > 1 else 0
    first = 0 if first is None else int(first)
    return first, np.asarray(comm.allgather(first), dtype=np.uint64)


def write_mesh(path, mesh, name="mesh", mode="w"):
    """The mesh as `HDF5File(path, mode).write(mesh, name)` writes it. Collective."""
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    own = mesh.topology().ghost_offset(tdim)
    cells = np.asarray(mesh.cells())[:own]
    gv = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.int64)[:own]
    first, partition = _offsets(comm, own)
    rows = first + np.arange(own, dtype=np.int64)
    n_c, n_v = mesh.num_entities_global(tdim), mesh.num_entities_global(0)
    write_ordered(path, name + "/coordinates", gv, np.asarray(mesh.coordinates()), n_v, comm,
                  mode=mode)
    write_ordered(path, name + "/topology", rows, gv[cells], n_c, comm,
                  attrs={"celltype": np.bytes_(mesh.ufl_cell().cellname()),
                         "partition": partition})
    write_ordered(path, name + "/cell_indices", rows, gc, n_c, comm)


def _dof_tables(V):
    """`(gc, dofs)`: this rank's own cells' global ids (uint64), and their global dofs, a
    row a cell, in the dofmap's integer type."""
    from femtools.periodic import cell_dofs
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    own = mesh.topology().ghost_offset(tdim)
    dm = V.dofmap()
    to_global = np.asarray(dm.tabulate_local_to_global_dofs(), dtype=np.int64)
    # every cell has the element's dofs (Lagrange and mixed spaces of them): one compiled
    # call for all, not a pybind call a cell
    n = V.element().space_dimension()
    per_cell = cell_dofs(dm, np.arange(own), n) if own else np.zeros((0, n), np.int64)
    # the dofmap's integer type, from a rank that has a cell
    dtype = next(t for t in comm.allgather(dm.cell_dofs(0).dtype.str if own else None) if t)
    gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.uint64)[:own]
    return gc, to_global[per_cell].astype(dtype)


def _mix(x):
    """splitmix64's finalizer, elementwise on uint64 (wrapping)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _fingerprint(comm, gc, dofs):
    """A 64-bit hash of which global dofs each cell (by global id) has, in which local
    position: a sum over the cells, so the same on any partition of the same numbering,
    and another for another numbering (another number of ranks, or partition)."""
    n = dofs.shape[1] if dofs.ndim == 2 else 0
    key = gc[:, None] * np.uint64(n + 1) + np.arange(n, dtype=np.uint64)
    mine = int(_mix(_mix(key) ^ dofs.astype(np.uint64)).sum(dtype=np.uint64))
    return np.uint64(sum(comm.allgather(mine)) % (1 << 64))


#: each space's fingerprint, by the space's id (dolfin never reuses one in a process): a
#: space's numbering is fixed, and the fingerprint costs a pass over every cell's dofs
_FINGERPRINTS = {}


def _space_fingerprint(V):
    key = V.id()
    if key not in _FINGERPRINTS:
        _FINGERPRINTS[key] = _fingerprint(V.mesh().mpi_comm(), *_dof_tables(V))
    return _FINGERPRINTS[key]


def _write_tables(path, name, comm, n_c, gc, dofs, mode):
    """`x_cell_dofs`, `cell_dofs` and `cells` of the group `name`, as dolfin writes them."""
    own, n = dofs.shape if dofs.ndim == 2 else (0, 0)
    local = dofs.reshape(-1)
    lengths = np.full(own, n, dtype=np.uint64)
    first, _ = _offsets(comm, own)
    dof_first, _ = _offsets(comm, len(local))
    n_dofs = comm.allreduce(len(local))
    x_rows = first + np.arange(own, dtype=np.int64)
    # in uint64 throughout: an int64 [0] beside a uint64 sum promotes to float64
    x_vals = np.uint64(dof_first) + np.concatenate(
        [np.zeros(1, np.uint64), np.cumsum(lengths, dtype=np.uint64)[:-1]])[:own]
    if comm.Get_rank() == comm.Get_size() - 1:      # the total, by the last rank
        x_rows = np.r_[x_rows, n_c]
        x_vals = np.r_[x_vals, np.uint64(n_dofs)]
    write_ordered(path, name + "/x_cell_dofs", x_rows, x_vals, n_c + 1, comm, mode=mode)
    write_ordered(path, name + "/cell_dofs", dof_first + np.arange(len(local)), local,
                  n_dofs, comm)
    write_ordered(path, name + "/cells", x_rows[:own], gc, n_c, comm)


def write_dofs(path, V, name="dofs", mode="w"):
    """The dof tables of V alone -- `x_cell_dofs`, `cell_dofs` and `cells`, as
    `write_function` writes them -- with the element's `signature` and the numbering's
    `fingerprint` on the group. Written once for a space and a partition, they serve every
    `write_function(..., dofs=(path, name))` of a Function on it. Collective."""
    from femtools.hdf5_rows import opened
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    gc, dofs = _dof_tables(V)
    _write_tables(path, name, comm, mesh.num_entities_global(mesh.topology().dim()), gc,
                  dofs, mode)
    fingerprint = _fingerprint(comm, gc, dofs)
    _FINGERPRINTS[V.id()] = fingerprint
    with opened(path, "a", comm) as h5:
        if h5 is not None:
            h5[name].attrs["signature"] = np.bytes_(V.element().signature())
            h5[name].attrs["fingerprint"] = fingerprint


def _linked_tables(path, dofs, V, comm):
    """Refuse, on every rank, tables (`write_dofs`) that are not V's on this partition."""
    import os
    import h5py
    from femtools.hdf5_rows import agreed
    target, group = dofs
    where = os.path.join(os.path.dirname(os.path.abspath(path)), target)
    found = None
    if comm.Get_rank() == 0:
        try:
            with h5py.File(where, "r") as h5:
                a = h5[group].attrs
                found = (bytes(a["signature"]), np.uint64(a["fingerprint"]))
        except (OSError, KeyError) as e:
            found = str(e)
    found = comm.bcast(found, root=0)
    fingerprint = _space_fingerprint(V)
    if os.path.realpath(where) == os.path.realpath(path):
        bad = ("{}: the dof tables must be in another file than the Function's, which "
               "mode 'w' would truncate".format(where))
    elif isinstance(found, str):
        bad = "{}:{}: no dof tables ({})".format(where, group, found)
    elif found[0] != np.bytes_(V.element().signature()):
        bad = "{}:{}: the dof tables of another element".format(where, group)
    elif found[1] != fingerprint:
        bad = ("{}:{}: the dof tables of another numbering (another number of ranks, or "
               "partition); write this one's with write_dofs".format(where, group))
    else:
        bad = None
    if agreed(comm, bad is not None):
        raise ValueError(bad or "another rank refused the dof tables")


def write_function(path, u, name, mode="a", dofs=None):
    """The Function as `HDF5File(path, mode).write(u, name)` writes it. Collective.

    With `dofs = (file, group)`, the dof tables `write_dofs` wrote for u's space on this
    partition (`file` relative to `path`'s directory, as HDF5 resolves it) are linked
    into the group (external links) rather than written again: dolfin, h5py and every
    reader follow the links, so the group reads as before while only `vector_0` is
    written. The tables are checked to be the space's, numbered as now."""
    import h5py
    from femtools.hdf5_rows import opened
    V = u.function_space()
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    vec = u.vector()
    lo, hi = vec.local_range()
    _, partition = _offsets(comm, hi - lo)
    if dofs is None:
        gc, tables = _dof_tables(V)
        _write_tables(path, name, comm, mesh.num_entities_global(mesh.topology().dim()), gc,
                      tables, mode)
        mode = "a"
    else:
        _linked_tables(path, dofs, V, comm)
    write_ordered(path, name + "/vector_0", np.arange(lo, hi, dtype=np.int64),
                  vec.get_local(), vec.size(), comm, mode=mode, attrs={"partition": partition})
    with opened(path, "a", comm) as h5:
        if h5 is not None:
            if dofs is not None:
                for k in ("x_cell_dofs", "cell_dofs", "cells"):
                    h5[name + "/" + k] = h5py.ExternalLink(dofs[0], dofs[1] + "/" + k)
            h5[name].attrs["signature"] = np.bytes_(V.element().signature())


def write_mesh_arrays(path, x, cells, name="mesh", mode="w"):
    """A serial mesh from its arrays, as dolfin writes one on one rank: `x` the vertex
    coordinates, `cells` (triangles or tetrahedra) indices into them."""
    import h5py
    from femtools.compress import put
    x = np.ascontiguousarray(x, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    with h5py.File(path, mode) as h5:
        put(h5, name + "/coordinates", x)
        t = put(h5, name + "/topology", cells)
        t.attrs["celltype"] = np.bytes_("triangle" if cells.shape[1] == 3 else "tetrahedron")
        t.attrs["partition"] = np.zeros(1, dtype=np.uint64)
        put(h5, name + "/cell_indices", np.arange(len(cells), dtype=np.int64))
