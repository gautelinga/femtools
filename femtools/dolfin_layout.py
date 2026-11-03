"""A dolfin Mesh or Function written in the layout `dolfin.HDF5File.write` gives it,
compressed (`compress`; dolfin's own writer has no compression).

    write_mesh(path, mesh)                 # <name>/coordinates, topology, cell_indices
    write_function(path, u, "u")           # <name>/vector_0, cell_dofs, x_cell_dofs, cells
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

Every dataset goes through `hdf5_rows.write_ordered`: no rank holds one whole. dolfin and
mpi4py are imported where they are used.
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


def write_function(path, u, name, mode="a"):
    """The Function as `HDF5File(path, mode).write(u, name)` writes it. Collective."""
    import h5py
    V = u.function_space()
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    own = mesh.topology().ghost_offset(tdim)
    dm = V.dofmap()
    from femtools.periodic import cell_dofs
    to_global = np.asarray(dm.tabulate_local_to_global_dofs(), dtype=np.int64)
    # every cell has the element's dofs (Lagrange and mixed spaces of them): one compiled
    # call for all, not a pybind call a cell
    n = V.element().space_dimension()
    per_cell = cell_dofs(dm, np.arange(own), n) if own else np.zeros((0, n), np.int64)
    # the dofmap's integer type, from a rank that has a cell
    dtype = next(t for t in comm.allgather(dm.cell_dofs(0).dtype.str if own else None) if t)
    local = to_global[per_cell.reshape(-1)].astype(dtype)
    per_cell = to_global = None
    lengths = np.full(own, n, dtype=np.uint64)
    first, _ = _offsets(comm, own)
    dof_first, _ = _offsets(comm, len(local))
    n_c = mesh.num_entities_global(tdim)
    n_dofs = comm.allreduce(len(local))
    x_rows = first + np.arange(own, dtype=np.int64)
    # in uint64 throughout: an int64 [0] beside a uint64 sum promotes to float64
    x_vals = np.uint64(dof_first) + np.concatenate(
        [np.zeros(1, np.uint64), np.cumsum(lengths, dtype=np.uint64)[:-1]])[:own]
    if comm.Get_rank() == comm.Get_size() - 1:      # the total, by the last rank
        x_rows = np.r_[x_rows, n_c]
        x_vals = np.r_[x_vals, np.uint64(n_dofs)]
    gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.uint64)[:own]
    vec = u.vector()
    lo, hi = vec.local_range()
    _, partition = _offsets(comm, hi - lo)
    write_ordered(path, name + "/x_cell_dofs", x_rows, x_vals, n_c + 1, comm, mode=mode)
    write_ordered(path, name + "/cell_dofs", dof_first + np.arange(len(local)), local,
                  n_dofs, comm)
    write_ordered(path, name + "/cells", x_rows[:own], gc, n_c, comm)
    write_ordered(path, name + "/vector_0", np.arange(lo, hi, dtype=np.int64),
                  vec.get_local(), vec.size(), comm, attrs={"partition": partition})
    if comm.Get_rank() == 0:
        with h5py.File(path, "a") as h5:
            h5[name].attrs["signature"] = np.bytes_(V.element().signature())
    comm.Barrier()


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
