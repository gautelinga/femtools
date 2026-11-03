"""Run under `mpirun -n 2` by tests/test_periodic_partition.py; prints one RESULT line per case.

A unit cube, periodic on all three axes, split in two halves along y. One cell of rank
0's half touching the x=1 face at a single vertex -- no facet on the face -- is moved to
rank 1: by dolfin 2019's defect rank 1
never learns that vertex is a slave, so the pairing must break, and the check must say
so. `repair` must then leave nothing endangered, and the space built on it must pass.
"""
import os
import sys

import numpy as np
import h5py
import dolfin as df
from mpi4py import MPI

from femtools import periodic_partition as pp
from femtools.periodic import periodic_box

comm = MPI.COMM_WORLD
tmp = sys.argv[1]
L = np.array([1.0, 1.0, 1.0])
periodic = (True, True, True)
tol = 1e-8
P2 = None


def result(name, value):
    if comm.rank == 0:
        print('RESULT', name, value, flush=True)


def write(path, x, topo, owner):
    perm = np.argsort(owner, kind='stable')
    offsets = np.searchsorted(owner[perm], np.arange(comm.size)).astype(np.uint64)
    with h5py.File(path, 'w') as h:
        h['mesh/coordinates'] = x
        t = h.create_dataset('mesh/topology', data=topo[perm])
        t.attrs['celltype'] = np.bytes_('tetrahedron')
        t.attrs['partition'] = offsets
        h['mesh/cell_indices'] = perm.astype(np.int64)


def space_ok(path):
    mesh = df.Mesh()
    with df.HDF5File(comm, path, 'r') as h:
        h.read(mesh, 'mesh', True)
    W = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement('Lagrange', mesh.ufl_cell(), 2),
        df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)]),
        constrained_domain=periodic_box(L, tol, periodic))
    try:
        pp.check_pairing(W, L, periodic, comm)
        return True
    except RuntimeError:
        return False


serial = df.UnitCubeMesh(MPI.COMM_SELF, 6, 6, 6)
x = serial.coordinates().copy()
topo = serial.cells().astype(np.int64)
# two real halves (a rank left without cells makes dolfin's read hang), then one cell
# from rank 0's half that touches the x=1 face at a single vertex, deep in that half
owner = (x[topo].mean(1)[:, 1] >= 0.5).astype(np.int64)
on_x1 = np.abs(x[:, 0] - 1.0) < tol
deep = on_x1 & (np.abs(x[:, 1] - 1.0 / 6) < tol) & (x[:, 2] > tol) & (x[:, 2] < 1 - tol)
touch_only = np.nonzero((on_x1[topo].sum(1) == 1) & deep[topo].any(1) & (owner == 0))[0]
owner[touch_only[0]] = 1

bad, _ = pp.endangered(x, topo, owner, L, periodic, tol)
result('endangered_crafted', len(bad))
path = os.path.join(tmp, 'crafted.h5')
if comm.rank == 0:
    write(path, x, topo, owner)
comm.Barrier()
result('crafted_space_ok', space_ok(path))

fixed, moved = pp.repair(x, topo, owner, L, periodic, tol)
bad, _ = pp.endangered(x, topo, fixed, L, periodic, tol)
result('moved', moved)
result('endangered_repaired', len(bad))
path = os.path.join(tmp, 'repaired.h5')
if comm.rank == 0:
    write(path, x, topo, fixed)
comm.Barrier()
result('repaired_space_ok', space_ok(path))

# `repaired` on a distributed mesh that needs a move: the crafted partition, read as
# written, goes through the layer gather and the in-memory rebuild with the move
def read_as_written(path):
    mesh = df.Mesh()
    with df.HDF5File(comm, path, 'r') as h:
        h.read(mesh, 'mesh', True)
    return mesh


def space_on_ok(mesh):
    W = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement('Lagrange', mesh.ufl_cell(), 2),
        df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)]),
        constrained_domain=periodic_box(L, tol, periodic))
    try:
        pp.check_pairing(W, L, periodic, comm)
        return True
    except RuntimeError:
        return False


crafted = os.path.join(tmp, 'crafted.h5')
lines = []
# L=None: the box is the coordinates' extent, as the ADE side reads it
mesh = pp.repaired(read_as_written(crafted), None, periodic, comm, log=lines.append)
result('repaired_log', lines[0].split(':')[1].split()[0] if lines else 'none')
result('repaired_cells', mesh.num_entities_global(3))
result('repaired_cell_ids', int(comm.allreduce(
    int(np.sum(np.asarray(mesh.topology().global_indices(3), dtype=np.int64))))
    == len(topo) * (len(topo) - 1) // 2))
result('repaired_distributed_space_ok', space_on_ok(mesh))

# read_mesh end to end, from an XDMF file as the solver reads it
xdmf = os.path.join(tmp, 'cube.xdmf')
if comm.rank == 0:
    with df.XDMFFile(MPI.COMM_SELF, xdmf) as f:
        f.write(serial)
comm.Barrier()
mesh = pp.read_mesh(xdmf, L, periodic, comm, log=lambda *a: None)
W = df.FunctionSpace(mesh, df.MixedElement([
    df.VectorElement('Lagrange', mesh.ufl_cell(), 2),
    df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)]),
    constrained_domain=periodic_box(L, tol, periodic))
pp.check_pairing(W, L, periodic, comm)
result('read_mesh_cells', mesh.num_entities_global(3))
result('read_mesh_space_ok', True)

# a read-only mesh directory: nothing is written there any more, so the read works
ro_mesh = os.path.join(tmp, 'ro_mesh')
if comm.rank == 0:
    os.makedirs(ro_mesh, exist_ok=True)
    for name in ('cube.xdmf', 'cube.h5'):
        os.replace(os.path.join(tmp, name), os.path.join(ro_mesh, name))
    os.chmod(ro_mesh, 0o555)
comm.Barrier()
here = os.getcwd()
os.chdir(tmp)
mesh = pp.read_mesh(os.path.join(ro_mesh, 'cube.xdmf'), L, periodic, comm, log=lambda *a: None)
result('read_only_mesh_dir_cells', mesh.num_entities_global(3))

# a working directory no rank can write: the repair writes nothing (in memory), so a
# partition that needs a move is repaired there all the same
ro_cwd = os.path.join(tmp, 'ro_cwd')
if comm.rank == 0:
    os.makedirs(ro_cwd, exist_ok=True)
    os.chmod(ro_cwd, 0o555)
comm.Barrier()
crafted_mesh = read_as_written(crafted)
os.chdir(ro_cwd)
fixed = pp.repaired(crafted_mesh, L, periodic, comm, log=lambda *a: None)
os.chdir(here)
x, cells = fixed.coordinates(), fixed.cells()
gv = np.asarray(fixed.topology().global_indices(0))
gc = np.asarray(fixed.topology().global_indices(3))
mine = comm.gather((gc, gv[cells], x[cells]), root=0)
before = comm.gather((np.asarray(crafted_mesh.topology().global_indices(3)),
                     np.asarray(crafted_mesh.topology().global_indices(0))[crafted_mesh.cells()],
                     crafted_mesh.coordinates()[crafted_mesh.cells()]), root=0)
same = None
if comm.rank == 0:
    def table(parts):
        ids = np.concatenate([p[0] for p in parts])
        o = np.argsort(ids)
        return ids[o], np.concatenate([p[1] for p in parts])[o], np.concatenate([p[2] for p in parts])[o]
    a, b = table(mine), table(before)
    same = all(np.array_equal(u, v) for u, v in zip(a, b))
result('unwritable_cwd_repaired_in_memory', comm.bcast(same, root=0)
       and pp.repaired(fixed, L, periodic, comm, log=lambda *a: None) is fixed)
if comm.rank == 0:
    os.chmod(ro_mesh, 0o755)
    os.chmod(ro_cwd, 0o755)
