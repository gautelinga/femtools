"""Run under `mpirun -n 2` by tests/test_periodic_partition.py; prints one RESULT line per case.

The triangle-mesh twin of `periodic_partition_worker.py`: a unit square, periodic on both
axes, split in two halves along y, and one triangle of rank 0's half touching the x=1
edge at a single vertex moved to rank 1, which then never learns that vertex is a slave.
The check must say so, the repair must move the triangle, and `repaired` and `read_mesh`
must give correctly paired spaces. Then the same square centred at the origin: the check
with its lower corner given.
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
tol = 1e-8
periodic = (True, True, False)


def result(name, value):
    if comm.rank == 0:
        print('RESULT', name, value, flush=True)


def write(path, x, topo, owner):
    perm = np.argsort(owner, kind='stable')
    offsets = np.searchsorted(owner[perm], np.arange(comm.size)).astype(np.uint64)
    with h5py.File(path, 'w') as h:
        h['mesh/coordinates'] = x
        t = h.create_dataset('mesh/topology', data=topo[perm])
        t.attrs['celltype'] = np.bytes_('triangle')
        t.attrs['partition'] = offsets
        h['mesh/cell_indices'] = perm.astype(np.int64)


def read_as_written(path):
    mesh = df.Mesh()
    with df.HDF5File(comm, path, 'r') as h:
        h.read(mesh, 'mesh', True)
    return mesh


def space_ok(mesh, lo, hi, lower=None):
    W = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement('Lagrange', mesh.ufl_cell(), 2),
        df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)]),
        constrained_domain=periodic_box(hi + [0.], tol, periodic, lo + [0.]))
    try:
        pp.check_pairing(W, hi, periodic, comm, lower=lower)
        return True
    except pp.PairingError:
        return False


def crafted(lo, hi, name):
    """The square from lo to hi, its crafted partition written to `name`."""
    serial = df.RectangleMesh(MPI.COMM_SELF, df.Point(*lo), df.Point(*hi), 12, 12, 'crossed')
    x = serial.coordinates().copy()
    topo = serial.cells().astype(np.int64)
    mid = (lo[1] + hi[1]) / 2
    owner = (x[topo].mean(1)[:, 1] >= mid).astype(np.int64)
    on_x1 = np.abs(x[:, 0] - hi[0]) < tol
    deep = on_x1 & (x[:, 1] > lo[1] + 0.1) & (x[:, 1] < mid - 0.1)
    touch_only = np.nonzero((on_x1[topo].sum(1) == 1) & deep[topo].any(1) & (owner == 0))[0]
    owner[touch_only[0]] = 1
    path = os.path.join(tmp, name)
    if comm.rank == 0:
        write(path, x, topo, owner)
    comm.Barrier()
    return serial, x, topo, owner, path


lo, hi = [0., 0.], [1., 1.]
serial, x, topo, owner, path = crafted(lo, hi, 'crafted.h5')
bad, _ = pp.endangered(x, topo, owner, hi, periodic, tol)
result('endangered_crafted', len(bad))
result('crafted_space_ok', space_ok(read_as_written(path), lo, hi))

fixed, moved = pp.repair(x, topo, owner, hi, periodic, tol)
bad, _ = pp.endangered(x, topo, fixed, hi, periodic, tol)
result('moved', moved)
result('endangered_repaired', len(bad))

lines = []
mesh = pp.repaired(read_as_written(path), None, periodic, comm, log=lines.append)
result('repaired_log', lines[0].split(':')[1].split()[0] if lines else 'none')
result('repaired_cells', mesh.num_entities_global(2))
result('repaired_cell_ids', int(comm.allreduce(
    int(np.sum(np.asarray(mesh.topology().global_indices(2), dtype=np.int64))))
    == len(topo) * (len(topo) - 1) // 2))
result('repaired_space_ok', space_ok(mesh, lo, hi))

# read_mesh end to end, from the dolfin HDF5 file
whole = os.path.join(tmp, 'square.h5')
if comm.rank == 0:
    with df.HDF5File(MPI.COMM_SELF, whole, 'w') as f:
        f.write(serial, 'mesh')
comm.Barrier()
mesh = pp.read_mesh(whole, hi, periodic, comm, log=lambda *a: None)
result('read_mesh_cells', mesh.num_entities_global(2))
result('read_mesh_space_ok', space_ok(mesh, lo, hi))

# centred at the origin: the repair takes the upper corner alone (None: the largest
# coordinates); the check needs the lower one too
lo, hi = [-0.5, -0.5], [0.5, 0.5]
_, _, _, _, path = crafted(lo, hi, 'centred.h5')
result('centred_crafted_space_ok', space_ok(read_as_written(path), lo, hi, lower=lo))
mesh = pp.repaired(read_as_written(path), None, periodic, comm, log=lambda *a: None)
result('centred_repaired_space_ok', space_ok(mesh, lo, hi, lower=lo))
