# femtools

Tools for solvers written in legacy dolfin (FEniCS 2019.2):

- `femtools.xdmf.XDMFFile2`: a drop-in for `dolfin.XDMFFile` that keeps the midside
  nodes of a quadratic field (`Tetrahedron_10`), writes several fields onto one grid,
  lets a MeshFunction share a file with Functions, and compresses every dataset.
- `femtools.compress`, `femtools.hdf5_rows`: compressed HDF5 datasets, and rows of them
  read, written and exchanged between MPI ranks without any rank holding a whole one.
- `femtools.facets` and `femtools-cut-facets`: a facet XDMF cut down to the facets
  with selected markers.
- `femtools.probes`: `Probes` and `GradProbes`, values and gradients of a Function at
  points, from fenicstools.
- `femtools.gradient`: the weighted gradient matrix, from fenicstools.
- `femtools.periodic`: `PeriodicBox` and `periodic_box` (compiled), the periodic box a
  `constrained_domain` is built with, on 2D and 3D meshes; `faces_congruent`, whether
  identified faces carry the same nodes.
- `femtools.dolfin_layout`: a Mesh or Function written in the layout
  `dolfin.HDF5File.write` gives it, compressed; `write_mesh_arrays` without dolfin.
- `femtools.periodic_partition`: `read_mesh`, a mesh read in blocks and partitioned by
  ParMETIS so that dolfin's parallel periodic dofmap pairs every face entity;
  `check_pairing`, `repartition`.
- `femtools.abort`: `run(main)`, a command's entry point under MPI: an uncaught exception
  on one rank aborts the job instead of hanging it.
- `femtools.memory`: `release` and `drop`, freed heap handed back to the operating system.

`import femtools` needs numpy and h5py only. dolfin, mpi4py and scipy are imported
where they are used; `femtools.probes` and `femtools.gradient` compile their C++ with
cppimport (and pybind11, a compiler and `libdolfin-dev`) the first time they are
imported, `femtools.periodic` and `femtools.periodic_partition` with dolfin's JIT the
first time they are used. `femtools.periodic_partition` needs a ParMETIS with a 64-bit
`idx_t` for meshes past 2^31 cells; it opens `$PETSC_DIR/$PETSC_ARCH/lib/libparmetis.so`
first, then the system's.

## Install

Into the interpreter that has dolfin, without dependencies (a pip mpi4py would be
built against another MPI than dolfin's):

```bash
python3 -m pip install --user --break-system-packages --no-deps --no-build-isolation -e .
```

## Use

```python
import dolfin as df
from femtools.xdmf import XDMFFile2

mesh = df.UnitCubeMesh(8, 8, 8)
u = df.Function(df.VectorFunctionSpace(mesh, "CG", 2), name="u")
p = df.Function(df.FunctionSpace(mesh, "CG", 1), name="p")

with XDMFFile2(mesh.mpi_comm(), "up.xdmf") as f:   # where dolfin.XDMFFile was
    f.write(u, 0.0)
    f.write((u, p), 1.0)                            # two fields on one grid
```

Open `up.xdmf` in ParaView and raise "Nonlinear Subdivision Level" to see the quadratic
field.

```bash
femtools-cut-facets boundaries.xdmf walls.xdmf --markers 3,5,7:106
```

```python
from femtools.periodic import periodic_box
V = df.FunctionSpace(mesh, "CG", 2, constrained_domain=periodic_box(
    upper=(Lx, Ly), periodic=(True, False), lower=(0, 0)))
```

```python
from femtools.probes import Probes
probes = Probes(points.flatten(), V)
probes(u)
values = probes.array()
```

## Tests

```bash
python3 -m pytest tests
```

The dolfin tests skip where dolfin is missing; `-m "not slow"` leaves out the ones
that spawn `mpirun`.

## Licence

LGPL-3.0-or-later (COPYING.LESSER, COPYING). The probes and the gradient matrix come
from fenicstools: see NOTICE.
