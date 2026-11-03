# femtools

Tools for solvers written in legacy dolfin (FEniCS 2019.2), gathered from the solvers that
needed them (stokes-beadpack, addictif, twoasis) so that each is written once. The parts that
do not touch dolfin also serve readers without it, such as partrac's tools.

## Features

### Files: XDMF and HDF5

- **`xdmf.XDMFFile2`**: a drop-in for `dolfin.XDMFFile` that writes dolfin's own layout but
  keeps the midside nodes of a P2 field (`Tetrahedron_10`, `Triangle_6`); dolfin's
  interpolates them away silently. It also writes several fields on one grid and compresses
  every dataset.
- **`xdmf.grids`, `xdmf.parse_xdmf`**: where every step's HDF5 arrays are, in one reader;
  `parse_xdmf` keeps the interface of partractools' function of that name.
- **`xdmfread`**: reads datasets with their gzip chunks inflated on a thread pool, whole or
  a chunk-row at a time. h5py inflates on one thread, which takes hours at billions of cells.
- **`compress`, `hdf5_rows`**: compressed datasets, and rows of them read, written and
  exchanged between MPI ranks without any rank holding a whole one. Under MPI every rank
  writes its own block, through MPI-IO (rank zero writes them all only with a serial h5py).
- **`dolfin_layout`**: a Mesh or Function written in the layout `dolfin.HDF5File` gives it,
  compressed, which dolfin reads on any number of ranks; `write_dofs` writes a space's dof
  tables once, for Functions that link them (a checkpoint of vectors alone);
  `write_mesh_arrays` without dolfin.
- **`boundaries`**: a boundary facet file (facets, an integer marker each, the points they
  use), written serially or from blocks on many ranks, and read back. It is the layout
  partrac's `wall_facets` reads. `BoundaryMarkers` is a facet MeshFunction that writes its
  marked exterior facets as that file and reads them back onto a mesh, on any number of
  ranks.
- **`facets`, `femtools-cut-facets`**: a facet file cut to the facets with selected markers,
  such as a full dolfin facet MeshFunction to its walls.

### Periodic meshes and refinement

- **`periodic`**: `PeriodicBox` and the compiled `periodic_box`, the `constrained_domain` of
  a 2D or 3D periodic box with any lower corner; `faces_congruent` checks identified faces.
- **`periodic_partition`**: reads a mesh in blocks and partitions it with ParMETIS so that
  dolfin's parallel periodic dofmap pairs every face entity, which dolfin's own partition
  does not guarantee.
- **`plaza_periodic`, `refine_periodic`**: dolfin's Plaza refinement with identified faces
  bisected alike, serial or under MPI, with the parent-cell map. Plain refinement breaks the
  periodic pairing.
- **`hierarchical_transfer`**: P1 and P2 fields moved to a refined mesh through ancestor
  cells, with no point search.

### Point evaluation and gradients

- **`probes`**: `Probes` and `GradProbes`, a Function's values and gradients at fixed
  points, located once and evaluated many times. From Mikael Mortensen's fenicstools, as
  extended in addictif.
- **`gradient`**: the weighted gradient matrix, a P1 field's derivative at the nodes as one
  matrix-vector product. From Mikael Mortensen's fenicstools, unchanged in what it computes.

### Running at scale

- **`dof_reorder`, `femtools-build-dof-reorder`**: dolfin's dof reordering with a sparser
  graph, built as a library to preload over libdolfin's, without the memory peak of
  dolfin's own at large sizes.
- **`abort`**: `run(main)`, so an exception on one MPI rank ends the job instead of
  hanging it.
- **`memory`**: hands freed heap back to the operating system, and reports what a process
  holds.

## Requirements

`import femtools` needs numpy and h5py only. dolfin, mpi4py and scipy are imported where
they are used. `probes` and `gradient` compile their C++ with cppimport (pybind11, a
compiler and `libdolfin-dev`) on first import; `periodic`, `periodic_partition` and
`plaza_periodic` use dolfin's JIT on first use. `periodic_partition` needs a ParMETIS with
64-bit `idx_t` past 2^31 cells.

## Install

Into the interpreter that has dolfin, without dependencies (a pip mpi4py would be built
against another MPI than dolfin's):

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

In ParaView, raise "Nonlinear Subdivision Level" to see the quadratic field.

```python
from femtools.periodic import periodic_box
V = df.FunctionSpace(mesh, "CG", 2, constrained_domain=periodic_box(
    upper=(Lx, Ly), periodic=(True, False), lower=(0, 0)))

from femtools.probes import Probes
probes = Probes(points.flatten(), V)
probes(u)
values = probes.array()
```

```python
from femtools.boundaries import BoundaryMarkers
markers = BoundaryMarkers(mesh)                     # where a facet MeshFunction was
markers.mark(df.CompiledSubDomain("on_boundary && near(x[2], 0)"), 5)
bc = df.DirichletBC(V, g, markers, 5)
markers.write("boundaries.xdmf")                    # the exterior facets marked nonzero
markers = BoundaryMarkers.read("boundaries.xdmf", mesh)
```

```bash
femtools-cut-facets boundaries.xdmf walls.xdmf --markers 3,5,7:106
femtools-cut-facets full.xdmf boundaries.xdmf --markers 1:    # every marker from 1
femtools-build-dof-reorder --out libdof_reorder_star.so
LD_PRELOAD=$PWD/libdof_reorder_star.so python3 solve.py
```

## Tests

```bash
python3 -m pytest              # the quick tier, seconds
python3 -m pytest -m slow      # the mpirun and compile tests, minutes
python3 -m pytest -m ""        # everything
```

The dolfin tests skip where dolfin is missing.

## Credits and licence

- **Probes and the weighted gradient matrix:** Mikael Mortensen's
  [fenicstools](https://github.com/mikaem/fenicstools), Copyright (C) 2011–2013 Mikael
  Mortensen. The probes come through addictif's extended copy (adding `GradProbes`).
- **Periodic Plaza refinement and the dof reordering:** adapted from
  [DOLFIN](https://bitbucket.org/fenics-project/dolfin) 2019.2, Copyright (C) 2008–2015
  Anders Logg, Ola Skavhaug, Garth N. Wells and others.

Both are LGPL-3.0-or-later, and their files keep their headers; NOTICE lists what came from
where and what changed. femtools is LGPL-3.0-or-later (COPYING.LESSER, COPYING).
