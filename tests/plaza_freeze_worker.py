#!/usr/bin/env python3
"""Refine a periodic cube with the edges of its z = 0 face frozen, on however many ranks
mpirun gives (`tests/test_plaza_periodic.py`).

    mpirun -n 3 python3 tests/plaza_freeze_worker.py

The cube's interior vertices are moved by a smooth function of position, the same on every
rank, and its first layer is flattened, so the longest-edge closure of the face reaches
cells that hold no edge of it. Every cell with
its centroid below z = 0.45 is marked. Prints, on rank 0, `RESULT <mode> <cells> <z = 0
facets> <marked> <unmarked> <frozen cells> <marked cells left whole>`, mode `free` (no
freeze) and `frozen`.
"""
import numpy as np

import dolfin as df

from femtools.refine_periodic import refine_congruent


def z0_facets(mesh):
    """The facets with all three vertices on z = 0, each once (through its one cell)."""
    mesh.init(3, 2)
    mesh.init(2, 1)
    z = mesh.coordinates()[:, 2]
    out = []
    for c in df.cells(mesh):
        for f in df.facets(c):
            if (np.abs(z[f.entities(0)]) < 1e-12).all():
                out.append(f)
    return out


def main():
    comm = df.MPI.comm_world
    mesh = df.UnitCubeMesh(comm, 8, 8, 8)
    x = mesh.coordinates()
    # the first layer flattened: the face's edges are then the longest of the faces
    # leaning on them, and the closure has to propagate into the layer
    x[np.abs(x[:, 2] - 0.125) < 1e-12, 2] = 0.05
    inner = ((x > 1e-12) & (x < 1 - 1e-12)).all(axis=1)
    x[inner] += 0.03 * np.stack([np.sin(7.1 * x[inner, 1] + 3.3 * x[inner, 2]),
                                 np.sin(5.3 * x[inner, 0] + 2.9 * x[inner, 2]),
                                 np.sin(6.7 * x[inner, 0] + 4.1 * x[inner, 1])], axis=1)
    mesh.bounding_box_tree().build(mesh)
    marked = np.array([c.midpoint().z() < 0.45 for c in df.cells(mesh)], bool)
    markers = df.MeshFunction("bool", mesh, 3, False)
    markers.array()[:] = marked
    n_marked = int(df.MPI.sum(comm, float(marked.sum())))
    edges = np.unique([e for f in z0_facets(mesh) for e in f.entities(1)]).astype(np.int64)
    for mode in ("free", "frozen"):
        report = {}
        new, _, parent = refine_congruent(mesh, markers, [1.0, 1.0, 1.0], [0, 1],
                                          frozen=edges if mode == "frozen" else None,
                                          report=report)
        children = np.bincount(np.asarray(parent), minlength=mesh.num_cells())
        whole = int(df.MPI.sum(comm, float((marked & (children == 1)).sum())))
        nz = int(df.MPI.sum(comm, float(len(z0_facets(new)))))
        if comm.rank == 0:
            print("RESULT {} {} {} {} {} {} {}".format(
                mode, new.num_entities_global(3), nz, n_marked, report.get("unmarked", 0),
                report.get("frozen_cells", 0), whole), flush=True)


if __name__ == "__main__":
    main()
