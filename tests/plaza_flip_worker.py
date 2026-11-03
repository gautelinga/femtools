"""Run by tests/test_plaza_periodic.py, serially or under `mpirun -n 3`; one RESULT line a pass.

    python3 tests/plaza_flip_worker.py <mesh.h5>

Three passes of periodic Plaza refinement on a sheared box whose x faces, once bisected,
carry triangles with two exactly equal longest edges; its vertices numbered at random.
After each pass: whether the x faces pair by vertex and by edge, and how many tied
triangles the pass had to decide.
"""
import os
import sys

import dolfin as df
import numpy as np

from femtools.plaza_periodic import refine_periodic
from femtools.periodic import faces_congruent

comm = df.MPI.comm_world
path = sys.argv[1]
if comm.rank == 0 and not os.path.exists(path):
    b = df.BoxMesh(df.MPI.comm_self, df.Point(0, 0, 0), df.Point(1, 1, 1), 4, 4, 4)
    x0 = b.coordinates().copy()
    x0[:, 1] += 0.5 * x0[:, 2]              # the right triangles on x = 0, 1 become ties
    perm = np.random.default_rng(7).permutation(len(x0))
    x = np.empty_like(x0)
    x[perm] = x0
    t = perm[b.cells()]
    m = df.Mesh(df.MPI.comm_self)
    ed = df.MeshEditor()
    ed.open(m, "tetrahedron", 3, 3)
    ed.init_vertices(len(x))
    ed.init_cells(len(t))
    for i, v in enumerate(x):
        ed.add_vertex(i, v)
    for i, c in enumerate(t):
        ed.add_cell(i, np.asarray(c, dtype=np.uintp))
    ed.close()
    with df.HDF5File(df.MPI.comm_self, path, "w") as h:
        h.write(m, "mesh")
comm.barrier()
mesh = df.Mesh(comm)
with df.HDF5File(comm, path, "r") as h:
    h.read(mesh, "mesh", False)
box = [1.0, 1.5, 1.0]


def tied_face_triangles(mesh):
    """Triangles on x = 0 whose two longest edges are exactly equal, over all ranks."""
    x = mesh.coordinates()
    n = 0
    for c in mesh.cells():
        for k in range(4):
            f = np.delete(c, k)
            if np.all(np.abs(x[f, 0]) < 1e-12):
                q = x[f]
                L = sorted(np.linalg.norm(q[i] - q[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
                n += L[1] == L[2]
    return comm.allreduce(int(n))       # a face held by two ranks counts twice: only > 0 matters


for p in range(3):
    ties = tied_face_triangles(mesh)
    marks = df.MeshFunction("bool", mesh, 3, False)
    gc = np.asarray(mesh.topology().global_indices(3))
    r = np.random.default_rng(100 + p).random(mesh.num_entities_global(3))
    marks.array()[:] = r[gc] < 0.2
    mesh, _, _ = refine_periodic(mesh, marks, box, [0], redistribute=False)
    ok = faces_congruent(mesh, box, [0], 1e-9, edges=True)[0]
    if comm.rank == 0:
        print("RESULT", p, mesh.num_entities_global(3), "ok" if ok else "FAIL", ties, flush=True)
