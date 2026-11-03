"""`import femtools`, its I/O modules, `hierarchical_transfer`, `dof_reorder`, `abort` and
`memory` need numpy and h5py alone: dolfin and mpi4py made unimportable, they import
without loading cppimport or the compiled modules, and a quadratic grid is written and
read back."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import sys
for name in ("dolfin", "mpi4py", "ufl", "ffc", "dijitso", "petsc4py"):
    sys.modules[name] = None
import numpy as np
import femtools, femtools.xdmf, femtools.compress, femtools.hdf5_rows, femtools.facets
import femtools.dolfin_layout, femtools.abort, femtools.memory
import femtools.hierarchical_transfer, femtools.dof_reorder
assert not any(m in sys.modules for m in ("cppimport", "femtools.probes", "femtools.gradient"))
from femtools.xdmf import write_quadratic, _scan, _read_grid_geometry, _read_grid_attribute
coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
mid = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
coords = np.vstack([coords] + [0.5 * (coords[a] + coords[b]) for a, b in mid])
conn = np.arange(10).reshape(1, 10)
write_quadratic(sys.argv[1], coords, conn, [coords[:, 0]], ["x"])
grid = _scan(sys.argv[1])[0]
x, c = _read_grid_geometry(sys.argv[1], grid)
assert grid["topology"] == "Tetrahedron_10" and (c == conn).all() and (x == coords).all()
assert (_read_grid_attribute(sys.argv[1], grid, "x").ravel() == coords[:, 0]).all()
for name in ("dolfin", "mpi4py"):
    assert sys.modules[name] is None, name
print("OK")
"""


def test_import_without_dolfin_or_mpi4py(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "q.xdmf")],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
