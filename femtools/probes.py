__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
A Function's values, and gradients, at many points, evaluated many times.

    probes = Probes(points.flatten(), V)      # every rank given the same points
    probes(u)                                 # one snapshot
    probes.array()                            # (points, components, snapshots) on root
    probes.array(N=0)                         # one snapshot: (points, components)

A point is kept by the lowest rank that finds it in its cells, so a point on a partition
boundary is evaluated once; a point outside the mesh is found by no rank and reads 0 in
`array`. `GradProbes` takes gradients too (`grad`, `array_grad`). The C++ classes are
built with cppimport on first import (femtools/cpp/probe11.cpp).
"""
import os

import cppimport
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

probe11 = cppimport.imp_from_filepath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "probe11.cpp"),
    "femtools.cpp.probe11")


def _on_root(ids, local, total, root):
    """Every rank's rows `local` (one a local probe, `ids` their places) as one array on
    root, `total` rows, zeros where no rank holds the probe; None elsewhere."""
    local = np.ascontiguousarray(local, dtype=float)
    tail = local.shape[1:]
    width = int(np.prod(tail)) if tail else 1
    counts = comm.gather(len(ids), root=root)
    all_ids = comm.gather(np.asarray(ids, dtype=np.int64), root=root)
    is_root = comm.Get_rank() == root
    buf = np.empty(sum(counts) * width) if is_root else None
    comm.Gatherv(local.reshape(-1), [buf, [c * width for c in counts]] if is_root else None,
                 root=root)
    if not is_root:
        return None
    z = np.zeros((total,) + tail)
    z[np.concatenate(all_ids)] = buf.reshape((sum(counts),) + tail)
    return z


class _Collection(object):
    """What Probes and GradProbes share in Python: call to evaluate, iterate over the
    local probes as (id, probe), and gather to root."""

    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.local_size()

    def __iter__(self):
        for i in range(self.local_size()):
            yield self[i]

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def _array(self, values, N, filename, component, root, what):
        """`values` (local probes, components, snapshots) on root, one snapshot N or
        all, one component or all, saved to `filename` + `_snapshot_N` or `_all` too.
        Every rank checks N and the component alike, before the collective gather."""
        n_comp, n_snap = values.shape[1], values.shape[2]
        if N is not None and not 0 <= N < n_snap:
            raise IndexError("{}: no snapshot {}, there are {}".format(what, N, n_snap))
        if component is not None and not 0 <= component < n_comp:
            raise IndexError("{}: no component {}, there are {}".format(
                what, component, n_comp))
        if component is not None:
            values = values[:, [component], :]
        if N is not None:
            values = values[:, :, N]
        z = _on_root(self.get_probe_ids(), values, self.get_total_number_probes(), root)
        if z is None:
            return None
        if filename:
            np.save(filename + ("_snapshot_" + str(N) if N is not None else "_all"), z)
        return np.squeeze(z)

    def array(self, N=None, filename=None, component=None, root=0):
        """Every probe's values on root: all snapshots or snapshot N, every component or
        one. Collective; None on the other ranks."""
        return self._array(self.values(), N, filename, component, root, "array")


class Probe(probe11.Probe):

    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()

    def __getitem__(self, i):
        return self.get_probe_at_snapshot(i)


class Probes(_Collection, probe11.Probes):
    pass


class GradProbe(probe11.GradProbe):

    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()

    def __getitem__(self, i):
        return self.get_probe_at_snapshot(i)


class GradProbes(_Collection, probe11.GradProbes):

    def grad(self, *args):
        return self.eval_grad(*args)

    def array_grad(self, N=None, filename=None, component=None, root=0):
        """`array` for the gradients: component j*gdim + k is d(value j)/d(x_k)."""
        return self._array(self.grad_values(), N, filename, component, root, "array_grad")
