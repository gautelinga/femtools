"""`XDMFFile2`: a drop-in for legacy dolfin's `XDMFFile` that keeps a quadratic field.

    with XDMFFile2(comm, 'u.xdmf') as f:
        f.write((u, p))

dolfin's XDMFFile interpolates a quadratic function down to one value per vertex,
silently, so the picture loses everything the midside nodes carry. XDMF can hold
quadratic tetrahedra (`Tetrahedron_10`), which ParaView renders once "Nonlinear
Subdivision Level" is turned up; this writes those.

## What `XDMFFile2` covers

It is a drop-in for `dolfin.XDMFFile`, not just for the one case it was written for,
and it writes the *same file layout* legacy dolfin writes -- so dolfin reads back
what it produces, and a reader that copes with one copes with the other:

    write(Function)                write(Function, t)
    write(Mesh)                    write(MeshFunction)      write(MeshValueCollection)
    write(points)                  write(points, values)
    read(Mesh)                     read(MeshFunction)       read(MeshValueCollection)
    write_checkpoint / read_checkpoint
    close, rename, name, label, id, parameters, Encoding

It differs from `dolfin.XDMFFile` in exactly four ways, all of them deliberate:

- **A degree-2 Function keeps its midside nodes.** dolfin throws them away without
  saying so; this writes `Tetrahedron_10`.
- **`write((u, p))` writes several functions onto one node set.** dolfin takes one
  function per call; several sharing a grid are then certainly registered against
  each other, and the mesh is stored once instead of once per field.
- **A MeshFunction and a Function can share a file.** `dolfin.XDMFFile` *segfaults*
  on that combination -- `write(mf)` followed by `write(u)` takes the process down,
  in that order only -- which is measured in `tests/test_xdmf.py`.
- **A MeshFunction can share a *grid*, as cell data.** `write((u, p, tags))` writes
  the MeshFunction as a `Center="Cell"` attribute on the one grid the fields are on,
  which is what a value per cell is; dolfin has no call for this and rejects a tuple
  outright. It has to be over the same mesh at its top dimension, and is refused
  otherwise -- a facet MeshFunction against the cells writes numbers belonging to
  other entities onto a grid that looks perfectly well formed.

The three `parameters` are dolfin's, and they mean what dolfin means by them:
`flush_output` (write as you go rather than at close), `functions_share_mesh`
(successive functions attach to the grid already written rather than starting a new
one) and `rewrite_function_mesh` (store the mesh again for each time step, or point
every step at the one copy). Like dolfin's, the mapping refuses a key it does not
know rather than silently ignoring it.

`read_checkpoint` reads dolfin's own checkpoints -- which already round-trip a P2
function exactly, so those are handed straight to dolfin -- *and* the quadratic
visualisation grids written here, which dolfin cannot read at all. That reconstruction
goes via an unconstrained space and then interpolates onto whatever space the caller's
Function lives in, so it is safe for a periodic space, where matching dof coordinates
is not.

Every dataset is shuffled and gzipped (`femtools.compress`). In parallel it writes
with MPI-IO when h5py has parallel HDF5; without it the gather path is refused above
`GATHER_LIMIT` cells (`FEMTOOLS_GATHER_LIMIT`). `write_quadratic` and
`write_quadratic_blocks` write the same layout from plain arrays, the second from
ranks' shares.

dolfin, mpi4py and scipy are imported only inside the functions that need them:
`XdmfDocument`, `write_quadratic` and the readers need numpy and h5py alone.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from femtools import compress

# XDMF's Tetrahedron_10: four corners, then the midsides of these edges in order.
XDMF_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
# dolfin's own P2 node order: four corners, then the midsides of these.
DOLFIN_EDGES = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))
# XDMF slot -> dolfin local node, corners first
XDMF_ORDER = [0, 1, 2, 3] + [4 + DOLFIN_EDGES.index(e) for e in XDMF_EDGES]

XI = 'http://www.w3.org/2001/XInclude'
DOCTYPE = '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'

# (cell name, nodes per cell) -> XDMF TopologyType. Only the simplices dolfin meshes
# here are made of, plus the quadratic forms XDMF can hold.
XDMF_TOPOLOGY = {
    ('point', 1): 'Polyvertex',
    ('interval', 2): 'Polyline',
    ('triangle', 3): 'Triangle',
    ('tetrahedron', 4): 'Tetrahedron',
    ('interval', 3): 'Edge_3',
    ('triangle', 6): 'Triangle_6',
    ('tetrahedron', 10): 'Tetrahedron_10',
}
# and back, for reading
TOPOLOGY_CELL = {v: k for k, v in XDMF_TOPOLOGY.items()}
# meshio's names, which `write_quadratic` still takes for its callers' sake
MESHIO_TOPOLOGY = {'tetra': 'Tetrahedron', 'tetra10': 'Tetrahedron_10',
                   'triangle': 'Triangle', 'triangle6': 'Triangle_6',
                   'line': 'Polyline', 'vertex': 'Polyvertex'}


class Encoding(object):
    """`XDMFFile.Encoding` -- the arrays go to HDF5 beside the XML.

    dolfin also has an ASCII encoding, inlining the arrays into the XML. Support
    for it was removed: nothing here wrote it, and it is refused loudly rather
    than quietly written as HDF5.
    """
    HDF5 = 'HDF5'

    @staticmethod
    def of(value):
        """dolfin's enum, this one, or the string; anything else is an error."""
        if value is None:
            return Encoding.HDF5
        name = str(getattr(value, 'name', value)).upper()
        if name == 'ASCII':
            raise ValueError('ASCII encoding is no longer supported; '
                             'the arrays go to HDF5')
        if name != 'HDF5':
            raise ValueError('{!r} is not an XDMF encoding'.format(value))
        return name


class XDMFParameters(object):
    """dolfin's three parameters, and its habit of refusing a key it does not know.

    A plain dict would accept `flush_ouput` and quietly do nothing with it, which is
    the class of silence this whole module exists to remove.
    """

    _DEFAULTS = (('flush_output', False),
                 ('functions_share_mesh', False),
                 ('rewrite_function_mesh', True))

    def __init__(self):
        self._values = dict(self._DEFAULTS)

    def __getitem__(self, key):
        self._check(key)
        return self._values[key]

    def __setitem__(self, key, value):
        self._check(key)
        self._values[key] = bool(value)

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def values(self):
        return self._values.values()

    def update(self, other):
        for key, value in dict(other).items():
            self[key] = value

    def _check(self, key):
        if key not in self._values:
            raise KeyError(
                '{!r} is not an XDMF parameter; dolfin has {}'.format(
                    key, ', '.join(sorted(self._values))))

    def str(self, verbose=False):
        return '\n'.join('{:<22} | {}'.format(k, v)
                         for k, v in sorted(self._values.items()))

    def __repr__(self):
        return 'XDMFParameters({})'.format(self._values)


#: cells `XDMFFile2` gathers to rank zero at most, under MPI with a serial h5py (the
#: parallel h5py writes collectively and is not limited); FEMTOOLS_GATHER_LIMIT
GATHER_LIMIT = int(os.environ.get('FEMTOOLS_GATHER_LIMIT', 20_000_000))

class XdmfDocument(object):
    """The XDMF XML and its HDF5 sidecar, in the layout legacy dolfin writes.

    Deliberately free of dolfin: the standalone route has no dolfin at all, and the
    two producers must not drift into writing different files for the same data.
    Everything dolfin-shaped -- gathering a distributed mesh, tabulating a dofmap --
    happens in `XDMFFile2` above this and arrives here as plain arrays.
    """

    def __init__(self, path, comm=None):
        self.path = path
        stem = os.path.splitext(os.path.basename(path))[0]
        self.h5_name = stem + '.h5'
        self.h5_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                    self.h5_name)
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        # Whether the HDF5 sidecar is opened by every rank at once and written in
        # slabs, or by rank zero after everything has been gathered to it. Recorded
        # rather than inferred, so a test can insist the collective path really ran
        # instead of passing on a silent fall back to the gather.
        self.driver = None
        self._parallel = None       # decided once; it cannot change under us
        self._h5 = None
        self.root = ET.Element('Xdmf', {'Version': '3.0', 'xmlns:xi': XI})
        self.domain = ET.SubElement(self.root, 'Domain')
        self.series = None          # the temporal collection, once a time is given
        self._counts = {}
        self._groups = set()        # HDF5 groups already spoken for

    # -- HDF5 ------------------------------------------------------------------

    def _next(self, kind):
        n = self._counts.get(kind, 0)
        self._counts[kind] = n + 1
        return n

    def free_group(self, base):
        """An HDF5 group name not already spoken for.

        Every other writer here numbers its groups -- `Mesh/0`, `MeshFunction/1`,
        `VisualisationVector/2` -- and the points path did not, so a second
        `write(points)` deleted the first one's arrays (`dataset` recreates a group
        that exists) while the first grid's XML went on pointing at them.
        """
        name = base
        while name in self._groups:
            name = '{}/{}'.format(base, self._next(base))
        self._groups.add(name)
        return name

    def mesh_group(self, timed):
        """A free HDF5 group for one grid's mesh, named the way dolfin names it.

        dolfin calls the first untimed one `Mesh/mesh` and numbers the timed ones.
        The numbering matters here in a way it does not there: this file can hold a
        linear mesh grid and a quadratic function grid at once, and if the second
        took the first's group name it would replace the linear topology with the
        quadratic one and leave the mesh grid pointing at cells of the wrong size.
        """
        name = 'Mesh/mesh' if not timed else None
        while name is None or name in self._groups:
            name = 'Mesh/{}/mesh'.format(self._next('Mesh'))
        self._groups.add(name)
        return name

    def parallel(self):
        """Whether this document can write collectively.

        Needs more than one rank *and* an h5py with parallel HDF5 in it. Debian ships
        the MPI build as a separate package and its `h5py/__init__.py` picks it
        automatically when `OMPI_COMM_WORLD_SIZE` is set, so this is a question about
        the environment rather than about anything the caller did; with the serial
        build the gather path is used and the answer is simply False.
        """
        if self._parallel is None:
            if self.comm is None or self.size == 1:
                self._parallel = False
            else:
                import h5py
                self._parallel = bool(h5py.get_config().mpi)
        return self._parallel

    def _h5file(self):
        if self._h5 is None:
            import h5py
            if self.parallel():
                # every rank opens; the driver is what makes the writes below slabs
                # of one file rather than one file each
                self._h5 = h5py.File(self.h5_path, 'w', driver='mpio', comm=self.comm)
                self.driver = 'mpio'
            else:
                self._h5 = h5py.File(self.h5_path, 'w')
                self.driver = 'serial'
        return self._h5

    def dataset_distributed(self, group, rows, values, n_global, dtype=None):
        """One global dataset, each rank writing the rows it owns.

        `rows` are this rank's *global* row numbers and `values` the data for them.
        They are redistributed to a block layout first, so that every rank ends up
        writing a single contiguous hyperslab -- scattered selections are what
        collective HDF5 is worst at, and dolfin's global entity numbering is
        thoroughly scattered (rank zero held vertices spanning [162,727] of 729).

        Returns the reference a DataItem points at, as `dataset` does.
        """
        values = np.ascontiguousarray(values)
        shape = (int(n_global),) + values.shape[1:]
        lo, hi, slab = _redistribute(self.comm, rows, values, n_global)
        h5 = self._h5file()
        if group in h5:
            del h5[group]
        node = h5.create_dataset(group, shape=shape, dtype=dtype or values.dtype,
                                 **compress.options(shape))
        node.attrs['partition'] = np.array([0], dtype=np.uint64)
        _collective_write(node, lo, slab)
        return '{}:/{}'.format(self.h5_name, group)

    def dataset(self, group, array):
        """Stores one array and returns the reference a DataItem points at.

        The `partition` attribute is dolfin's, and is written so that dolfin's own
        reader finds a file of ours exactly as it finds one of its own.
        """
        array = np.ascontiguousarray(array)
        h5 = self._h5file()
        if group in h5:
            del h5[group]
        node = compress.put(h5, group, array)
        node.attrs['partition'] = np.array([0], dtype=np.uint64)
        return '{}:/{}'.format(self.h5_name, group)

    def dataset_slab(self, group, offset, values, n_global, dtype=None):
        """One global dataset written as a contiguous slab per rank, no exchange.

        For rows whose order in the file is arbitrary -- cells, and the entities a
        MeshFunction is defined over -- each rank simply takes the next block, so
        nothing has to move. `dolfin.XDMFFile` reorders its cells between rank counts
        for the same reason, so this matches it rather than departing from it.
        """
        values = np.ascontiguousarray(values)
        shape = (int(n_global),) + values.shape[1:]
        h5 = self._h5file()
        if group in h5:
            del h5[group]
        node = h5.create_dataset(group, shape=shape, dtype=dtype or values.dtype,
                                 **compress.options(shape))
        node.attrs['partition'] = np.array([0], dtype=np.uint64)
        _collective_write(node, int(offset), values)
        return '{}:/{}'.format(self.h5_name, group)

    # -- XML -------------------------------------------------------------------

    def _data_item(self, parent, ref, shape, unsigned=False):
        item = ET.SubElement(parent, 'DataItem')
        item.set('Dimensions', ' '.join(str(int(n)) for n in shape))
        if unsigned:
            item.set('NumberType', 'UInt')
        item.set('Format', 'HDF')
        item.text = ref
        return item

    def grid(self, name, parent=None):
        return ET.SubElement(self.domain if parent is None else parent,
                             'Grid', {'Name': name, 'GridType': 'Uniform'})

    def topology(self, grid, cells, topology_type, ref=None, shape=None):
        # `shape` is the *global* one: writing in parallel, a rank holds a slice, and
        # the XML has to describe the whole dataset rather than that rank's part
        n_cells, per_cell = shape if shape is not None else cells.shape
        node = ET.SubElement(grid, 'Topology', {
            'NumberOfElements': str(n_cells), 'TopologyType': topology_type,
            'NodesPerElement': str(per_cell)})
        self._data_item(node, ref if ref is not None else cells,
                        (n_cells, per_cell), unsigned=True)
        return node

    def geometry(self, grid, coords, ref=None, shape=None):
        node = ET.SubElement(grid, 'Geometry', {'GeometryType': 'XYZ'})
        self._data_item(node, ref if ref is not None else coords,
                        shape if shape is not None else coords.shape)
        return node

    def attribute(self, grid, name, values, center='Node', ref=None, shape=None):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        shape = shape if shape is not None else values.shape
        kind = {1: 'Scalar', 3: 'Vector', 9: 'Tensor'}.get(shape[1])
        if kind is None:
            raise ValueError('{} components is not a Scalar, Vector or Tensor'
                             .format(shape[1]))
        node = ET.SubElement(grid, 'Attribute', {
            'Name': name, 'AttributeType': kind, 'Center': center})
        self._data_item(node, ref if ref is not None else values, shape)
        return node

    def time(self, grid, value):
        """A `<Time>`, and the temporal collection it has to live in."""
        ET.SubElement(grid, 'Time', {'Value': '{:.15e}'.format(float(value))})

    def collection(self):
        """The temporal collection dolfin puts every timed grid inside."""
        if self.series is None:
            self.series = ET.SubElement(self.domain, 'Grid', {
                'Name': 'TimeSeries', 'GridType': 'Collection',
                'CollectionType': 'Temporal'})
        # A grid written before the first timed one stays where it is. It used to be
        # moved in here, which put a grid carrying no <Time> inside a temporal
        # collection -- ambiguous to a reader, and it misrepresents a static mesh as a
        # time step. dolfin does something worse with the same sequence: measured,
        # `write(mesh)` then `write(u, t)` **drops the mesh grid altogether**.
        return self.series

    # -- output ----------------------------------------------------------------

    def save(self):
        # One XML, written by one rank. Every rank builds the tree -- that is what
        # keeps the dataset names in step -- but they must not all write the file.
        if self.rank != 0:
            return
        ET.indent(self.root, space='  ')
        body = ET.tostring(self.root, encoding='unicode')
        with open(self.path, 'w') as handle:
            handle.write('<?xml version="1.0"?>\n{}\n{}\n'.format(DOCTYPE, body))

    def flush(self):
        # the HDF5 flush is collective under mpio, so every rank makes it
        if self._h5 is not None:
            self._h5.flush()
        self.save()

    def close(self):
        self.save()
        if self._h5 is not None:
            self._h5.close()          # collective under mpio
            self._h5 = None


def _redistribute(comm, rows, values, n_global):
    """Send each row to the rank that owns its slice of the file.

    Returns `(lo, hi, slab)`: this rank's block of the global dataset, filled. The
    caller then writes one contiguous hyperslab, which is the whole point -- the
    alternative is a scattered selection per rank, which is what collective HDF5
    handles worst.

    Costs one argsort, one permutation and three collectives: 60 ms for 44.6 MB over
    four ranks with worst-case scattered row numbers, against a write of the order of
    a second. That is the price of keeping the file's row order independent of how
    many ranks wrote it, which is the convention `dolfin.XDMFFile` follows -- its
    node coordinates come out byte-identical on one rank and on three.
    """
    from femtools.hdf5_rows import agreed, block_edges, to_owners
    rank = comm.Get_rank()
    edges = block_edges(n_global, comm.Get_size())
    lo, hi = int(edges[rank]), int(edges[rank + 1])
    recv_rows, recv_values = to_owners(comm, edges, rows, np.asarray(values))
    local = recv_rows - lo
    # Every row of the file must be written exactly once. A gap means the ownership
    # rule dropped an entity and a repeat means two ranks claimed it; either way the
    # file would be quietly wrong, which is worse than failing here -- on every rank,
    # before the collective write
    bad = len(local) != hi - lo or len(np.unique(local)) != hi - lo
    if agreed(comm, bad):
        raise ValueError(
            'rank {} received {} rows ({} distinct) for a block of {}, so the '
            'ownership rule does not partition the entities'.format(
                rank, len(local), len(np.unique(local)), hi - lo) if bad else
            'another rank received rows that do not partition its block')
    slab = np.zeros((hi - lo,) + recv_values.shape[1:], dtype=recv_values.dtype)
    slab[local] = recv_values
    return lo, hi, slab


def _int32(a):
    """Topology as the XDMF files here store it, 32-bit: refused rather than wrapped past
    2^31 nodes (a P2 mesh has ~1.33 nodes a cell, so ~1.6B cells)."""
    a = np.asarray(a)
    if a.size and int(a.max()) >= 2 ** 31:
        raise ValueError("XDMF topology: node {} past the 32-bit range the file stores"
                         .format(int(a.max())))
    return a.astype(np.int32)


def _pad3(values):
    """XDMF's Vector is three-component; dolfin pads a 2D field the same way."""
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 2:
        return np.column_stack((values, np.zeros(len(values))))
    return values


class XDMFFile2(object):
    """A stand-in for `dolfin.XDMFFile` that does not throw the quadratic part away.

        with XDMFFile2(comm, 'u.xdmf') as f:
            f.write(u_)

    dolfin's own XDMFFile interpolates any Function down to one value per vertex
    before writing, so a P2 velocity comes out linear and the picture loses the
    midside nodes. This writes
    `Tetrahedron_10` cells instead, whose nodes are the P2 degrees of freedom, which
    ParaView renders once "Nonlinear Subdivision Level" is raised.

    Everything else dolfin's XDMFFile does, this does too, in dolfin's own file
    layout -- see the module docstring for the surface and for the three deliberate
    differences. Anything it genuinely cannot write raises and says why, rather than
    accepting the call and doing nothing with it.
    """

    Encoding = Encoding

    def __init__(self, comm_or_name, name=None):
        if name is None:
            self.comm, self.path = None, comm_or_name
        else:
            self.comm, self.path = comm_or_name, name
        self.parameters = XDMFParameters()
        self._doc = None
        self._name, self._label = '', ''
        self._grids = {}            # (mesh id, degree, time) -> grid element
        self._stored = {}           # (mesh id, degree) -> (topology ref, geometry ref)
        self._delegate = None       # a real dolfin.XDMFFile, for checkpoints
        self._mesh = None           # the last mesh seen, for a MeshValueCollection

    # -- dolfin's Variable surface --------------------------------------------

    def name(self):
        return self._name

    def label(self):
        return self._label

    def rename(self, name, label=''):
        self._name, self._label = name, label

    def id(self):
        return id(self)

    # -- lifetime --------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(compress_checkpoint=exc[0] is None)    # never mask the exception
        return False

    def close(self, compress_checkpoint=True):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None
            if not compress_checkpoint:
                return
            # dolfin's checkpoint cannot compress: its HDF5 file rewritten, as every
            # output is (`compress.rewrite`). An appended series is rewritten at each
            # close, its earlier steps copied as their stored chunks, not recompressed
            comm = self._mpi_comm()
            failed = None
            if comm.Get_rank() == 0:
                h5 = os.path.splitext(self.path)[0] + '.h5'
                code, what = compress.rewrite(h5)
                failed = '{}: {}'.format(h5, what) if code == 1 else None
            failed = comm.bcast(failed, root=0)
            if failed:
                raise RuntimeError('compressing the checkpoint ' + failed)

    def _mpi_comm(self):
        import dolfin as df
        return self.comm if self.comm is not None else df.MPI.comm_world

    def _document(self, encoding):
        Encoding.of(encoding)       # only HDF5 survives; anything else raises
        if self._doc is None:
            if self._delegate is not None:
                raise RuntimeError(
                    '{} already holds a checkpoint, which has a file layout of its '
                    'own; write the visualisation grids to another file'
                    .format(self.path))
            comm = self._mpi_comm()
            self._doc = XdmfDocument(self.path,
                                     comm=comm if _size(comm) > 1 else None)
        return self._doc

    def _finish(self):
        if self._doc is None:
            return
        if self.parameters['flush_output']:
            self._doc.flush()
        else:
            # the XML is kept complete after every write, close or no close: a
            # caller without a context manager otherwise ends with datasets in
            # the .h5 and no .xdmf naming them at all. `flush_output` only adds
            # the HDF5 flush; `save` is rank zero writing one small file.
            self._doc.save()

    def _rank(self):
        import dolfin as df
        comm = self.comm if self.comm is not None else df.MPI.comm_world
        return df.MPI.rank(comm)

    # -- write -----------------------------------------------------------------

    def write(self, obj, *args, time=None, encoding=None):
        """Writes a Function, a Mesh, a MeshFunction, a MeshValueCollection or points.

            f.write(u)                 f.write(u, t)
            f.write((u, p))            # several fields onto one node set
            f.write((u, p, tags))      # ... and a per-cell field beside them
            f.write(mesh)              f.write(facet_markers)
            f.write(points)            f.write(points, values)

        The trailing `encoding` argument dolfin takes is accepted in the same place,
        and `time=`/`encoding=` keywords as well. A time on anything but a Function
        is refused the way dolfin refuses it -- there is no grid to hang it on, and
        accepting it silently is the class of silence this module exists to remove.
        Several functions written together share one grid, the highest degree among
        them, so a P1 pressure beside a P2 velocity is carried onto the quadratic
        nodes. That is exact -- linear interpolation of a linear field -- but it does
        mean the file cannot tell you the pressure was only ever linear.

        A MeshFunction in the same tuple is written as a **cell-centred** attribute
        on that one grid rather than as a grid of its own: one value per cell, which
        is what a MeshFunction over the mesh's top dimension is. It must be over the
        same mesh and at that dimension, and is refused otherwise -- a facet
        MeshFunction has nothing to do with the cells of this grid, and writing it
        against them would put plausible numbers on the wrong entities.
        """
        import dolfin as df
        if _is_points(obj):
            # `write(points, values)` -- the values are an array, not the trailing
            # encoding, so this cannot go through the (time, encoding) parse below
            if time is not None:
                raise TypeError('points take no time; dolfin.XDMFFile refuses '
                                'this too')
            values = None
            for arg in args:
                if isinstance(arg, (list, tuple, np.ndarray)):
                    values = arg
                else:
                    encoding = _merge_write_arg(encoding, arg, 'encoding')
            return self._write_points(obj, values, encoding)

        t, e = _split_write_args(args)
        if t is not None:
            time = _merge_write_arg(time, t, 'time')
        if e is not None:
            encoding = _merge_write_arg(encoding, e, 'encoding')
        if isinstance(obj, (tuple, list)) and obj and any(
                isinstance(o, df.Function) for o in obj):
            # any position, not just the first: the docstring says order does not
            # matter, and dispatching on obj[0] made `write((tags, u))` fall through
            # to "does not write a tuple", which names the wrong problem
            functions, tags = _split_cell_tags(list(obj))
            return self._write_functions(functions, time, encoding, tags)
        if isinstance(obj, df.Function):
            return self._write_functions([obj], time, encoding)
        if time is not None:
            # dolfin has no write(Mesh, t) overload and says so; dropping the
            # time quietly wrote a grid with no <Time> and no error
            raise TypeError('write({}, t) takes no time; only a Function does'
                            .format(type(obj).__name__))
        if isinstance(obj, df.Mesh):
            return self._write_mesh_only(obj, encoding)
        if _is_mesh_function(obj):
            return self._write_mesh_function(obj, encoding)
        if _is_mesh_value_collection(obj):
            return self._write_mesh_value_collection(obj, encoding)
        raise TypeError('XDMFFile2 does not write a {}; dolfin.XDMFFile writes '
                        'Function, Mesh, MeshFunction, MeshValueCollection and '
                        'points'.format(type(obj).__name__))

    # -- functions -------------------------------------------------------------

    def _guard_gather(self, mesh):
        """Refuse, on every rank, to gather a large mesh to rank zero: under MPI with only
        serial h5py loaded the writes go through rank zero holding the whole mesh and
        field (~40 GB at 200M cells). Debian's h5py loads its MPI build only when
        OMPI_COMM_WORLD_SIZE or MPI_LOCALNRANKS is set, which srun with MPICH may not."""
        import dolfin as df
        if df.MPI.size(mesh.mpi_comm()) == 1:
            return
        n = mesh.num_entities_global(mesh.topology().dim())
        if n > GATHER_LIMIT:
            raise RuntimeError(
                'XDMFFile2: {} cells under MPI with a serial h5py would be gathered to rank '
                'zero; load the MPI build of h5py (H5PY_ALWAYS_USE_MPI=1) or raise '
                'FEMTOOLS_GATHER_LIMIT ({}) knowingly'.format(n, GATHER_LIMIT))

    def _write_functions(self, functions, time, encoding, tags=()):
        if not functions:
            raise ValueError('nothing to write')
        # every rank builds the document: opening the HDF5 collectively, and creating
        # each dataset, are calls every rank has to make, so nothing here may return
        # early the way the gather path does
        doc = self._document(encoding)
        if doc.parallel():
            return self._write_functions_parallel(doc, functions, time, tags)
        self._guard_gather(_check_functions(functions)[0])
        coords, conn, values, names, degree, mesh = _function_nodes(functions)
        self._mesh = mesh
        # gathered before the early return below: every rank holds a slice of the
        # values and the gather is collective, so a rank that has already left
        # would hang the ones that have not
        cells = _gather_cell_tags(tags, mesh)
        if self._rank() != 0:
            return
        cellname = mesh.ufl_cell().cellname()
        topology_type = XDMF_TOPOLOGY[(cellname, conn.shape[1])]

        key = (mesh.id(), degree, time)
        grid = self._grids.get(key) if self.parameters['functions_share_mesh'] else None
        if grid is None:
            parent = doc.collection() if time is not None else None
            grid = doc.grid('mesh', parent)
            store = self._stored.get((mesh.id(), degree))
            if store is None or self.parameters['rewrite_function_mesh']:
                group = doc.mesh_group(time is not None)
                store = (doc.dataset(group + '/topology', _int32(conn)),
                         doc.dataset(group + '/geometry', coords))
                self._stored[(mesh.id(), degree)] = store
            doc.topology(grid, conn, topology_type, ref=store[0])
            doc.geometry(grid, coords, ref=store[1])
            if time is not None:
                doc.time(grid, time)
            self._grids[key] = grid

        for name, value in zip(names, values):
            value = _pad3(value)
            ref = doc.dataset('VisualisationVector/{}'.format(
                doc._next('VisualisationVector')), value)
            doc.attribute(grid, name, value, 'Node', ref=ref)
        # the cell-centred attributes go on the same grid, after the nodal ones.
        # They live under VisualisationVector too: that group is where an attribute
        # of a function grid goes, whatever it is centred on and whatever its rank
        # -- dolfin puts scalars there as readily as vectors.
        for name, value in cells:
            ref = doc.dataset('VisualisationVector/{}'.format(
                doc._next('VisualisationVector')), value)
            doc.attribute(grid, name, value, 'Cell', ref=ref)
        self._finish()

    def _write_functions_parallel(self, doc, functions, time, tags=()):
        """The same file, written in slabs instead of gathered to rank zero.

        Every rank runs every line: the XML is built on all of them and thrown away
        on all but rank zero, which costs nothing and keeps the dataset counters --
        `Mesh/0`, `VisualisationVector/3` -- in step across ranks. They have to be,
        because creating a dataset is a collective operation and the ranks must agree
        on its name.
        """
        info = _function_nodes_distributed(functions)
        self._mesh = info['mesh']
        mesh, degree = info['mesh'], info['degree']
        conn, n_nodes, n_cells = info['conn'], info['n_nodes'], info['n_cells']
        topology_type = XDMF_TOPOLOGY[(mesh.ufl_cell().cellname(), conn.shape[1])]

        key = (mesh.id(), degree, time)
        grid = self._grids.get(key) if self.parameters['functions_share_mesh'] else None
        if grid is None:
            parent = doc.collection() if time is not None else None
            grid = doc.grid('mesh', parent)
            store = self._stored.get((mesh.id(), degree))
            if store is None or self.parameters['rewrite_function_mesh']:
                group = doc.mesh_group(time is not None)
                store = (doc.dataset_slab(group + '/topology', info['cell_offset'],
                                          _int32(conn), n_cells),
                         doc.dataset_distributed(group + '/geometry',
                                                 info['coord_rows'], info['coords'],
                                                 n_nodes))
                self._stored[(mesh.id(), degree)] = store
            doc.topology(grid, conn, topology_type, ref=store[0],
                         shape=(n_cells, conn.shape[1]))
            doc.geometry(grid, info['coords'], ref=store[1], shape=(n_nodes, 3))
            if time is not None:
                doc.time(grid, time)
            self._grids[key] = grid

        for name, rows, values in info['fields']:
            values = _pad3(values)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            ref = doc.dataset_distributed(
                'VisualisationVector/{}'.format(doc._next('VisualisationVector')),
                rows, values, n_nodes)
            doc.attribute(grid, name, values, 'Node', ref=ref,
                          shape=(n_nodes, values.shape[1]))
        # A cell attribute takes the same rank-major block the topology took, at the
        # same offset -- the file's row order for a cell is the order `conn` was
        # written in, and `mf.array()` is over the same local cells in the same
        # order. Cells are not shared between ranks, so unlike the node fields there
        # is no ownership question to settle.
        for name, values in _local_cell_tags(tags, mesh, len(conn)):
            ref = doc.dataset_slab(
                'VisualisationVector/{}'.format(doc._next('VisualisationVector')),
                info['cell_offset'], values, n_cells)
            doc.attribute(grid, name, values, 'Cell', ref=ref, shape=(n_cells, 1))
        self._finish()

    # -- mesh ------------------------------------------------------------------

    def _write_mesh_only(self, mesh, encoding):
        self._mesh = mesh
        doc = self._document(encoding)
        if doc.parallel():
            return self._write_mesh_parallel(doc, mesh)
        self._guard_gather(mesh)
        coords, cells = _gather_mesh(mesh, self.comm)
        if self._rank() != 0:
            return
        grid = doc.grid('mesh')
        group = doc.mesh_group(False)
        store = (doc.dataset(group + '/topology', _int32(cells)),
                 doc.dataset(group + '/geometry', coords))
        self._stored[(mesh.id(), 1)] = store
        doc.topology(grid, cells,
                     XDMF_TOPOLOGY[(mesh.ufl_cell().cellname(), cells.shape[1])],
                     ref=store[0])
        doc.geometry(grid, coords, ref=store[1])
        self._grids[(mesh.id(), 1, None)] = grid
        self._finish()

    def _write_mesh_parallel(self, doc, mesh):
        """The mesh in slabs: vertices by global index, cells rank-major."""
        gvert = np.asarray(mesh.topology().global_indices(0))
        owned = owned_entities(mesh, 0)
        n_vert = mesh.num_entities_global(0)
        tdim = mesh.topology().dim()
        n_cell = mesh.num_entities_global(tdim)
        cells = gvert[np.asarray(mesh.cells())]
        offset = _exscan(mesh.mpi_comm(), len(cells))

        grid = doc.grid('mesh')
        group = doc.mesh_group(False)
        store = (doc.dataset_slab(group + '/topology', offset,
                                  _int32(cells), n_cell),
                 doc.dataset_distributed(group + '/geometry', gvert[owned],
                                         _pad_coords(mesh.coordinates())[owned],
                                         n_vert))
        self._stored[(mesh.id(), 1)] = store
        doc.topology(grid, cells,
                     XDMF_TOPOLOGY[(mesh.ufl_cell().cellname(), cells.shape[1])],
                     ref=store[0], shape=(n_cell, cells.shape[1]))
        doc.geometry(grid, None, ref=store[1], shape=(n_vert, 3))
        self._grids[(mesh.id(), 1, None)] = grid
        self._finish()

    # -- mesh functions --------------------------------------------------------

    def _write_mesh_function(self, mf, encoding):
        mesh = mf.mesh()
        self._mesh = mesh
        dim = mf.dim()
        doc = self._document(encoding)
        if doc.parallel():
            return self._write_mesh_function_parallel(doc, mf)
        self._guard_gather(mesh)
        coords, entities, values = _gather_mesh_function(mf, self.comm)
        if self._rank() != 0:
            return
        index = doc._next('MeshFunction')
        group = 'MeshFunction/{}'.format(index)
        grid = doc.grid('mesh')
        topo_ref = doc.dataset(group + '/mesh/topology', _int32(entities))
        geom_ref = doc.dataset(group + '/mesh/geometry', coords)
        doc.topology(grid, entities, _entity_topology(mesh, dim, entities.shape[1]),
                     ref=topo_ref)
        doc.geometry(grid, coords, ref=geom_ref)
        name = mf.name() if mf.name() and mf.name() != 'f' else 'meshfunction'
        values = values.reshape(-1, 1)
        doc.attribute(grid, name, values, 'Cell',
                      ref=doc.dataset(group + '/values', values))
        self._finish()

    def _write_mesh_function_parallel(self, doc, mf):
        """Vertices by global index; entities and their values rank-major together.

        The two must go to the *same* block on each rank or a value would end up
        against another entity's row -- which is why they share one offset rather than
        each taking their own.
        """
        mesh, dim = mf.mesh(), mf.dim()
        gvert = np.asarray(mesh.topology().global_indices(0))
        owned_v = owned_entities(mesh, 0)
        owned_e = owned_entities(mesh, dim)
        n_vert = mesh.num_entities_global(0)
        n_ent = mesh.num_entities_global(dim)
        rows = gvert[_connectivity(mesh, dim)][owned_e]
        values = np.asarray(mf.array())[owned_e].reshape(-1, 1)
        offset = _exscan(mesh.mpi_comm(), int(owned_e.sum()))

        group = 'MeshFunction/{}'.format(doc._next('MeshFunction'))
        grid = doc.grid('mesh')
        topo = doc.dataset_slab(group + '/mesh/topology', offset,
                                _int32(rows), n_ent)
        geom = doc.dataset_distributed(group + '/mesh/geometry', gvert[owned_v],
                                       _pad_coords(mesh.coordinates())[owned_v],
                                       n_vert)
        doc.topology(grid, rows, _entity_topology(mesh, dim, rows.shape[1]),
                     ref=topo, shape=(n_ent, rows.shape[1]))
        doc.geometry(grid, None, ref=geom, shape=(n_vert, 3))
        name = mf.name() if mf.name() and mf.name() != 'f' else 'meshfunction'
        doc.attribute(grid, name, values, 'Cell',
                      ref=doc.dataset_slab(group + '/values', offset, values, n_ent),
                      shape=(n_ent, 1))
        self._finish()

    def _write_mesh_value_collection(self, mvc, encoding):
        # A MeshValueCollection holds a mesh in C++ and does not expose it to Python
        # in this dolfin -- `dir(mvc)` has dim, size, values and nothing else -- so
        # unlike every other object here it cannot say what it is defined over. The
        # mesh already written to this file is the one it belongs to in any use that
        # makes sense; with no mesh written, there is nothing to guess from.
        mesh = self._mesh
        if mesh is None:
            raise ValueError(
                'a MeshValueCollection does not carry its mesh in this dolfin, so '
                'write the mesh (or a MeshFunction, or a Function) to {} first'
                .format(self.path))
        # dolfin visualises it as the entities it marks, which is what a MeshFunction
        # over the same dimension holds -- but only the marked ones, so it is built
        # by hand rather than by widening it to every entity.
        dim = mvc.dim()
        tdim = mesh.topology().dim()
        mesh.init(tdim, dim)
        mesh.init(dim, 0)
        doc = self._document(encoding)
        parallel = doc.parallel()
        # a distributed mesh needs global vertex numbers and the ownership filter
        # whether the file is written collectively or gathered: the gather path
        # used to skip both, writing rank zero's local entities under rank-local
        # vertex numbers and losing every other rank's marks
        distributed = _size(self._mpi_comm()) > 1
        need_global = parallel or distributed
        owned = owned_entities(mesh, dim) if need_global else None
        gvert = (np.asarray(mesh.topology().global_indices(0))
                 if need_global else None)
        triples = None
        try:
            triples = _cpp().mvc_values(mvc)
        except TypeError:       # a value type the helper has no overload for
            pass                # (only size_t has one; the loop below takes the rest)
        if triples is not None:
            # three aligned arrays in map order, then everything is indexing:
            # the entity each pair names, the ownership filter, its vertices
            cell_of, local_of, values = (np.asarray(a) for a in triples)
            entity = (_connectivity(mesh, tdim, dim)[cell_of, local_of]
                      if len(cell_of) else np.zeros(0, dtype=np.int64))
            if need_global:
                keep = owned[entity]
                entity, values = entity[keep], values[keep]
            rows = _connectivity(mesh, dim, 0)[entity]
            if need_global:
                rows = gvert[rows]
        else:
            cell_entities = mesh.topology()(tdim, dim)
            entity_vertices = mesh.topology()(dim, 0)
            rows, values = [], []
            for (cell, local), value in mvc.values().items():
                entity = cell_entities(cell)[local]
                if need_global and not owned[entity]:
                    continue          # another rank writes this one
                vertices_of = np.asarray(entity_vertices(entity))
                rows.append(gvert[vertices_of] if need_global else vertices_of)
                values.append(value)
        if parallel:
            return self._write_mvc_parallel(doc, mvc, mesh, dim, rows, values)
        entities = np.array(rows, dtype=np.int64).reshape(len(rows), -1) \
            if len(rows) else np.zeros((0, dim + 1), dtype=np.int64)
        values = np.asarray(values, dtype=_mvc_dtype(mvc)).reshape(-1, 1)
        if distributed:
            # every rank sends its owned marks and rank zero writes the lot,
            # against coordinates assembled by global vertex index
            mpi = mesh.mpi_comm()
            coords = gather_global(mpi, gvert, _pad_coords(mesh.coordinates()),
                               mesh.num_entities_global(0))
            parts = mpi.gather((entities, values), root=0)
            if self._rank() != 0:
                return
            entities = np.vstack([p[0] for p in parts])
            values = np.vstack([p[1] for p in parts])
        else:
            coords = _pad_coords(mesh.coordinates())
        if self._rank() != 0:
            return
        index = doc._next('MeshValueCollection')
        group = 'MeshValueCollection/{}'.format(index)
        grid = doc.grid(mvc.name() or 'm')
        doc.topology(grid, entities, _entity_topology(mesh, dim, entities.shape[1]),
                     ref=doc.dataset(group + '/topology', _int32(entities)))
        doc.geometry(grid, coords, ref=doc.dataset(group + '/geometry', coords))
        doc.attribute(grid, mvc.name() or 'm', values, 'Cell',
                      ref=doc.dataset(group + '/values', values))
        self._finish()

    def _write_mvc_parallel(self, doc, mvc, mesh, dim, rows, values):
        """Only the marked entities, rank-major, with the vertices they refer to."""
        import dolfin as df
        comm = mesh.mpi_comm()
        width = dim + 1
        entities = np.array(rows, dtype=np.int64).reshape(len(rows), width) \
            if len(rows) else np.zeros((0, width), dtype=np.int64)
        marks = np.asarray(values, dtype=_mvc_dtype(mvc)).reshape(-1, 1)
        n_marked = int(df.MPI.sum(comm, float(len(entities))))
        offset = _exscan(comm, len(entities))
        gvert = np.asarray(mesh.topology().global_indices(0))
        owned_v = owned_entities(mesh, 0)
        n_vert = mesh.num_entities_global(0)

        group = 'MeshValueCollection/{}'.format(doc._next('MeshValueCollection'))
        name = mvc.name() or 'm'
        grid = doc.grid(name)
        doc.topology(grid, entities, _entity_topology(mesh, dim, width),
                     ref=doc.dataset_slab(group + '/topology', offset,
                                          _int32(entities), n_marked),
                     shape=(n_marked, width))
        doc.geometry(grid, None, shape=(n_vert, 3),
                     ref=doc.dataset_distributed(
                         group + '/geometry', gvert[owned_v],
                         _pad_coords(mesh.coordinates())[owned_v], n_vert))
        doc.attribute(grid, name, marks, 'Cell', shape=(n_marked, 1),
                      ref=doc.dataset_slab(group + '/values', offset, marks,
                                           n_marked))
        self._finish()

    # -- points ----------------------------------------------------------------

    def _write_points(self, points, values, encoding):
        try:
            coords = np.asarray(_cpp().points_array(points))
        except TypeError:           # not dolfin Points; the caster refused them
            coords = np.array([p.array() if hasattr(p, 'array') else list(p)
                               for p in points], dtype=float)
        coords = _pad_coords(coords)
        doc = self._document(encoding)
        if values is not None:
            values = np.asarray(values, dtype=float).reshape(-1, 1)
        if doc.parallel():
            # Points are given to every rank, not partitioned across them, so rank
            # zero writes the lot and the others take part with an empty slab. That
            # keeps the meaning it has in serial rather than concatenating the same
            # list once per rank.
            n = len(coords)
            mine = slice(0, n) if doc.rank == 0 else slice(0, 0)
            conn = np.arange(n, dtype=np.int64).reshape(-1, 1)
            group = doc.free_group('Points')
            grid = doc.grid('points')
            doc.topology(grid, conn, 'Polyvertex', shape=(n, 1),
                         ref=doc.dataset_slab(group + '/topology', 0,
                                              _int32(conn)[mine], n))
            doc.geometry(grid, None, shape=(n, 3),
                         ref=doc.dataset_slab(group + '/geometry', 0,
                                              coords[mine], n))
            if values is not None:
                doc.attribute(grid, 'values', values, 'Node', shape=(n, 1),
                              ref=doc.dataset_slab(group + '/values', 0,
                                                   values[mine], n))
            return self._finish()
        if self._rank() != 0:
            return
        group = doc.free_group('Points')
        grid = doc.grid('points')
        conn = np.arange(len(coords), dtype=np.int64).reshape(-1, 1)
        doc.topology(grid, conn, 'Polyvertex',
                     ref=doc.dataset(group + '/topology', _int32(conn)))
        doc.geometry(grid, coords, ref=doc.dataset(group + '/geometry', coords))
        if values is not None:
            doc.attribute(grid, 'values', values, 'Node',
                          ref=doc.dataset(group + '/values', values))
        self._finish()

    # -- checkpoints -----------------------------------------------------------

    def write_checkpoint(self, u, function_name, time_step=0.0, encoding=None,
                         append=False):
        """dolfin's checkpoint, unchanged -- it already round-trips P2 exactly.

        The checkpoint stores the dof vector and the element beside it, so nothing is
        lost and there is nothing here to improve on. It has a file layout of its own,
        though, so a file cannot hold both a checkpoint and visualisation grids;
        asking for both says so. The parameters are dolfin's, names included
        (`function_name`, `time_step`), and `append` is forwarded: dropping it made
        a time series keep only its last step, since dolfin truncates the file on
        every non-appending call.
        """
        import dolfin as df
        if self._doc is not None:
            raise RuntimeError(
                '{} already holds visualisation grids, and a checkpoint has a file '
                'layout of its own; use another file'.format(self.path))
        if self._delegate is None:
            comm = self.comm if self.comm is not None else df.MPI.comm_world
            self._delegate = df.XDMFFile(comm, self.path)
            for key, value in self.parameters.items():
                self._delegate.parameters[key] = value
        Encoding.of(encoding)   # only HDF5 survives, which is dolfin's default
        self._delegate.write_checkpoint(u, function_name, float(time_step),
                                        df.XDMFFile.Encoding.HDF5, bool(append))

    def read_checkpoint(self, u, function_name, counter=-1):
        """Reads dolfin's checkpoint, or a quadratic grid written here.

        dolfin's reader knows only its own checkpoint format. If the file instead
        holds `Tetrahedron_10` visualisation grids -- which dolfin cannot read at all
        -- the field is rebuilt from the nodes, onto an unconstrained space, and then
        interpolated onto whatever space `u` lives in. Going through the
        unconstrained space is what makes it safe for a periodic one, where matching
        dof coordinates does not line up node for node.

        `counter` defaults to -1 -- the last stored step -- because that is dolfin's
        default; a drop-in defaulting to the first step would hand back the wrong
        field with no error.
        """
        import dolfin as df
        grids = _scan(self.path)
        if any(g['topology'] in ('Tetrahedron_10', 'Triangle_6', 'Edge_3')
               for g in grids):
            return self._read_quadratic(u, function_name, counter, grids)
        comm = self.comm if self.comm is not None else df.MPI.comm_world
        with df.XDMFFile(comm, self.path) as f:
            f.read_checkpoint(u, function_name, counter)

    def _read_quadratic(self, u, name, counter, grids):
        import dolfin as df
        picked = [g for g in grids if name in g['attributes']]
        if not picked:
            raise ValueError('no field named {!r} in {}; the file has {}'.format(
                name, self.path, ', '.join(sorted(
                    {a for g in grids for a in g['attributes']})) or 'none'))
        if not -len(picked) <= counter < len(picked):
            raise ValueError('{} has {} step(s) of {!r}; counter {} is out of '
                             'range'.format(self.path, len(picked), name, counter))
        grid = picked[counter]
        coords, conn = _read_grid_geometry(self.path, grid)
        values = _read_grid_attribute(self.path, grid, name)

        V = u.function_space()
        element = V.ufl_element()
        degree = element.degree()
        width = int(np.prod(u.value_shape())) if len(u.value_shape()) else 1
        mesh = V.mesh()
        cell = mesh.ufl_cell()
        # The detour through an unconstrained space is what makes this safe for a
        # *periodic* one, where matching dof coordinates is not. But when the caller's space is already
        # that unconstrained space there is nothing to detour around, and building a
        # second copy of it to interpolate out of is pure cost -- 13.9 s against
        # 6.6 s for the real packing's P2 velocity, for a bit-identical vector.
        # `_is_plain` is the writer's own test, and it is exact rather than a
        # heuristic: a periodic space is strictly smaller than the plain one.
        if _is_plain(V, mesh, degree, width):
            _scatter_nodes(u, V, coords, values, width, degree)
            return u
        if width > 1:
            plain = df.FunctionSpace(mesh, df.VectorElement(
                'Lagrange', cell, degree, dim=width))
        else:
            plain = df.FunctionSpace(mesh, df.FiniteElement('Lagrange', cell, degree))
        target = df.Function(plain)
        _scatter_nodes(target, plain, coords, values, width, degree)
        u.assign(df.interpolate(target, V))
        return u

    # -- read ------------------------------------------------------------------

    def read(self, obj, name=''):
        """Reads a Mesh, a MeshFunction or a MeshValueCollection.

        A file with no quadratic grid in it is handed to dolfin, which reads its own
        layout -- and ours, which is the same one -- with no help from here. Only a
        file holding `Tetrahedron_10`, which dolfin's reader cannot parse, is read
        here instead.
        """
        import dolfin as df
        grids = _scan(self.path)
        quadratic = any(g['topology'] in ('Tetrahedron_10', 'Triangle_6', 'Edge_3')
                        for g in grids)
        if not quadratic:
            comm = self.comm if self.comm is not None else df.MPI.comm_world
            with df.XDMFFile(comm, self.path) as f:
                return f.read(obj) if not name else f.read(obj, name)
        if isinstance(obj, df.Mesh):
            return self._read_mesh(obj, grids)
        raise ValueError(
            '{} holds quadratic cells, which this reads back only as a Mesh (its '
            'corners) or through read_checkpoint (the field)'.format(self.path))

    def _read_mesh(self, mesh, grids):
        import dolfin as df
        if df.MPI.size(mesh.mpi_comm()) > 1:
            raise RuntimeError('reading a quadratic XDMF is serial only; dolfin '
                               'partitions its own files, this does not')
        linear = [g for g in grids if g['topology'] in ('Tetrahedron', 'Triangle')]
        grid = linear[0] if linear else grids[0]
        coords, conn = _read_grid_geometry(self.path, grid)
        cellname, per_cell = TOPOLOGY_CELL[grid['topology']]
        corners = conn[:, :{'tetrahedron': 4, 'triangle': 3,
                            'interval': 2}[cellname]]
        used, conn = np.unique(corners, return_inverse=True)
        conn = conn.reshape(corners.shape)
        coords = coords[used]
        editor = df.MeshEditor()
        editor.open(mesh, cellname, {'tetrahedron': 3, 'triangle': 2,
                                     'interval': 1}[cellname], coords.shape[1])
        editor.init_vertices(len(coords))
        editor.init_cells(len(conn))
        for i, x in enumerate(coords):
            editor.add_vertex(i, x)
        for i, c in enumerate(conn):
            editor.add_cell(i, np.asarray(c, dtype=np.uintp))
        editor.close()
        return mesh


# ---------------------------------------------------------------------------
# the dolfin-shaped helpers: everything that has to touch a Function or a Mesh
# ---------------------------------------------------------------------------

def _size(comm):
    if comm is None:
        return 1
    return comm.Get_size() if hasattr(comm, 'Get_size') else 1


def _split_write_args(args):
    """dolfin's trailing `(t, encoding)`, either, both or neither.

    A time can arrive as a numpy scalar -- `times[i]` off a float32 array is an
    `np.float32`, which is not a `float` subclass -- so the test is numpy-aware.
    Booleans are neither: `True` routed to time would be silently accepted as 1.0.
    """
    time, encoding = None, None
    for arg in args:
        if isinstance(arg, (bool, np.bool_)):
            encoding = arg              # let Encoding.of refuse it by name
        elif isinstance(arg, (float, int, np.floating, np.integer)):
            time = float(arg)
        else:
            encoding = arg
    return time, encoding


def _merge_write_arg(keyword, positional, what):
    """One value for `time` or `encoding`, given positionally or by keyword."""
    if keyword is not None and positional is not None:
        raise TypeError('{} given twice, positionally and by keyword'.format(what))
    return positional if positional is not None else keyword


def _is_points(obj):
    """A list of dolfin Points, as `write(points)` takes -- not a list of Functions."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    first = obj[0]
    return hasattr(first, 'x') and hasattr(first, 'y') and hasattr(first, 'z')


def _mvc_dtype(mvc):
    """The dtype a MeshValueCollection's values really have, off its class name.

    They were all forced to uint64, which floors a double collection's values and
    wraps a -1 'unmarked' int to 2**64-1 -- dolfin writes each value type
    faithfully, so must this. dolfin's classes are MeshValueCollection_sizet,
    _double, _int and _bool.
    """
    name = type(mvc).__name__
    if name.endswith('_double'):
        return np.float64
    if name.endswith('_int'):
        return np.int32
    if name.endswith('_bool'):
        return np.uint8
    return np.uint64


def _is_mesh_function(obj):
    return type(obj).__name__.startswith('MeshFunction')


def _is_mesh_value_collection(obj):
    return type(obj).__name__.startswith('MeshValueCollection')


def _pad_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] < 3:
        coords = np.column_stack(
            [coords] + [np.zeros(len(coords))] * (3 - coords.shape[1]))
    return coords


_CPP_CODE = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshValueCollection.h>
namespace py = pybind11;

// MeshConnectivity stores the whole table as one contiguous vector, which the
// Python layer does not expose (its MeshConnectivity has `size` and nothing
// else); this hands it to numpy in one copy.
py::array_t<unsigned int> connectivity(const dolfin::Mesh& mesh,
                                       std::size_t d0, std::size_t d1)
{
  const std::vector<unsigned int>& c = mesh.topology()(d0, d1)();
  return py::array_t<unsigned int>(c.size(), c.data());
}

// A list of Points costs three Python method calls per point to read from
// Python; the stl caster converts the whole list at C++ speed instead.
py::array_t<double> points_array(const std::vector<dolfin::Point>& pts)
{
  py::array_t<double> out({pts.size(), std::size_t(3)});
  double* p = out.mutable_data();
  for (const auto& pt : pts) { *p++ = pt.x(); *p++ = pt.y(); *p++ = pt.z(); }
  return out;
}

// The (cell, local_entity) -> value map of a MeshValueCollection, as three
// aligned arrays in map order. From Python the map arrives as a dict costing
// per entry; from here it is three pointer walks.
py::tuple mvc_values(const dolfin::MeshValueCollection<std::size_t>& mvc)
{
  const auto& map = mvc.values();
  const std::size_t n = map.size();
  py::array_t<std::int64_t> cells(n), locals(n);
  py::array_t<std::size_t> vals(n);
  auto* c = cells.mutable_data();
  auto* l = locals.mutable_data();
  auto* v = vals.mutable_data();
  for (const auto& kv : map) {
    *c++ = kv.first.first;
    *l++ = kv.first.second;
    *v++ = kv.second;
  }
  return py::make_tuple(cells, locals, vals);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("connectivity", &connectivity);
  m.def("points_array", &points_array);
  m.def("mvc_values", &mvc_values);
}
'''
_CPP_MODULE = False             # False: not tried yet; the module afterwards


def _cpp():
    """The JIT-compiled C++ helpers. Raises if they will not build.

    `dolfin.compile_cpp_code` needs a compiler and the dolfin headers
    (`libdolfin-dev`); the first call compiles (~4 s) and dijitso caches the
    result, so every later process pays milliseconds. Both are assumed present,
    as they are by dolfin itself, which JIT-compiles every
    variational form -- so this is only reached where all of that already worked.

    A `np.fromiter` fallback for the connectivity used to stand behind this, with
    `XDMF2_NO_CPP=1` to reach it. It is gone: nothing in production could take it,
    it existed for a machine this project cannot run on anyway, and a branch whose
    only evidence of correctness is a test that forces it is a liability rather
    than a safety net. The two fallbacks that remain below are not about the
    compiler at all -- they are about *argument types* the pybind casters refuse,
    which happens with a working helper.

    This is reached only from the dolfin-facing paths; the document core read HDF5 with numpy and h5py and never come here, so the module
    is still usable with no dolfin at all.
    """
    global _CPP_MODULE
    if _CPP_MODULE is False:
        import dolfin as df
        _CPP_MODULE = df.compile_cpp_code(_CPP_CODE)
    return _CPP_MODULE


def _connectivity(mesh, dim, of_dim=0):
    """The entity-to-vertex table of `dim`, as an array rather than a Python list.

    `mesh.topology()(d, 0)()` hands back a **list** of ints -- 295200 of them for the
    facets of a 48k-cell mesh -- and walking it was most of the cost of writing a
    MeshFunction, at about 4.5x dolfin's time. The C++ storage behind that list is
    one contiguous vector, so the JIT helper above reads it in one copy (measured
    96x quicker than the list at 777k facets) and the ratio drops to about one.
    The two dimensions that have a real array behind them avoid the question
    altogether.
    """
    tdim = mesh.topology().dim()
    if of_dim == 0 and dim == tdim:
        return np.asarray(mesh.cells(), dtype=np.int64)
    if of_dim == 0 and dim == 0:
        return np.arange(mesh.num_entities(0), dtype=np.int64).reshape(-1, 1)
    mesh.init(dim, of_dim)
    conn = mesh.topology()(dim, of_dim)
    n_entities = mesh.num_entities(dim)
    per = conn.size(0) if n_entities else 0
    flat = _cpp().connectivity(mesh, dim, of_dim).astype(np.int64)
    return flat.reshape(n_entities, per)


def _entity_topology(mesh, dim, per_cell):
    cellname = {0: 'point', 1: 'interval', 2: 'triangle', 3: 'tetrahedron'}[dim]
    return XDMF_TOPOLOGY[(cellname, per_cell)]


def _gather_mesh(mesh, comm):
    """Global vertices and cells of a possibly distributed mesh, on rank zero.

    Ghost entities arrive from more than one rank; they are written by global index,
    so the copies land on top of each other and agree.
    """
    tdim = mesh.topology().dim()
    mpi = mesh.mpi_comm()
    gvert = np.asarray(mesh.topology().global_indices(0))
    gcell = np.asarray(mesh.topology().global_indices(tdim))
    coords = gather_global(mpi, gvert, _pad_coords(mesh.coordinates()),
                       mesh.num_entities_global(0))
    cells = gather_global(mpi, gcell, gvert[np.asarray(mesh.cells())],
                      mesh.num_entities_global(tdim))
    return coords, cells


def _gather_mesh_function(mf, comm):
    """The entities a MeshFunction is defined over, and its values, on rank zero."""
    mesh = mf.mesh()
    dim = mf.dim()
    mesh.init(dim, 0)
    mesh.init_global(dim)
    gvert = np.asarray(mesh.topology().global_indices(0))
    gent = np.asarray(mesh.topology().global_indices(dim))
    rows = gvert[_connectivity(mesh, dim)]
    mpi = mesh.mpi_comm()
    coords = gather_global(mpi, gvert, _pad_coords(mesh.coordinates()),
                       mesh.num_entities_global(0))
    n_e = mesh.num_entities_global(dim)
    entities = gather_global(mpi, gent, rows, n_e)
    values = gather_global(mpi, gent, np.asarray(mf.array()), n_e)
    return coords, entities, values


_PLAIN_SPACES = {}


def scatter_vertex_values(f, per_vertex, width=1):
    """Per-vertex values into a Function's owned dofs, ghosts left to `apply`.

    `vertex_to_dof_map` covers this rank's
    ghost vertices too, and their dof indices run past the end of `get_local()`,
    which is over the ownership range alone -- assigning through them would either
    raise or, with a negative index, write a value onto some other node's dof.

    A blocked dofmap keeps a node's components together, so testing the first
    component covers all `width` of them. Works on a periodic space as well:
    `vertex_to_dof_map` sends a slave vertex to its master's dof, so both faces
    write the same dof with the same value.
    """
    import dolfin as df
    v2d = df.vertex_to_dof_map(f.function_space()).reshape(-1, width)
    values = f.vector().get_local()
    owned = (v2d[:, 0] >= 0) & (v2d[:, 0] < len(values))
    values[v2d[owned]] = np.asarray(per_vertex).reshape(len(v2d), width)[owned]
    f.vector().set_local(values)
    f.vector().apply("insert")
    return f


def _plain_space(mesh, degree, width):
    """An unconstrained Lagrange space, built once per mesh, degree and width.

    Building one is not cheap -- 48 ms for a P2 space on 48k cells, which was more
    than half the cost of writing a field -- and a time series would otherwise build
    an identical one for every step.
    """
    import dolfin as df
    key = (mesh.id(), degree, width)
    space = _PLAIN_SPACES.get(key)
    if space is None:
        cell = mesh.ufl_cell()
        element = (df.VectorElement('Lagrange', cell, degree, dim=width) if width > 1
                   else df.FiniteElement('Lagrange', cell, degree))
        space = df.FunctionSpace(mesh, element)
        _PLAIN_SPACES[key] = space
    return space


def _is_plain(V, mesh, degree, width):
    """Whether a space is already unconstrained Lagrange of this degree and width.

    A periodic space identifies its slave dofs with masters, so it is strictly
    smaller than the unconstrained one; comparing the dimensions is therefore an
    exact test and not a heuristic. When it holds, the field can be read straight out
    of its own dofs and neither a new space nor an interpolation is needed.
    """
    element = V.ufl_element()
    if element.family() not in ('Lagrange', 'P', 'CG') or element.degree() != degree:
        return False
    n = mesh.num_entities_global(0)
    if degree == 2:
        mesh.init(1)
        mesh.init_global(1)
        n += mesh.num_entities_global(1)
    return V.dim() == n * width


#: Bytes one gather round may deliver to rank zero, all ranks together. MPI addresses a
#: receive buffer with int counts and displacements, so 2**31 - 1 is the ceiling; a
#: module constant so a test can force many rounds with small arrays.
GATHER_ROUND_BYTES = 2**31 - 1


def gather_rows(mpi, a):
    """Every rank's rows of `a`, concatenated in rank order on rank zero; None elsewhere.

    Raw bytes through `Gatherv`, in rounds of at most `GATHER_ROUND_BYTES` in total.
    mpi4py's lower-case `gather` pickles and receives everything into one buffer
    addressed in int bytes, so it fails past 2 GiB in all -- by a hang or a segfault,
    depending on the MPI underneath, never by a message. It did: the Pe=1000 ladder's
    rung 4 build (~515M cells, ~8 GB of parent map) died in rank 0 on 2026-09-25, where
    rung 3 (1.5 GB) had passed. Rows keep their dtype and trailing shape.
    """
    from mpi4py import MPI
    a = np.ascontiguousarray(a)
    rank, size = mpi.Get_rank(), mpi.Get_size()
    # Bytes carry no type, so every rank must read them the same way. The pickling
    # gather sent each rank's dtype along; here the ranks agree first, from the ranks
    # that have rows -- an empty `np.zeros(0)` on a rank with nothing to send must not
    # decide how rank zero reads everyone else's. All refuse together, or none does.
    meta = mpi.allgather((a.shape[0], a.dtype.str, a.shape[1:], a.dtype.hasobject))
    counts = np.asarray([m[0] for m in meta], dtype=np.int64)
    kinds = {(m[1], m[2]) for m in meta if m[0] > 0}
    if len(kinds) > 1 or any(m[3] for m in meta):
        raise ValueError("gather_rows: the ranks' rows disagree in dtype or shape, or are "
                         "Python objects, which cannot go as bytes: {}".format(sorted(kinds)))
    if kinds:
        dtype_str, tail = kinds.pop()
        a = a.astype(np.dtype(dtype_str), copy=False).reshape((-1,) + tuple(tail))
    row_bytes = max(1, a.dtype.itemsize * int(np.prod(a.shape[1:], dtype=np.int64)))
    per_round = max(1, GATHER_ROUND_BYTES // (size * row_bytes))   # rows per rank per round
    starts = np.concatenate(([0], np.cumsum(counts)))
    out = (np.empty((int(counts.sum()),) + a.shape[1:], dtype=a.dtype)
           if rank == 0 else None)
    flat = a.view(np.uint8).reshape(-1) if a.size else np.zeros(0, np.uint8)
    for lo in range(0, int(counts.max()) if size else 0, per_round):
        take = np.clip(counts - lo, 0, per_round)                  # rows each rank sends
        mine = flat[lo * row_bytes:(lo + int(take[rank])) * row_bytes]
        if rank == 0:
            sizes = take * row_bytes
            displs = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            buf = np.empty(int(sizes.sum()), dtype=np.uint8)
            mpi.Gatherv(mine, [buf, sizes.tolist(), displs.tolist(), MPI.BYTE], root=0)
            rows = buf.view(a.dtype).reshape((-1,) + a.shape[1:])
            pos = 0
            for r, t in enumerate(take.tolist()):
                out[starts[r] + lo:starts[r] + lo + t] = rows[pos:pos + t]
                pos += t
        else:
            mpi.Gatherv(mine, None, root=0)
    return out


def gather_global(mpi, index, payload, n_global):
    """Rank zero's view of a distributed array, addressed by global entity index.

    Rank-local rows carrying the global row numbers they belong to.

    In serial this is a reindex and nothing else -- no gather, which matters: an
    mpi4py `gather` pickles its payload even with one rank, and on the formats dolfin
    also writes that pickling was a large part of the cost of writing them.
    """
    import dolfin as df
    payload = np.asarray(payload)
    if df.MPI.size(mpi) == 1:
        out = np.empty((n_global,) + payload.shape[1:], dtype=payload.dtype)
        out[index] = payload
        covered = np.zeros(n_global, dtype=bool)
        covered[index] = True
        if not covered.all():
            raise ValueError(_uncovered_message(n_global, int((~covered).sum())))
        return out
    where = gather_rows(mpi, np.asarray(index, dtype=np.int64))
    value = gather_rows(mpi, payload)
    out, missing = None, 0
    if df.MPI.rank(mpi) == 0:
        # shaped like the gathered rows, not rank zero's own: an empty rank zero may
        # hold a default-typed array (gather_rows agrees the dtype; so must this)
        out = np.empty((n_global,) + value.shape[1:], dtype=value.dtype)
        seen = np.zeros(n_global, dtype=bool)
        out[where] = value           # ghost copies land on top of each other, equal
        seen[where] = True
        missing = int((~seen).sum())
    # broadcast rather than raise where it was found: only rank zero holds the
    # gathered rows and so only it can tell, but if it alone raised, every other rank
    # would sit in the next collective for ever. A hang says far less than a message.
    missing = mpi.bcast(missing, root=0)
    if missing:
        raise ValueError(_uncovered_message(n_global, missing))
    return out


def _uncovered_message(n_global, missing):
    """Why it matters that every row of the file got a value.

    `np.empty` does not fill, so a row no rank claimed is whatever was in that memory.
    Unlike `_redistribute`, which owns its rows and can insist on *exactly* once, this
    is only "at least once": entities shared between ranks arrive from each of them,
    with the same value, which is how the gather path is meant to work.

    The check earns its place. A surface mesh built by hand through
    `MeshEditor.add_cell` cannot be given a global cell index -- so every
    rank numbered its cells 0..n-1, they landed on top of each other here, and the
    rows past the largest local count kept whatever `np.empty` had picked up. The file
    was written, the process exited 0, every printed total was right, and the topology
    in it ran from -2144430970 to 1969384815.
    """
    return ('assembling {} rows left {} of them unwritten: the entity indices are '
            'not a cover, so some rank\'s numbering is local where it should be '
            'global'.format(n_global, missing))


def _split_cell_tags(objects):
    """A `write((u, p, tags))` tuple split into its Functions and its MeshFunctions.

    Order does not matter -- the nodal fields are written first regardless -- but
    anything that is neither is refused by name rather than failing later inside the
    node assembly, where the message would be about a missing `function_space`.
    """
    import dolfin as df
    functions, tags = [], []
    for obj in objects:
        if isinstance(obj, df.Function):
            functions.append(obj)
        elif _is_mesh_function(obj):
            tags.append(obj)
        else:
            raise TypeError(
                'a tuple holds Functions to write on one node set and MeshFunctions '
                'to write as cell data beside them, not a {}'
                .format(type(obj).__name__))
    if not functions:
        raise ValueError('cell data needs a grid to sit on: a tuple of MeshFunctions '
                         'alone has no node set. Write one on its own -- f.write(mf) '
                         '-- for a grid of its own')
    return functions, tags


def _check_cell_tags(tags, mesh):
    """Every MeshFunction in a `write((u, ..., mf))` tuple is cell data for that grid.

    Two things have to hold and neither is safe to assume. It must be over the *same*
    mesh, or its values are being written against another mesh's cells. And it must be
    over the mesh's **top** dimension, which is the only dimension with one value per
    cell of the grid: a facet MeshFunction has a different length and a different
    meaning, and pairing it with the cells would put entirely plausible numbers on the
    wrong entities -- the sort of file that looks right in ParaView and is not.
    """
    tdim = mesh.topology().dim()
    for mf in tags:
        if mf.mesh().id() != mesh.id():
            raise ValueError('the MeshFunction is over a different mesh from the '
                             'functions; it cannot be their cell data')
        if mf.dim() != tdim:
            raise ValueError(
                'a MeshFunction written beside functions is cell data, so it must be '
                'over dimension {} (the cells); this one is over dimension {}. Write '
                'it on its own -- f.write(mf) -- for a grid of its own'
                .format(tdim, mf.dim()))
    return tdim


def _tag_name(mf, index):
    """A MeshFunction's name, or one that says which of them it was.

    'f' is dolfin's placeholder for an unnamed MeshFunction, and `_write_mesh_function`
    already treats it as no name at all; this agrees with it.
    """
    name = mf.name()
    return name if name and name != 'f' else 'meshfunction{}'.format(
        '' if index == 0 else index)


def _gather_cell_tags(tags, mesh):
    """(name, values) per MeshFunction, on rank zero, addressed by global cell index.

    The rows of a function grid's topology are global cell indices -- `_function_nodes`
    assembles `conn` against them -- so the cell attribute must be assembled the same
    way or the values would sit against the wrong cells on more than one rank.
    """
    if not tags:
        return []
    _check_cell_tags(tags, mesh)
    tdim = mesh.topology().dim()
    mpi = mesh.mpi_comm()
    gcell = np.asarray(mesh.topology().global_indices(tdim))
    n_cell = mesh.num_entities_global(tdim)
    out = []
    for index, mf in enumerate(tags):
        values = np.asarray(mf.array()).reshape(-1, 1)
        out.append((_tag_name(mf, index), gather_global(mpi, gcell, values, n_cell)))
    return out


def _local_cell_tags(tags, mesh, n_local):
    """(name, values) per MeshFunction, each rank keeping only its own cells.

    `n_local` is the number of rows this rank contributed to the topology, which is
    what the values are written alongside. They must agree: a mesh carrying ghost
    cells would give the MeshFunction more entries than the topology has rows, and
    the slab would be written past its own block.
    """
    if not tags:
        return []
    _check_cell_tags(tags, mesh)
    out = []
    for index, mf in enumerate(tags):
        values = np.asarray(mf.array()).reshape(-1, 1)
        if len(values) != n_local:
            raise ValueError(
                'the MeshFunction has {} local values against {} local cells in the '
                'topology; this writes a mesh without ghost cells'
                .format(len(values), n_local))
        out.append((_tag_name(mf, index), values))
    return out


def _check_functions(functions):
    """What this can write, asked once for both the gathered and the slab path.

    It used to be asked only inside `_function_nodes`, which the parallel path does
    not go through -- so on more than one rank a DG function was quietly interpolated
    onto a Lagrange space and written as though it had been continuous all along, and
    a P2 field on a triangle mesh died with an IndexError out of `conn[:, XDMF_ORDER]`
    instead of being told it is not a tetrahedron.
    """
    spaces = [f.function_space() for f in functions]
    mesh = spaces[0].mesh()
    for V in spaces[1:]:
        if V.mesh().id() != mesh.id():
            raise ValueError('the functions are on different meshes')
    for V in spaces:
        element = V.ufl_element()
        if element.family() not in ('Lagrange', 'P', 'CG'):
            raise ValueError('{} is not a Lagrange element; use dolfin.XDMFFile'
                             .format(element.family()))
        if element.degree() not in (1, 2):
            raise ValueError('degree {} is not something this writes; use '
                             'dolfin.XDMFFile'.format(element.degree()))
    degree = max(V.ufl_element().degree() for V in spaces)
    cellname = mesh.ufl_cell().cellname()
    if cellname not in ('interval', 'triangle', 'tetrahedron'):
        raise ValueError('{} cells are not something this writes'.format(cellname))
    if degree == 2 and cellname != 'tetrahedron':
        # the midside numbering below is the tetrahedron's; degree 1 has no
        # midsides and works on any simplex, and refusing it was a regression --
        # a P1 field on a triangle mesh is a format dolfin writes and so must this
        raise ValueError('quadratic functions are written only on tetrahedra, '
                         'not {}; use dolfin.XDMFFile (which drops the midsides)'
                         .format(cellname))
    widths = [int(np.prod(f.value_shape())) if len(f.value_shape()) else 1
              for f in functions]
    return mesh, degree, widths


def _entity_dofs(dofmap, mesh, dim):
    """`dofmap.entity_dofs` as an integer array, empty or not.

    It hands back a Python list, and `np.asarray([])` is float64 -- which cannot
    index anything, so a rank owning no entities of this dimension died with
    "arrays used as indices must be of integer type" while every other rank sat in
    the collective HDF5 open waiting for it. Nothing dolfin partitions leaves a rank
    empty, so this only shows up on a mesh built by hand -- the no-slip surface of a
    flow, whose facets can miss a rank entirely -- and it shows up as a hang.
    """
    return np.asarray(dofmap.entity_dofs(mesh, dim), dtype=np.int64)


def _function_nodes(functions):
    """The shared node set, connectivity and gathered values of several Functions.

    The nodes are the mesh's vertices, then (at degree 2) one per edge -- the same
    numbering a reader of the stored dofmap builds, so the class and the standalone route write
    the same file for the same field. Everything comes off the mesh topology and
    `dofmap.entity_dofs`, both of which are whole arrays. The obvious implementation
    calls `dofmap.cell_dofs(c)` once per cell and builds a scalar space to interpolate
    on to; between them those were 96% of the cost of writing a P1 field, which is a
    format `dolfin.XDMFFile` writes too and was four times quicker at.
    """
    import dolfin as df
    mesh, degree, widths = _check_functions(functions)
    tdim = mesh.topology().dim()
    mpi = mesh.mpi_comm()
    root = df.MPI.rank(mpi) == 0
    gvert = np.asarray(mesh.topology().global_indices(0))
    gcell = np.asarray(mesh.topology().global_indices(tdim))
    n_vert = mesh.num_entities_global(0)
    n_cell = mesh.num_entities_global(tdim)

    coords = gather_global(mpi, gvert, _pad_coords(mesh.coordinates()), n_vert)
    conn = gather_global(mpi, gcell, gvert[np.asarray(mesh.cells())], n_cell)
    gedge, n_edge = None, 0
    if degree == 2:
        mesh.init(1)
        mesh.init(tdim, 1)
        mesh.init(1, 0)
        mesh.init_global(1)
        gedge = np.asarray(mesh.topology().global_indices(1))
        n_edge = mesh.num_entities_global(1)
        vertices = _pad_coords(mesh.coordinates())
        ends = _connectivity(mesh, 1)
        # dolfin numbers a cell's six local edges by the vertex pairs DOLFIN_EDGES
        # names -- checked, not assumed, and _check_midsides catches it if it moves
        cell_edges = _connectivity(mesh, tdim, 1)
        mid = gather_global(mpi, gedge,
                        0.5 * (vertices[ends[:, 0]] + vertices[ends[:, 1]]), n_edge)
        edge_conn = gather_global(mpi, gcell, gedge[cell_edges], n_cell)
        if root:
            coords = np.vstack((coords, mid))
            conn = np.hstack((conn, edge_conn + n_vert))[:, XDMF_ORDER]

    gathered, names = [], []
    for index, (func, width) in enumerate(zip(functions, widths)):
        source = func if _is_plain(func.function_space(), mesh, degree, width) \
            else df.interpolate(func, _plain_space(mesh, degree, width))
        V = source.function_space()
        to_global = np.asarray(V.dofmap().tabulate_local_to_global_dofs())
        # a sub-dofmap's entity dofs are indices into the *parent* vector, which is
        # what makes reading the components a slice rather than three deep copies
        dofmaps = [V.sub(c).dofmap() for c in range(width)] if width > 1 \
            else [V.dofmap()]
        vector = source.vector().gather_on_zero()
        columns = []
        for dofmap in dofmaps:
            take = gather_global(mpi, gvert,
                             to_global[_entity_dofs(dofmap, mesh, 0)],
                             n_vert)
            if degree == 2:
                edge_take = gather_global(
                    mpi, gedge, to_global[_entity_dofs(dofmap, mesh, 1)],
                    n_edge)
                if root:
                    take = np.concatenate((take, edge_take))
            if root:
                columns.append(vector[take])
        if root:
            gathered.append(np.column_stack(columns) if len(columns) > 1
                            else columns[0])
        names.append(func.name() if func.name() else 'f{}'.format(index))

    if not root:
        return None, None, None, None, degree, mesh
    if degree == 2:
        _check_midsides(coords, conn)
    return coords, conn, gathered, names, degree, mesh


def owned_entities(mesh, dim):
    """Which of this rank's entities of `dim` it is the one responsible for writing.

    Where there is a dofmap to ask, ownership comes from that (see
    `_function_nodes_distributed`). A Mesh or a MeshFunction has no dofs, so the rule
    is the lowest rank among those sharing the entity, read off
    `mesh.topology().shared_entities(dim)`. It needs no communication -- each rank
    already knows who else holds the entity -- and it partitions exactly: measured on
    a 6x6x6 cube over three ranks, 143 + 120 + 80 = 343 vertices and
    673 + 640 + 541 = 1854 edges, against those global counts. `_redistribute` checks
    that claim on every write rather than trusting it.
    """
    import dolfin as df
    rank = df.MPI.rank(mesh.mpi_comm())
    shared = mesh.topology().shared_entities(dim)
    owned = np.ones(mesh.num_entities(dim), dtype=bool)
    for local, others in shared.items():
        if others and min(others) < rank:
            owned[local] = False
    return owned


def _collective_write(node, lo, values):
    """One rank's part of a collective dataset write, empty parts included.

    Every rank must take part in a collective write -- but h5py's `__setitem__`
    returns without calling H5Dwrite when the selection is empty, so a rank with
    nothing to write silently skips the collective the other ranks are inside.
    Rank zero then spins in `H5D__write`'s Allreduce while the rest wait in the
    file-close barrier, all at full CPU, forever -- which is exactly how the
    parallel points write deadlocked (rank zero holds all the points, the other
    ranks hold none). The empty rank instead makes the low-level write with a
    none-selection, which participates in the collective and writes nothing.
    """
    if len(values):
        with node.collective:
            node[lo:lo + len(values)] = values
        return
    import h5py
    fspace = node.id.get_space()
    fspace.select_none()
    mspace = h5py.h5s.create_simple((0,))
    dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
    dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
    node.id.write(mspace, fspace, np.zeros(0, dtype=node.dtype), dxpl=dxpl)


def _exscan(comm, value):
    """This rank's offset into a rank-major concatenation."""
    offset = comm.exscan(int(value))
    return 0 if offset is None else int(offset)


def _function_nodes_distributed(functions):
    """The same node set as `_function_nodes`, but left where it already is.

    Every rank returns only the rows it owns, so nothing is gathered. Node ids are the
    global entity ids -- vertex index, then `n_vert +` edge index -- exactly as in
    serial, which is what keeps the file identical however many ranks wrote it.

    Ownership is taken from the **dofmap**, not from `shared_entities`: a dof belongs
    to exactly one rank, its ownership range is contiguous, and the rank that owns it
    is the one holding the value in `get_local()`. Measured on a 6x6x6 cube over three
    ranks, owned vertices plus owned edges came to 2197 against 343 + 1854 global --
    an exact partition, which is what `_redistribute` insists on.
    """
    import dolfin as df
    mesh, degree, widths = _check_functions(functions)
    tdim = mesh.topology().dim()
    comm = mesh.mpi_comm()

    gvert = np.asarray(mesh.topology().global_indices(0))
    n_vert = mesh.num_entities_global(0)
    n_cell = mesh.num_entities_global(tdim)
    vertices = _pad_coords(mesh.coordinates())
    conn = gvert[np.asarray(mesh.cells())]
    n_nodes, gedge, mid = n_vert, None, None
    if degree == 2:
        mesh.init(1)
        mesh.init(tdim, 1)
        mesh.init(1, 0)
        mesh.init_global(1)
        gedge = np.asarray(mesh.topology().global_indices(1))
        n_nodes = n_vert + mesh.num_entities_global(1)
        ends = _connectivity(mesh, 1)
        mid = 0.5 * (vertices[ends[:, 0]] + vertices[ends[:, 1]])
        conn = np.hstack((conn, gedge[_connectivity(mesh, tdim, 1)] + n_vert))
        conn = conn[:, XDMF_ORDER]
    cell_offset = _exscan(comm, len(conn))

    fields = []
    coord_rows = coords = None
    for index, (func, width) in enumerate(zip(functions, widths)):
        source = func if _is_plain(func.function_space(), mesh, degree, width) \
            else df.interpolate(func, _plain_space(mesh, degree, width))
        V = source.function_space()
        to_global = np.asarray(V.dofmap().tabulate_local_to_global_dofs())
        lo, hi = V.dofmap().ownership_range()
        dofmaps = [V.sub(c).dofmap() for c in range(width)] if width > 1 \
            else [V.dofmap()]
        local = source.vector().get_local()

        rows, columns, positions = [], [[] for _ in dofmaps], []
        for dim, ids, points in ((0, gvert, vertices), (1, gedge, mid)):
            if dim == 1 and degree != 2:
                continue
            per_entity = [to_global[_entity_dofs(d, mesh, dim)]
                          for d in dofmaps]
            owned = (per_entity[0] >= lo) & (per_entity[0] < hi)
            # a blocked dofmap keeps a node's components together, so they share an
            # owner; if that ever stopped holding, the components of a node would be
            # written by different ranks and one of them would be missing its value
            for other in per_entity[1:]:
                if not np.array_equal((other >= lo) & (other < hi), owned):
                    raise ValueError('the components of a node are owned by '
                                     'different ranks; this cannot write that')
            rows.append(ids[owned] + (0 if dim == 0 else n_vert))
            positions.append(points[owned])
            for c, dofs in enumerate(per_entity):
                columns[c].append(local[dofs[owned] - lo])

        field_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        values = [np.concatenate(c) if c else np.zeros(0) for c in columns]
        fields.append((func.name() if func.name() else 'f{}'.format(index),
                       field_rows,
                       np.column_stack(values) if len(values) > 1 else values[0]))
        if coords is None:
            coord_rows = field_rows
            coords = np.vstack(positions) if positions else np.zeros((0, 3))

    return {'n_nodes': n_nodes, 'n_cells': n_cell, 'degree': degree, 'mesh': mesh,
            'coord_rows': coord_rows, 'coords': coords,
            'conn': conn, 'cell_offset': cell_offset, 'fields': fields}


def _scatter_nodes(target, space, coords, values, width, degree):
    """Puts nodal values read from a file onto an unconstrained space's dofs.

    The space is unconstrained, so its dof coordinates are distinct and a nearest
    match is unambiguous -- which is exactly what is *not* true of a periodic space,
    and why the caller interpolates from here rather than writing into one directly.
    """
    from scipy.spatial import cKDTree
    # `dofmap().dofs()` numbers dofs **globally**, while `tabulate_dof_coordinates`
    # and `get_local` hold this rank's owned rows -- so the global numbers have to be
    # brought back to local ones, or rank one indexes a 180-row array at 342.
    first = space.dofmap().ownership_range()[0]
    if width > 1:
        sub_dofs = [np.asarray(space.sub(c).dofmap().dofs()) - first
                    for c in range(width)]
        x = space.tabulate_dof_coordinates().reshape(-1, 3)[sub_dofs[0]]
    else:
        sub_dofs = [np.asarray(space.dofmap().dofs()) - first]
        x = space.tabulate_dof_coordinates().reshape(-1, 3)
    tree = cKDTree(coords)
    distance, index = tree.query(x)
    scale = max(np.ptp(coords, axis=0).max(), 1.0)
    if distance.max() > 1e-9 * scale:
        raise ValueError(
            'a dof sits {:.3e} from the nearest node in the file, so the mesh is '
            'not the one the field was written on'.format(distance.max()))
    vector = target.vector()
    local = vector.get_local()
    values = np.asarray(values)
    for c, dofs in enumerate(sub_dofs):
        column = values[:, c] if values.ndim > 1 else values
        local[dofs] = column[index]
    vector.set_local(local)
    vector.apply('insert')


def _check_midsides(coords, conn, sample=200):
    """Every midside node must lie on the edge it is named for.

    The node order is a convention -- dolfin's differs from XDMF's -- and getting it
    wrong writes cells that look plausible and are wrong. This is cheap and catches
    it, including after a distributed gather, where the numbering has been rebuilt.
    """
    rng = np.random.RandomState(0)
    pick = conn[rng.choice(len(conn), min(sample, len(conn)), replace=False)]
    for k, (a, b) in enumerate(XDMF_EDGES):
        off = np.abs(coords[pick[:, 4 + k]]
                     - 0.5 * (coords[pick[:, a]] + coords[pick[:, b]])).max()
        edge = np.linalg.norm(coords[pick[:, a]] - coords[pick[:, b]], axis=1).max()
        if off > 1e-9 * edge:
            raise ValueError(
                'midside node {} sits {:.2e} off the edge it belongs to, so the '
                'node ordering is wrong'.format(4 + k, off))


# ---------------------------------------------------------------------------
# reading an XDMF back, with no dolfin
# ---------------------------------------------------------------------------

def _scan(path):
    """Every uniform grid in a file: its topology type and where its arrays are."""
    tree = ET.parse(path)
    grids = []
    for grid in tree.getroot().iter('Grid'):
        if grid.get('GridType') == 'Collection':
            continue
        topology = grid.find('Topology')
        geometry = grid.find('Geometry')
        if topology is None or geometry is None:
            continue
        attributes, centers = {}, {}
        for attribute in grid.findall('Attribute'):
            item = attribute.find('DataItem')
            attributes[attribute.get('Name')] = _item(item)
            centers[attribute.get('Name')] = attribute.get('Center')
        time = grid.find('Time')
        grids.append({
            'topology': topology.get('TopologyType'),
            'topology_item': _item(topology.find('DataItem')),
            'geometry_item': _item(geometry.find('DataItem')),
            'attributes': attributes,
            'centers': centers,
            'time': float(time.get('Value')) if time is not None else None,
        })
    return grids


def _item(node):
    """One DataItem as (format, reference, shape)."""
    if node is None:
        return None
    shape = tuple(int(n) for n in node.get('Dimensions').split())
    return (node.get('Format'), (node.text or '').strip(), shape)


def _load(path, item):
    """The array a DataItem names, from the sidecar HDF5 or from the XML itself."""
    fmt, ref, shape = item
    if fmt == 'HDF':
        import h5py
        filename, _, group = ref.partition(':')
        full = os.path.join(os.path.dirname(os.path.abspath(path)), filename)
        with h5py.File(full, 'r') as handle:
            return np.asarray(handle[group.lstrip('/')])
    return np.array(ref.split(), dtype=float).reshape(shape)


def _read_grid_geometry(path, grid):
    coords = _load(path, grid['geometry_item'])
    conn = _load(path, grid['topology_item']).astype(np.int64)
    return coords.reshape(-1, 3), conn.reshape(grid['topology_item'][2])


def _read_grid_attribute(path, grid, name):
    values = _load(path, grid['attributes'][name])
    return values.reshape(grid['attributes'][name][2])


def write_quadratic(path, coords, conn, values, names, cell_type='tetra10'):
    """Writes one XDMF holding the given cells and nodal fields.

    Goes through the same writer `XDMFFile2` uses, so the standalone route and the
    solver produce the same file layout for the same data rather than two dialects.
    """
    doc = XdmfDocument(path)
    grid = doc.grid('mesh')
    topology_type = MESHIO_TOPOLOGY.get(cell_type, cell_type)
    doc.topology(grid, conn, topology_type,
                 ref=doc.dataset('Mesh/mesh/topology',
                                 _int32(np.asarray(conn))))
    doc.geometry(grid, coords,
                 ref=doc.dataset('Mesh/mesh/geometry', np.asarray(coords, float)))
    for index, (name, value) in enumerate(zip(names, values)):
        value = _pad3(value)
        if np.asarray(value).ndim == 1:
            value = np.asarray(value).reshape(-1, 1)
        doc.attribute(grid, name, value, 'Node',
                      ref=doc.dataset('VisualisationVector/{}'.format(index), value))
    doc.close()


def write_quadratic_blocks(path, comm, cell_rows, conn, n_cells, node_rows, coords,
                           values, names, n_nodes, cell_type='tetra10'):
    """`write_quadratic` from ranks' shares, no rank holding the mesh: each dataset by
    `hdf5_rows.write_ordered` (the ranks' rows to the ranks whose blocks hold them, rank
    zero writing the compressed bytes -- serial h5py is enough), then rank zero writes the
    XML over them. `cell_rows`, `node_rows`: this rank's global rows of its cells and of
    the nodes it owns. The same file `write_quadratic` writes. Collective."""
    from femtools.hdf5_rows import as_rows, write_ordered
    h5_name = os.path.splitext(os.path.basename(path))[0] + '.h5'
    h5_path = os.path.join(os.path.dirname(os.path.abspath(path)), h5_name)
    zero = {'partition': np.array([0], dtype=np.uint64)}
    write_ordered(h5_path, 'Mesh/mesh/topology', cell_rows, _int32(conn),
                  n_cells, comm, mode='w', attrs=zero)
    write_ordered(h5_path, 'Mesh/mesh/geometry', node_rows, np.asarray(coords, float),
                  n_nodes, comm, attrs=zero)
    widths = []
    for index, value in enumerate(values):
        value = np.asarray(_pad3(value), float)
        value = as_rows(value)
        widths.append(comm.allreduce(value.shape[1] if len(value) else 0, op=_mpi_max()))
        write_ordered(h5_path, 'VisualisationVector/{}'.format(index), node_rows, value,
                      n_nodes, comm, attrs=zero)
    if comm.Get_rank() == 0:
        doc = XdmfDocument(path)
        grid = doc.grid('mesh')
        doc.topology(grid, None, MESHIO_TOPOLOGY.get(cell_type, cell_type),
                     ref='{}:/Mesh/mesh/topology'.format(h5_name), shape=(n_cells, 10))
        doc.geometry(grid, None, ref='{}:/Mesh/mesh/geometry'.format(h5_name),
                     shape=(n_nodes, 3))
        for index, (name, width) in enumerate(zip(names, widths)):
            doc.attribute(grid, name, np.zeros((0, width)), 'Node',
                          ref='{}:/VisualisationVector/{}'.format(h5_name, index),
                          shape=(n_nodes, width))
        doc.close()
    comm.Barrier()


def _mpi_max():
    from mpi4py import MPI
    return MPI.MAX

