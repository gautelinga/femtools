#!/usr/bin/env python3
"""What `femtools/xdmf.py` promises, checked against dolfin itself.

    python3 -m pytest tests/test_xdmf.py

Needs legacy dolfin. About a
minute, most of it FFC compiling the elements the first time.

`XDMFFile2` claims to be a drop-in for `dolfin.XDMFFile` that keeps the quadratic
part. Both halves of that need testing and they pull in opposite directions -- the
drop-in half says "behave exactly like dolfin", the quadratic half says "do not do
the one thing dolfin does here". So each case below says which of the two it is
about, and the four places the two genuinely disagree are tested as *differences*,
against dolfin's own behaviour measured in the same process.

The file the writer produces is read back three ways where it can be: by dolfin, by
this module's own reader, and by meshio, which knows nothing about either and is
therefore the one that says the XML is really XDMF rather than merely self-consistent.
"""


import pytest

pytest.importorskip("dolfin")   # collected only under the interpreter that has it
pytestmark = [pytest.mark.dolfin]

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


import dolfin as df                                            
from femtools import xdmf as export_p2_xdmf
from femtools.xdmf import (                             
    XDMFFile2, XDMFParameters, XDMF_EDGES, XDMF_ORDER, QUADRATIC,
    _scan, _read_grid_geometry, _read_grid_attribute)
from tests import support

df.set_log_level(40)
TMP = None


def tmp(filename):
    return os.path.join(TMP, filename)


# the two cells a quadratic field is written on, for the cases that hold on both
CELLS = pytest.mark.parametrize('cell', ['tetrahedron', 'triangle'])
# XDMF's name for each, and meshio's
QUADRATIC_TOPOLOGY = {'tetrahedron': 'Tetrahedron_10', 'triangle': 'Triangle_6'}
MESHIO_QUADRATIC = {'tetrahedron': 'tetra10', 'triangle': 'triangle6'}


def unit_mesh(n=2, cell='tetrahedron'):
    if cell == 'triangle':
        return df.UnitSquareMesh(n, n)
    return df.UnitCubeMesh(n, n, n)


QUADRATIC_EXPRESSION = {3: 'x[0]*x[0] + 2*x[1]*x[1] - x[2]*x[2]',
                        2: 'x[0]*x[0] + 2*x[1]*x[1] - 3*x[0]*x[1]'}


def quadratic_exact(coords, gdim=3):
    """`quadratic_field`'s values at the rows of `coords`, as the file holds them."""
    x, y = coords[:, 0], coords[:, 1]
    if gdim == 2:
        return x * x + 2 * y * y - 3 * x * y
    return x * x + 2 * y * y - coords[:, 2] * coords[:, 2]


def quadratic_field(mesh, name='u'):
    """A genuinely quadratic scalar: its midsides are not the corner mean."""
    V = df.FunctionSpace(mesh, 'CG', 2)
    u = df.Function(V)
    u.rename(name, name)
    u.interpolate(df.Expression(QUADRATIC_EXPRESSION[mesh.geometry().dim()],
                                degree=2))
    return u


def vector_field(mesh, degree=2, name='u'):
    """A vector whose first component is x[0]**2 in 2D and in 3D."""
    V = df.VectorFunctionSpace(mesh, 'CG', degree)
    u = df.Function(V)
    u.rename(name, name)
    components = (('x[0]*x[0]', 'x[1]*x[2]', 'x[2]*x[2]')
                  if mesh.geometry().dim() == 3 else ('x[0]*x[0]', 'x[0]*x[1]'))
    u.interpolate(df.Expression(components, degree=degree))
    return u


def assert_close(a, b, tol, what):
    diff = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    if not diff <= tol:
        raise AssertionError('{}: differ by {:.3e}, over {:.3e}'.format(
            what, diff, tol))
    return diff


# -- the quadratic half: what dolfin gets wrong ----------------------------------

@CELLS
def test_quadratic_survives(cell):
    """a P2 field keeps its midside nodes"""
    mesh = unit_mesh(cell=cell)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('q.xdmf')) as f:
        f.write(u)
    grids = _scan(tmp('q.xdmf'))
    assert len(grids) == 1, grids
    assert grids[0]['topology'] == QUADRATIC_TOPOLOGY[cell], grids[0]['topology']
    coords, conn = _read_grid_geometry(tmp('q.xdmf'), grids[0])
    values = _read_grid_attribute(tmp('q.xdmf'), grids[0], 'u').ravel()
    corners, edges, _ = QUADRATIC[cell]
    assert conn.shape[1] == corners + len(edges), conn.shape
    # the mesh's own coordinates: two for a 2D mesh (XY, as dolfin writes it)
    assert coords.shape[1] == mesh.geometry().dim(), coords.shape

    # the field really is quadratic: midsides depart from the corner mean
    gap = max(abs(values[c[corners + k]] - 0.5 * (values[c[i]] + values[c[j]]))
              for c in conn for k, (i, j) in enumerate(edges))
    assert gap > 1e-3, 'the test field is not quadratic enough to prove anything'

    # and every node's value is the exact analytic one
    exact = quadratic_exact(coords, mesh.geometry().dim())
    assert_close(values, exact, 1e-12, 'nodal values')
    print('midsides depart from the corner mean by {:.3f}'.format(gap))


def test_dolfin_loses_it():
    """dolfin loses exactly what this keeps

The difference is measured, not asserted from the documentation."""
    mesh = unit_mesh()
    u = quadratic_field(mesh)
    with df.XDMFFile(df.MPI.comm_world, tmp('lin.xdmf')) as f:
        f.write(u)
    grids = _scan(tmp('lin.xdmf'))
    assert grids[0]['topology'] == 'Tetrahedron', grids[0]['topology']
    coords, conn = _read_grid_geometry(tmp('lin.xdmf'), grids[0])
    assert conn.shape[1] == 4, 'dolfin wrote something other than linear tets'
    n_dolfin = len(coords)

    with XDMFFile2(df.MPI.comm_world, tmp('quad.xdmf')) as f:
        f.write(u)
    ours, _ = _read_grid_geometry(tmp('quad.xdmf'), _scan(tmp('quad.xdmf'))[0])
    assert len(ours) > n_dolfin, (len(ours), n_dolfin)
    print('dolfin {} nodes, this {}'.format(n_dolfin, len(ours)))


@CELLS
def test_node_order(cell):
    """the node order is XDMF's, not dolfin's"""
    mesh = unit_mesh(cell=cell)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('order.xdmf')) as f:
        f.write(u)
    coords, conn = _read_grid_geometry(tmp('order.xdmf'), _scan(tmp('order.xdmf'))[0])
    corners, edges, order = QUADRATIC[cell]
    for k, (a, b) in enumerate(edges):
        mid = coords[conn[:, corners + k]]
        want = 0.5 * (coords[conn[:, a]] + coords[conn[:, b]])
        assert_close(mid, want, 1e-12, 'midside slot {}'.format(corners + k))
    # and the two conventions really do differ, so the permutation is not a no-op
    assert order != list(range(len(order))), 'the permutation is the identity'
    print('all {} midsides sit on their edges; permutation {}'.format(len(edges), order))


def test_node_conventions():
    """both node orders are the ones their owners use

XDMF's stated from its own documentation, dolfin's measured.

    `test_node_order` checks the file against the edge lists in the module, so a
    wrong list would pass it. Here XDMF's lists are spelled out independently --
    Triangle_6 is three corners then (0,1), (1,2), (2,0); Tetrahedron_10 is four
    corners then (0,1), (1,2), (2,0), (0,3), (1,3), (2,3), as in VTK -- and dolfin's
    are read off where the writer takes them from: the P2 dof coordinates of a cell,
    and the mesh's cell-to-edge connectivity, which must agree with each other.
    """
    stated = {'triangle': [(0, 1), (1, 2), (2, 0)],
              'tetrahedron': [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]}
    for cell, xdmf_edges in stated.items():
        corners, edges, order = QUADRATIC[cell]
        assert [tuple(sorted(e)) for e in xdmf_edges] == list(edges), (cell, edges)
        mesh = unit_mesh(2, cell)
        tdim = mesh.topology().dim()
        mesh.init(tdim, 1)
        mesh.init(1, 0)
        V = df.FunctionSpace(mesh, 'CG', 2)
        x = V.tabulate_dof_coordinates()
        vertices = mesh.coordinates()
        for c in df.cells(mesh):
            v = c.entities(0)
            dofs = x[V.dofmap().cell_dofs(c.index())]
            # dolfin's dofs, reordered into XDMF's slots, are corners then midsides
            # of exactly the stated edges
            ours = dofs[order]
            assert_close(ours[:corners], vertices[v], 1e-14, 'corners')
            for k, (a, b) in enumerate(xdmf_edges):
                assert_close(ours[corners + k], 0.5 * (vertices[v[a]] + vertices[v[b]]),
                             1e-14, '{} slot {}'.format(cell, corners + k))
            # and the mesh's local edge i is the one holding dolfin's dof corners + i
            for i, e in enumerate(c.entities(1)):
                ends = vertices[df.Edge(mesh, e).entities(0)]
                assert_close(dofs[corners + i], ends.mean(axis=0), 1e-14,
                             '{} local edge {}'.format(cell, i))
    print('Triangle_6 and Tetrahedron_10 slots match dolfin\'s dofs and edges')


@CELLS
def test_several_fields(cell):
    """several fields share one node set"""
    mesh = unit_mesh(cell=cell)
    u = vector_field(mesh)
    p = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    p.rename('p', 'p')
    p.interpolate(df.Expression('x[0] + x[1]', degree=1))
    with XDMFFile2(df.MPI.comm_world, tmp('up.xdmf')) as f:
        f.write((u, p))
    grids = _scan(tmp('up.xdmf'))
    assert len(grids) == 1, 'both fields must share one grid, got {}'.format(len(grids))
    assert set(grids[0]['attributes']) == {'u', 'p'}, grids[0]['attributes']
    coords, conn = _read_grid_geometry(tmp('up.xdmf'), grids[0])
    assert conn.shape[1] == {'tetrahedron': 10, 'triangle': 6}[cell], \
        'the P1 pressure must be carried onto P2 nodes'
    pv = _read_grid_attribute(tmp('up.xdmf'), grids[0], 'p').ravel()
    assert_close(pv, coords[:, 0] + coords[:, 1], 1e-12, 'pressure on P2 nodes')
    uv = _read_grid_attribute(tmp('up.xdmf'), grids[0], 'u')
    # a 2D vector is padded to XDMF's three components, as dolfin pads it
    assert uv.shape[1] == 3, uv.shape
    assert_close(uv[:, 0], coords[:, 0] ** 2, 1e-12, 'u0')
    if cell == 'triangle':
        assert_close(uv[:, 1], coords[:, 0] * coords[:, 1], 1e-12, 'u1')
        assert not uv[:, 2].any(), 'the padding component is not zero'
    print('u and p on {} shared nodes'.format(len(coords)))


# -- the drop-in half: parity with dolfin ---------------------------------------

def test_p1_matches_dolfin():
    """a P1 field matches dolfin node for node"""
    mesh = unit_mesh(3)
    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    u.rename('u', 'u')
    u.interpolate(df.Expression('x[0] + 3*x[1] - x[2]', degree=1))
    with df.XDMFFile(df.MPI.comm_world, tmp('p1_dolfin.xdmf')) as f:
        f.write(u)
    with XDMFFile2(df.MPI.comm_world, tmp('p1_ours.xdmf')) as f:
        f.write(u)
    a, b = _scan(tmp('p1_dolfin.xdmf'))[0], _scan(tmp('p1_ours.xdmf'))[0]
    assert a['topology'] == b['topology'] == 'Tetrahedron', (a, b)
    xa, ca = _read_grid_geometry(tmp('p1_dolfin.xdmf'), a)
    xb, cb = _read_grid_geometry(tmp('p1_ours.xdmf'), b)
    va = _read_grid_attribute(tmp('p1_dolfin.xdmf'), a, 'u').ravel()
    vb = _read_grid_attribute(tmp('p1_ours.xdmf'), b, 'u').ravel()
    assert_close(xa, xb, 0.0, 'coordinates')
    assert_close(ca, cb, 0.0, 'connectivity')
    assert_close(va, vb, 0.0, 'values')
    print('{} nodes, {} cells, bit-identical to dolfin'.format(len(xa), len(ca)))


def test_mesh_roundtrip():
    """a mesh round-trips through dolfin's own reader"""
    mesh = unit_mesh(3)
    with XDMFFile2(df.MPI.comm_world, tmp('mesh.xdmf')) as f:
        f.write(mesh)
    back = df.Mesh()
    with df.XDMFFile(df.MPI.comm_world, tmp('mesh.xdmf')) as f:
        f.read(back)
    assert back.num_cells() == mesh.num_cells(), (back.num_cells(), mesh.num_cells())
    assert back.num_vertices() == mesh.num_vertices()
    assert_close(np.sort(back.coordinates(), axis=0),
                 np.sort(mesh.coordinates(), axis=0), 1e-14, 'coordinates')
    assert_close(back.num_cells() * 0 + df.assemble(df.Constant(1) * df.dx(back)),
                 df.assemble(df.Constant(1) * df.dx(mesh)), 1e-14, 'volume')
    print('dolfin read back {} cells written here'.format(back.num_cells()))


def test_meshfunction_roundtrip():
    """a mesh function round-trips through dolfin's own reader"""
    mesh = unit_mesh(3)
    mf = df.MeshFunction('size_t', mesh, 2, 0)
    mf.array()[:] = np.arange(len(mf.array())) % 5
    mf.rename('markers', 'markers')
    with XDMFFile2(df.MPI.comm_world, tmp('mf.xdmf')) as f:
        f.write(mf)
    back = df.MeshFunction('size_t', mesh, 2, 0)
    with df.XDMFFile(df.MPI.comm_world, tmp('mf.xdmf')) as f:
        f.read(back, 'markers')
    assert_close(back.array(), mf.array(), 0, 'marker values')
    print('{} facet markers, read back by dolfin'.format(len(mf.array())))


def test_parameters():
    """parameters refuse a key dolfin does not have"""
    p = XDMFParameters()
    assert set(p.keys()) == {'flush_output', 'functions_share_mesh',
                             'rewrite_function_mesh'}, set(p.keys())
    assert p['rewrite_function_mesh'] is True
    assert p['flush_output'] is False
    p['flush_output'] = True
    assert p['flush_output'] is True
    try:
        p['flush_ouput'] = True                      # the typo that used to be silent
    except KeyError as exc:
        assert 'flush_output' in str(exc), str(exc)
    else:
        raise AssertionError('an unknown parameter was accepted')
    # and dolfin's own set has exactly these three
    with df.XDMFFile(df.MPI.comm_world, tmp('probe_par.xdmf')) as f:
        theirs = {k for k in ('flush_output', 'functions_share_mesh',
                              'rewrite_function_mesh') if k in f.parameters}
    assert theirs == set(p.keys()), theirs
    print('three keys, and a typo raises')


def test_flush_output():
    """flush_output writes as it goes"""
    mesh = unit_mesh()
    u = quadratic_field(mesh)
    f = XDMFFile2(df.MPI.comm_world, tmp('flush.xdmf'))
    f.parameters['flush_output'] = True
    f.write(u)
    assert os.path.exists(tmp('flush.xdmf')), 'nothing written before close'
    grids = _scan(tmp('flush.xdmf'))
    assert grids and grids[0]['attributes'], 'the flushed file has no field in it'
    f.close()

    g = XDMFFile2(df.MPI.comm_world, tmp('noflush.xdmf'))
    g.write(u)
    unflushed = os.path.exists(tmp('noflush.xdmf'))
    g.close()
    assert os.path.exists(tmp('noflush.xdmf')), 'close must write the file'
    print('flushed file readable mid-write; unflushed present before close: {}'.format(unflushed))


def test_variable_surface():
    """the name, label and id of a Variable"""
    f = XDMFFile2(df.MPI.comm_world, tmp('var.xdmf'))
    assert f.name() == ''
    f.rename('flow', 'the flow')
    assert f.name() == 'flow' and f.label() == 'the flow'
    assert isinstance(f.id(), int)
    f.close()
    # dolfin has the same four
    with df.XDMFFile(df.MPI.comm_world, tmp('var2.xdmf')) as d:
        for method in ('name', 'label', 'rename', 'id'):
            assert hasattr(d, method), method
    print('name/label/rename/id')


def test_refusals():
    """an unwritable object says so instead of doing nothing"""
    mesh = unit_mesh()
    with XDMFFile2(df.MPI.comm_world, tmp('bad.xdmf')) as f:
        try:
            f.write('not a function')
        except TypeError as exc:
            assert 'XDMFFile2 does not write' in str(exc), str(exc)
        else:
            raise AssertionError('a string was accepted')
    # a discontinuous element is not something this writes
    V = df.FunctionSpace(mesh, 'DG', 0)
    u = df.Function(V)
    u.rename('u', 'u')
    with XDMFFile2(df.MPI.comm_world, tmp('bad2.xdmf')) as f:
        try:
            f.write(u)
        except ValueError as exc:
            assert 'Lagrange' in str(exc), str(exc)
        else:
            raise AssertionError('a DG function was accepted')
    print('a string and a DG function both raise')


# -- the four deliberate differences ---------------------------------------------

def test_mixed_file():
    """a mesh function and a function share a file, where dolfin segfaults

dolfin takes the process down on this, so its half runs in a subprocess."""
    script = '''
import dolfin as df
df.set_log_level(40)
mesh = df.UnitCubeMesh(2,2,2)
u = df.Function(df.FunctionSpace(mesh,'CG',1)); u.rename('u','u')
mf = df.MeshFunction('size_t', mesh, 2, 0); mf.rename('mf','mf')
with df.XDMFFile(df.MPI.comm_world, %r) as f:
    f.write(mf)
    f.write(u)
print('survived')
''' % tmp('dolfin_mixed.xdmf')
    proc = subprocess.run([sys.executable, '-c', script],
                          capture_output=True, timeout=300)
    dolfin_died = proc.returncode != 0
    assert dolfin_died, ('dolfin no longer segfaults on MeshFunction-then-Function; '
                         'this test recorded a real defect and should be revisited')

    mesh = unit_mesh()
    u = quadratic_field(mesh)
    mf = df.MeshFunction('size_t', mesh, 2, 0)
    mf.array()[:] = np.arange(len(mf.array())) % 3
    mf.rename('mf', 'mf')
    with XDMFFile2(df.MPI.comm_world, tmp('mixed.xdmf')) as f:
        f.write(mf)
        f.write(u)
    grids = _scan(tmp('mixed.xdmf'))
    assert len(grids) == 2, [g['topology'] for g in grids]
    kinds = {g['topology'] for g in grids}
    assert kinds == {'Triangle', 'Tetrahedron_10'}, kinds
    names = {n for g in grids for n in g['attributes']}
    assert names == {'mf', 'u'}, names
    print('dolfin exits {}; this writes both grids'.format(-proc.returncode))


def test_cell_data_in_tuple():
    """a mesh function in the tuple is cell data on the same grid

The fourth difference: dolfin has no call for this at all.

    A MeshFunction over the cells is one value per cell, which is exactly a
    cell-centred attribute of the grid the functions are written on -- so it goes
    there, rather than onto a grid of its own with its own copy of the mesh. dolfin
    writes one object per call and rejects a tuple outright, which is measured here
    rather than taken from the documentation.
    """
    mesh = unit_mesh(3)
    u = vector_field(mesh, degree=1, name='tau')
    q = quadratic_field(mesh, name='p')
    tags = df.MeshFunction('size_t', mesh, mesh.topology().dim())
    tags.array()[:] = np.arange(mesh.num_cells()) % 4 + 5
    tags.rename('markers', 'markers')

    try:
        with df.XDMFFile(df.MPI.comm_world, tmp('dolfin_tuple.xdmf')) as f:
            f.write((u, q, tags))
        dolfin_took_it = True
    except Exception:
        dolfin_took_it = False
    assert not dolfin_took_it, ('dolfin.XDMFFile now accepts a tuple; this difference '
                                'should be revisited')

    with XDMFFile2(df.MPI.comm_world, tmp('celldata.xdmf')) as f:
        f.write((u, q, tags))
    grids = _scan(tmp('celldata.xdmf'))
    assert len(grids) == 1, [g['topology'] for g in grids]
    grid = grids[0]
    # the functions are degree 1 and 2, so the grid is the quadratic one and the
    # cell attribute rides on it unchanged: refining the node set does not change
    # how many cells there are
    assert grid['topology'] == 'Tetrahedron_10', grid['topology']
    assert grid['centers'] == {'tau': 'Node', 'p': 'Node', 'markers': 'Cell'}, \
        grid['centers']

    values = _read_grid_attribute(tmp('celldata.xdmf'), grid, 'markers').ravel()
    assert np.array_equal(values, np.asarray(tags.array())), 'the markers moved'
    assert len(values) == mesh.num_cells(), (len(values), mesh.num_cells())

    # and the outside reader agrees it is one mesh with both kinds of field on it,
    # which is the whole point: meshio refuses a file holding two grids
    import meshio
    m = meshio.read(tmp('celldata.xdmf'))
    got = np.asarray(m.cell_data_dict['markers']['tetra10']).ravel()
    assert np.array_equal(got, np.asarray(tags.array())), 'meshio reads other markers'
    assert set(m.point_data) == {'tau', 'p'}, set(m.point_data)
    print('one grid, {} nodes and {} tagged cells; dolfin refuses the call'.format(len(m.points), len(got)))


def test_cell_data_refusals():
    """cell data has to really be over the cells

A MeshFunction over the wrong entities would be plausible and wrong.

    Facets outnumber cells, so pairing a facet MeshFunction with the cells does not
    even fail on a length check for every mesh -- it writes numbers that belong to
    other entities, on a grid that looks perfectly well formed. Both this and a
    MeshFunction belonging to another mesh are refused by name.
    """
    mesh = unit_mesh(2)
    other = unit_mesh(3)
    u = vector_field(mesh, degree=1)
    caught = []
    for what, obj in (
            ('a facet MeshFunction', df.MeshFunction('size_t', mesh, 2)),
            ('another mesh\'s cells', df.MeshFunction('size_t', other, 3)),
            ('something that is neither', other)):
        try:
            with XDMFFile2(df.MPI.comm_world, tmp('bad_cell.xdmf')) as f:
                f.write((u, obj))
        except (ValueError, TypeError) as exc:
            caught.append((what, type(exc).__name__))
        else:
            raise AssertionError('{} was written as cell data'.format(what))
    print(', '.join(('{} -> {}'.format(w, e) for w, e in caught)))


# -- time series ------------------------------------------------------------------

@CELLS
def test_time_series(cell):
    """a time series holds every step"""
    mesh = unit_mesh(cell=cell)
    V = df.FunctionSpace(mesh, 'CG', 2)
    u = df.Function(V)
    u.rename('u', 'u')
    times = [0.0, 0.5, 1.0]
    with XDMFFile2(df.MPI.comm_world, tmp('series.xdmf')) as f:
        f.parameters['rewrite_function_mesh'] = False
        for t in times:
            u.interpolate(df.Expression('t*(x[0]*x[0] + x[1]*x[1])', t=t, degree=2))
            f.write(u, t)
    grids = _scan(tmp('series.xdmf'))
    assert len(grids) == len(times), len(grids)
    assert [g['time'] for g in grids] == times, [g['time'] for g in grids]
    for grid, t in zip(grids, times):
        coords, _ = _read_grid_geometry(tmp('series.xdmf'), grid)
        values = _read_grid_attribute(tmp('series.xdmf'), grid, 'u').ravel()
        want = t * (coords[:, 0] ** 2 + coords[:, 1] ** 2)
        assert_close(values, want, 1e-12, 'step t={}'.format(t))
    # the steps are inside one temporal collection, as dolfin writes them
    import xml.etree.ElementTree as ET
    root = ET.parse(tmp('series.xdmf')).getroot()
    collections = [g for g in root.iter('Grid')
                   if g.get('CollectionType') == 'Temporal']
    assert len(collections) == 1, len(collections)
    print('{} steps in one temporal collection, each exact'.format(len(times)))


@CELLS
def test_rewrite_function_mesh(cell):
    """rewrite_function_mesh stores the mesh once or per step"""
    mesh = unit_mesh(cell=cell)
    u = quadratic_field(mesh)
    sizes = {}
    for flag in (False, True):
        path = tmp('rw_{}.xdmf'.format(flag))
        with XDMFFile2(df.MPI.comm_world, path) as f:
            f.parameters['rewrite_function_mesh'] = flag
            for t in (0.0, 1.0, 2.0):
                f.write(u, t)
        grids = _scan(path)
        assert len(grids) == 3, len(grids)
        refs = {g['geometry_item'][1] for g in grids}
        sizes[flag] = os.path.getsize(path.replace('.xdmf', '.h5'))
        if flag:
            assert len(refs) == 3, 'each step should carry its own mesh: {}'.format(refs)
        else:
            assert len(refs) == 1, 'every step should point at one mesh: {}'.format(refs)
    assert sizes[True] > sizes[False], sizes
    print('shared {} B, rewritten {} B'.format(sizes[False], sizes[True]))


@CELLS
def test_functions_share_mesh(cell):
    """functions_share_mesh puts the next field on the same grid"""
    mesh = unit_mesh(cell=cell)
    u = quadratic_field(mesh, 'u')
    v = quadratic_field(mesh, 'v')
    for flag, expect in ((True, 1), (False, 2)):
        path = tmp('share_{}.xdmf'.format(flag))
        with XDMFFile2(df.MPI.comm_world, path) as f:
            f.parameters['functions_share_mesh'] = flag
            f.write(u)
            f.write(v)
        grids = _scan(path)
        assert len(grids) == expect, (flag, len(grids))
        if flag:
            assert set(grids[0]['attributes']) == {'u', 'v'}, grids[0]['attributes']
    print('shared -> 1 grid, separate -> 2')


def periodic_x():
    """x[0] = 1 mapped onto x[0] = 0, in any dimension."""
    class Periodic(df.SubDomain):
        def inside(self, x, on_boundary):
            return bool(df.near(x[0], 0.0) and on_boundary)

        def map(self, x, y):
            y[0] = x[0] - 1.0
            for i in range(1, len(x)):
                y[i] = x[i]
    return Periodic()


# periodic in x[0], and quadratic in nothing: the P2 nodes hold it exactly only as
# point values, which is what is compared
PERIODIC_VELOCITY = ('cos(2*pi*x[0])*x[1] + t', 'sin(2*pi*x[0])*x[1]*x[1]')


def test_periodic_series_on_triangles():
    """a periodic 2D velocity, written step by step, is the whole mesh's nodes

The way twoasis writes its series, and the way it reads them back.

    One file per field, `rewrite_function_mesh` off and `flush_output` on, one
    `write(f, t)` a step. twoasis reads them by parsing the XDMF (`xdmf.grids`) and
    matching its own dof coordinates to the stored nodes, so the geometry must be
    every vertex and every edge midpoint of the whole mesh -- both sides of the
    period, which the periodic space identifies -- and every value the field's own
    value at its node.
    """
    import h5py
    from scipy.spatial import cKDTree
    from femtools.xdmf import grids as xdmf_grids
    mesh = unit_mesh(4, 'triangle')
    V = df.VectorFunctionSpace(mesh, 'CG', 2, constrained_domain=periodic_x())
    P = df.FunctionSpace(mesh, 'CG', 1, constrained_domain=periodic_x())
    assert V.dim() < df.VectorFunctionSpace(mesh, 'CG', 2).dim(), 'not periodic'
    u, p = df.Function(V), df.Function(P)
    u.rename('u', 'u')
    p.rename('p', 'p')
    files = {name: XDMFFile2(df.MPI.comm_world, tmp('tri_{}.xdmf'.format(name)))
             for name in 'up'}
    for f in files.values():
        f.parameters['rewrite_function_mesh'] = False
        f.parameters['flush_output'] = True
    times, saved = [0.0, 0.25, 0.5], []
    for t in times:
        u.interpolate(df.Expression(PERIODIC_VELOCITY, t=t, degree=4))
        p.interpolate(df.Expression('t*cos(2*pi*x[0]) + x[1]', t=t, degree=2))
        files['u'].write(u, t)
        files['p'].write(p, t)
        saved.append(u.vector().get_local().copy())
        # flushed as it goes: the XML names every step so far, before any close
        assert len(xdmf_grids(tmp('tri_u.xdmf'))) == len(saved)
    for f in files.values():
        f.close()

    steps = xdmf_grids(tmp('tri_u.xdmf'))
    assert [g['time'] for g in steps] == times, [g['time'] for g in steps]
    assert len({g['geometry'] for g in steps}) == 1, 'the mesh was stored per step'
    assert _scan(tmp('tri_u.xdmf'))[0]['topology'] == 'Triangle_6'
    assert _scan(tmp('tri_p.xdmf'))[0]['topology'] == 'Triangle'
    with h5py.File(steps[0]['geometry'][0], 'r') as h:
        nodes = h[steps[0]['geometry'][1]][:]
    assert nodes.shape == (mesh.num_vertices() + mesh.num_edges(), 2), nodes.shape
    mesh.init(1, 0)
    want = np.vstack((mesh.coordinates(),
                      [e.midpoint().array()[:2] for e in df.edges(mesh)]))
    for a, b, what in ((nodes[:, :2], want, 'a node is no vertex or midpoint'),
                       (want, nodes[:, :2], 'a vertex or midpoint is missing')):
        distance, _ = cKDTree(a).query(b)
        assert distance.max() < 1e-14, what

    tree = cKDTree(nodes[:, :2])
    x = V.tabulate_dof_coordinates()
    for g, t, ref in zip(steps, times, saved):
        with h5py.File(g['attributes']['u'][0], 'r') as h:
            values = h[g['attributes']['u'][1]][:]
        assert values.shape == (len(nodes), 3), values.shape
        # the field's value at every node, either side of the period
        assert_close(values[:, 0], np.cos(2 * np.pi * nodes[:, 0]) * nodes[:, 1] + t,
                     1e-14, 'u0 at the nodes, t={}'.format(t))
        assert_close(values[:, 1], np.sin(2 * np.pi * nodes[:, 0]) * nodes[:, 1] ** 2,
                     1e-14, 'u1 at the nodes, t={}'.format(t))
        assert not values[:, 2].any(), 'the padding component is not zero'
        # and twoasis's read: each component's dofs matched to the nodes
        back = np.empty_like(ref)
        for c in range(2):
            dofs = np.asarray(V.sub(c).dofmap().dofs())
            _, index = tree.query(x[dofs])
            back[dofs] = values[index, c]
        assert_close(back, ref, 1e-14, 'matched read at t={}'.format(t))
    for g, t in zip(xdmf_grids(tmp('tri_p.xdmf')), times):
        with h5py.File(g['geometry'][0], 'r') as h:
            px = h[g['geometry'][1]][:]
        with h5py.File(g['attributes']['p'][0], 'r') as h:
            pv = h[g['attributes']['p'][1]][:].ravel()
        assert len(px) == mesh.num_vertices(), len(px)
        assert_close(pv, t * np.cos(2 * np.pi * px[:, 0]) + px[:, 1], 1e-14,
                     'p at t={}'.format(t))

    # read_checkpoint takes the last step by default, any other by counter
    for counter, ref in ((-1, saved[-1]), (0, saved[0])):
        back = df.Function(V)
        with XDMFFile2(df.MPI.comm_world, tmp('tri_u.xdmf')) as f:
            f.read_checkpoint(back, 'u', counter)
        assert_close(back.vector().get_local(), ref, 1e-14,
                     'read_checkpoint step {}'.format(counter))
    print('{} steps of {} nodes, one mesh, exact both ways'.format(len(times), len(nodes)))


# -- reading back -----------------------------------------------------------------

@CELLS
def test_read_quadratic_back(cell):
    """a quadratic grid reads back into a P2 Function"""
    mesh = unit_mesh(3, cell)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('rt.xdmf')) as f:
        f.write(u)
    back = df.Function(df.FunctionSpace(mesh, 'CG', 2))
    with XDMFFile2(df.MPI.comm_world, tmp('rt.xdmf')) as f:
        f.read_checkpoint(back, 'u', 0)
    assert_close(back.vector().get_local(), u.vector().get_local(), 1e-12,
                 'P2 round trip')
    # dolfin cannot read this file at all, which is why the path exists
    failed = False
    try:
        with df.XDMFFile(df.MPI.comm_world, tmp('rt.xdmf')) as f:
            f.read_checkpoint(back, 'u', 0)
    except Exception:
        failed = True
    assert failed, 'dolfin read a {} file, so this path is unnecessary'.format(
        QUADRATIC_TOPOLOGY[cell])
    print('exact round trip; dolfin refuses the same file')


@CELLS
def test_read_vector_back(cell):
    """a quadratic vector reads back"""
    mesh = unit_mesh(2, cell)
    u = vector_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('rtv.xdmf')) as f:
        f.write(u)
    back = df.Function(df.VectorFunctionSpace(mesh, 'CG', 2))
    with XDMFFile2(df.MPI.comm_world, tmp('rtv.xdmf')) as f:
        f.read_checkpoint(back, 'u', 0)
    assert_close(back.vector().get_local(), u.vector().get_local(), 1e-12,
                 'P2 vector round trip')
    print('{} components, exact'.format(mesh.geometry().dim()))


@CELLS
def test_read_into_periodic(cell):
    """a periodic space reads back, where coordinate matching does not

    DESIGN-NOTES.md, "Reading the flow back", records two failed attempts at matching
    periodic dof coordinates.

    The read path goes through an unconstrained space and interpolates, so it is
    unaffected -- this is the case that would catch a rewrite that stopped doing so.
    """
    class Periodic(df.SubDomain):
        def inside(self, x, on_boundary):
            return bool(df.near(x[0], 0.0) and on_boundary)

        def map(self, x, y):
            y[0] = x[0] - 1.0
            for i in range(1, len(x)):
                y[i] = x[i]

    mesh = unit_mesh(2, cell)
    V = df.FunctionSpace(mesh, 'CG', 2, constrained_domain=Periodic())
    u = df.Function(V)
    u.rename('u', 'u')
    u.interpolate(df.Expression('sin(2*pi*x[0]) + x[1]*x[1]', degree=2))
    with XDMFFile2(df.MPI.comm_world, tmp('per.xdmf')) as f:
        f.write(u)
    back = df.Function(V)
    with XDMFFile2(df.MPI.comm_world, tmp('per.xdmf')) as f:
        f.read_checkpoint(back, 'u', 0)
    assert V.dim() < df.FunctionSpace(mesh, 'CG', 2).dim(), \
        'the constrained space is not actually smaller, so this proves nothing'
    assert_close(back.vector().get_local(), u.vector().get_local(), 1e-12,
                 'periodic round trip')
    print('{} constrained dofs of {} unconstrained, exact'.format(V.dim(), df.FunctionSpace(mesh, 'CG', 2).dim()))


def test_read_mesh_from_quadratic():
    """reading a mesh out of a quadratic file gives its corners"""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('qm.xdmf')) as f:
        f.write(u)
    back = df.Mesh()
    with XDMFFile2(df.MPI.comm_world, tmp('qm.xdmf')) as f:
        f.read(back)
    assert back.num_cells() == mesh.num_cells(), (back.num_cells(), mesh.num_cells())
    assert back.num_vertices() == mesh.num_vertices()
    assert_close(df.assemble(df.Constant(1) * df.dx(back)),
                 df.assemble(df.Constant(1) * df.dx(mesh)), 1e-14, 'volume')
    print('{} cells recovered from Tetrahedron_10 corners'.format(back.num_cells()))


@CELLS
def test_read_mesh_past_a_facet_function(cell):
    """a facet MeshFunction written first, then a quadratic field: the mesh read back is
    the cells', not the facets'"""
    mesh = unit_mesh(2, cell)
    mf = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
    mf.array()[:] = np.arange(len(mf.array())) % 5
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('qmf.xdmf')) as f:
        f.write(mf)
        f.write(u)
    back = df.Mesh()
    with XDMFFile2(df.MPI.comm_world, tmp('qmf.xdmf')) as f:
        f.read(back)
    assert back.ufl_cell().cellname() == cell
    assert back.num_cells() == mesh.num_cells(), (back.num_cells(), mesh.num_cells())


def _whole_tree_xml(doc):
    """The XML as the whole tree serialises: what `save` writes a grid at a time."""
    import copy
    import xml.etree.ElementTree as ET
    root = copy.deepcopy(doc.root)
    ET.indent(root, space='  ')
    return '<?xml version="1.0"?>\n{}\n{}\n'.format(
        export_p2_xdmf.DOCTYPE, ET.tostring(root, encoding='unicode'))


@pytest.mark.parametrize('share', [False, True])
def test_saved_xml_is_the_whole_tree_after_every_write(share):
    """a grid at a time, appended in place: after every write the file is the whole
    tree's XML -- a static grid first, steps, steps sharing a grid, a grid after the
    series, and a file changed behind the writer's back"""
    mesh = unit_mesh(2)
    u, p = quadratic_field(mesh, 'u'), quadratic_field(mesh, 'p')
    mf = df.MeshFunction('size_t', mesh, 2, 0)
    mf.rename('m', 'm')
    path = tmp('saves_{}.xdmf'.format(share))
    with XDMFFile2(df.MPI.comm_world, path) as f:
        f.parameters['functions_share_mesh'] = share
        f.write(mf)
        for k in range(4):
            f.write(u, float(k))
            f.write(p, float(k))
            assert open(path).read() == _whole_tree_xml(f._doc), k
            if k == 1:
                with open(path, 'a') as handle:     # someone else's edit: rewritten
                    handle.write(' ')
        f.write(mesh)
        assert open(path).read() == _whole_tree_xml(f._doc)
        f.write(u, 9.0)
        assert open(path).read() == _whole_tree_xml(f._doc)
    assert not os.path.exists(path + '.tmp')


def test_read_delegates():
    """a file with no quadratic grid is read by dolfin itself"""
    mesh = unit_mesh(2)
    mf = df.MeshFunction('size_t', mesh, 2, 0)
    mf.array()[:] = np.arange(len(mf.array())) % 7
    mf.rename('m', 'm')
    with df.XDMFFile(df.MPI.comm_world, tmp('plain.xdmf')) as f:
        f.write(mf)
    back = df.MeshFunction('size_t', mesh, 2, 0)
    with XDMFFile2(df.MPI.comm_world, tmp('plain.xdmf')) as f:
        f.read(back, 'm')
    assert_close(back.array(), mf.array(), 0, 'delegated mesh function read')
    print('dolfin-written file read through XDMFFile2')


# -- checkpoints -------------------------------------------------------------------

def test_checkpoint():
    """checkpoints are dolfin's, and already exact for P2"""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('chk.xdmf')) as f:
        f.write_checkpoint(u, 'u', 0.0)
    back = df.Function(df.FunctionSpace(mesh, 'CG', 2))
    with XDMFFile2(df.MPI.comm_world, tmp('chk.xdmf')) as f:
        f.read_checkpoint(back, 'u', 0)
    assert_close(back.vector().get_local(), u.vector().get_local(), 0.0,
                 'checkpoint round trip')
    # and dolfin reads the same file, since it wrote it
    other = df.Function(df.FunctionSpace(mesh, 'CG', 2))
    with df.XDMFFile(df.MPI.comm_world, tmp('chk.xdmf')) as f:
        f.read_checkpoint(other, 'u', 0)
    assert_close(other.vector().get_local(), u.vector().get_local(), 0.0,
                 'dolfin reading the same checkpoint')
    print('bit-exact both ways')


@pytest.mark.mpi
def test_checkpoint_on_three_ranks():
    """an appended checkpoint written and compressed on three ranks (the rewrite at each
    close by every rank) reads back exactly, every step"""
    script = '''
import sys
import h5py, numpy as np, dolfin as df
from femtools.xdmf import XDMFFile2
mesh = df.UnitCubeMesh(16, 16, 16)   # past WORTH_BYTES, so compressed
V = df.FunctionSpace(mesh, 'CG', 2)
us = [df.interpolate(df.Expression('x[0]*x[1] + %d*x[2]' % k, degree=2), V) for k in range(2)]
for k, u in enumerate(us):
    with XDMFFile2(df.MPI.comm_world, sys.argv[1]) as f:
        f.write_checkpoint(u, 'u', float(k), append=k > 0)
for k, u in enumerate(us):
    back = df.Function(V)
    with XDMFFile2(df.MPI.comm_world, sys.argv[1]) as f:
        f.read_checkpoint(back, 'u', k)
    assert np.array_equal(back.vector().get_local(), u.vector().get_local()), k
if df.MPI.rank(df.MPI.comm_world) == 0:
    with h5py.File(sys.argv[1][:-5] + '.h5', 'r') as h:
        sizes = []
        h.visititems(lambda n, o: sizes.append(o.compression) if isinstance(o, h5py.Dataset)
                     and o.size > 1000 else None)
    assert sizes and all(c == 'gzip' for c in sizes), sizes
    print('ok')
'''
    path = os.path.join(TMP, 'chk_par.py')
    with open(path, 'w') as handle:
        handle.write(script)
    proc = support.run([sys.executable, path, tmp('chk_par.xdmf')], 3, timeout=600)
    assert proc.stdout.strip().endswith('ok'), proc.stdout + proc.stderr


@pytest.mark.mpi
def test_quadratic_grid_read_back_on_three_ranks():
    """a P2 field written on three ranks reads back exactly on three, each rank matching
    its own dofs among the nodes routed to it; a mesh that is not the file's is refused
    on every rank rather than leaving the others in the vector's assembly; a DG field
    is refused on every rank rather than written as though it were continuous"""
    script = '''
import sys
import numpy as np, dolfin as df
from femtools.xdmf import XDMFFile2
C = df.MPI.comm_world
mesh = df.UnitCubeMesh(5, 5, 5)
V = df.VectorFunctionSpace(mesh, 'CG', 2)
u = df.interpolate(df.Expression(('x[0]*x[1]', 'x[2]*x[2]', 'x[0] - x[1]'), degree=2), V)
u.rename('u', 'u')
with XDMFFile2(C, sys.argv[1]) as f:
    f.write(u, 0.0)
back = df.Function(V)
with XDMFFile2(C, sys.argv[1]) as f:
    f.read_checkpoint(back, 'u', 0)
same = np.array_equal(back.vector().get_local(), u.vector().get_local())
other = df.Function(df.VectorFunctionSpace(df.UnitCubeMesh(4, 4, 4), 'CG', 2))
try:
    with XDMFFile2(C, sys.argv[1]) as f:
        f.read_checkpoint(other, 'u', 0)
    refused = False
except ValueError:
    refused = True
# a DG field is refused on every rank, not interpolated onto Lagrange nodes
dg = df.Function(df.FunctionSpace(mesh, 'DG', 0)); dg.rename('d', 'd')
try:
    with XDMFFile2(C, sys.argv[1] + '.dg.xdmf') as f:
        f.write(dg)
    dg_refused = False
except ValueError as exc:
    dg_refused = 'Lagrange' in str(exc)
ok = C.allreduce(int(same and refused and dg_refused), op=__import__('mpi4py').MPI.MIN)
if df.MPI.rank(C) == 0:
    print('ok' if ok else 'FAILED')
'''
    path = os.path.join(TMP, 'quad_par.py')
    with open(path, 'w') as handle:
        handle.write(script)
    proc = support.run([sys.executable, path, tmp('quad_par.xdmf')], 3, timeout=600)
    assert proc.stdout.strip().endswith('ok'), proc.stdout + proc.stderr


def test_checkpoint_conflict():
    """a checkpoint and visualisation grids refuse to share a file"""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('conflict.xdmf')) as f:
        f.write(u)
        try:
            f.write_checkpoint(u, 'u', 0.0)
        except RuntimeError as exc:
            assert 'layout of its own' in str(exc), str(exc)
        else:
            raise AssertionError('a checkpoint was written over the grids')
    with XDMFFile2(df.MPI.comm_world, tmp('conflict2.xdmf')) as f:
        f.write_checkpoint(u, 'u', 0.0)
        try:
            f.write(u)
        except RuntimeError as exc:
            assert 'checkpoint' in str(exc), str(exc)
        else:
            raise AssertionError('grids were written over a checkpoint')
    print('both orders refuse, naming the conflict')


# -- the XML is really XDMF ---------------------------------------------------------

@CELLS
def test_meshio_reads_it(cell):
    """an independent reader agrees the file is XDMF

meshio knows nothing of dolfin or of this module, so it is the outside check."""
    import meshio
    mesh = unit_mesh(2, cell)
    last = mesh.geometry().dim() - 1
    u = vector_field(mesh)
    p = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    p.rename('p', 'p')
    p.interpolate(df.Expression('x[{}]'.format(last), degree=1))
    with XDMFFile2(df.MPI.comm_world, tmp('mio.xdmf')) as f:
        f.write((u, p))
    got = meshio.read(tmp('mio.xdmf'))
    kind = MESHIO_QUADRATIC[cell]
    assert kind in {b.type for b in got.cells}, [b.type for b in got.cells]
    assert set(got.point_data) == {'u', 'p'}, set(got.point_data)
    coords, conn = _read_grid_geometry(tmp('mio.xdmf'), _scan(tmp('mio.xdmf'))[0])
    assert_close(got.points, coords, 1e-14, 'meshio vs this reader, coordinates')
    assert np.array_equal(got.get_cells_type(kind), conn), 'meshio reads other cells'
    assert_close(got.point_data['p'].ravel(), coords[:, last], 1e-12,
                 'meshio reading the pressure')
    print('{} points, {} {} cells, fields {}'.format(len(got.points), len(got.get_cells_type(kind)), kind, ','.join(sorted(got.point_data))))


def test_ascii_refused():
    """ASCII encoding is refused, not silently written as HDF5

Support was removed: the arrays always go to HDF5. The string, any casing
    and dolfin's own enum are all refused the same way, whether the file is fresh
    or already holds grids -- and the refusal leaves the file usable."""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('ascii.xdmf')) as f:
        for value in ('ASCII', 'ascii', df.XDMFFile.Encoding.ASCII):
            try:
                f.write(u, value)
            except ValueError as exc:
                assert 'ASCII' in str(exc), str(exc)
            else:
                raise AssertionError('{!r} was accepted'.format(value))
        f.write(u)                       # the refusals changed nothing
        try:
            f.write(u, df.XDMFFile.Encoding.ASCII)   # nor is it accepted later
        except ValueError:
            pass
        else:
            raise AssertionError('ASCII was accepted once the file held a grid')
    assert os.path.exists(tmp('ascii.h5')), 'the HDF5 write did not happen'
    assert len(_scan(tmp('ascii.xdmf'))) == 1, 'a refused write left a grid'
    print('refused before and after a write; HDF5 unaffected')


# -- the standalone route still works ------------------------------------------------

@CELLS
def test_writer_agreement(cell):
    """write_quadratic and XDMFFile2 write the same layout"""
    from femtools.xdmf import write_quadratic
    mesh = unit_mesh(2, cell)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('cls.xdmf')) as f:
        f.write(u)
    grid = _scan(tmp('cls.xdmf'))[0]
    coords, conn = _read_grid_geometry(tmp('cls.xdmf'), grid)
    values = _read_grid_attribute(tmp('cls.xdmf'), grid, 'u')
    write_quadratic(tmp('fn.xdmf'), coords, conn, [values.ravel()], ['u'],
                    cell_type=MESHIO_QUADRATIC[cell])
    other = _scan(tmp('fn.xdmf'))[0]
    assert other['topology'] == grid['topology'] == QUADRATIC_TOPOLOGY[cell]
    c2, k2 = _read_grid_geometry(tmp('fn.xdmf'), other)
    v2 = _read_grid_attribute(tmp('fn.xdmf'), other, 'u')
    assert_close(c2, coords, 0.0, 'coordinates')
    assert_close(k2, conn, 0.0, 'connectivity')
    assert_close(v2.ravel(), values.ravel(), 0.0, 'values')
    print('both writers agree bit for bit')


def test_mesh_then_p1_share():
    """a mesh and a P1 field can share one grid

The P1 node set has to be in vertex order for this to be sound.

    `write(mesh)` numbers nodes by vertex; the P1 dofmap does not. Sharing a grid
    between the two while they disagreed put dof-ordered values against
    vertex-ordered coordinates -- a file that looks entirely plausible and has the
    field scrambled across it.
    """
    mesh = unit_mesh(3)
    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    u.rename('u', 'u')
    u.interpolate(df.Expression('x[0] + 5*x[1] + 25*x[2]', degree=1))
    with XDMFFile2(df.MPI.comm_world, tmp('share_mesh.xdmf')) as f:
        f.parameters['functions_share_mesh'] = True
        f.write(mesh)
        f.write(u)
    grids = _scan(tmp('share_mesh.xdmf'))
    assert len(grids) == 1, 'the field should have joined the mesh grid'
    coords, conn = _read_grid_geometry(tmp('share_mesh.xdmf'), grids[0])
    values = _read_grid_attribute(tmp('share_mesh.xdmf'), grids[0], 'u').ravel()
    want = coords[:, 0] + 5 * coords[:, 1] + 25 * coords[:, 2]
    assert_close(values, want, 1e-12, 'values against the shared coordinates')
    print('{} nodes, field and coordinates in step'.format(len(coords)))


def test_mesh_and_quadratic_groups():
    """a mesh grid and a quadratic grid keep their own arrays

They cannot share an HDF5 group: one holds 4-node cells and the other 10."""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('two.xdmf')) as f:
        f.write(mesh)
        f.write(u)
    grids = _scan(tmp('two.xdmf'))
    assert len(grids) == 2, [g['topology'] for g in grids]
    linear = [g for g in grids if g['topology'] == 'Tetrahedron'][0]
    quad = [g for g in grids if g['topology'] == 'Tetrahedron_10'][0]
    assert linear['topology_item'][1] != quad['topology_item'][1], \
        'both grids point at the same array'
    lx, lc = _read_grid_geometry(tmp('two.xdmf'), linear)
    qx, qc = _read_grid_geometry(tmp('two.xdmf'), quad)
    assert lc.shape[1] == 4 and qc.shape[1] == 10, (lc.shape, qc.shape)
    assert len(lx) == mesh.num_vertices(), (len(lx), mesh.num_vertices())
    assert_close(np.sort(lx, axis=0), np.sort(_pad(mesh.coordinates()), axis=0),
                 1e-14, 'the mesh grid still holds the mesh')
    print('linear {} cells, quadratic {} cells, separate arrays'.format(len(lc), len(qc)))


def test_mesh_value_collection():
    """a mesh value collection is written as the entities it marks

Its mesh comes from the file, since dolfin does not expose it on the object."""
    mesh = unit_mesh(2)
    mvc = df.MeshValueCollection('size_t', mesh, 2)
    mvc.set_value(0, 0, 3)
    mvc.set_value(1, 1, 4)
    mvc.rename('m', 'm')
    with XDMFFile2(df.MPI.comm_world, tmp('mvc.xdmf')) as f:
        f.write(mesh)
        f.write(mvc)
    grids = _scan(tmp('mvc.xdmf'))
    assert len(grids) == 2, [g['topology'] for g in grids]
    marked = [g for g in grids if 'm' in g['attributes']][0]
    assert marked['topology'] == 'Triangle', marked['topology']
    values = _read_grid_attribute(tmp('mvc.xdmf'), marked, 'm').ravel()
    assert sorted(values.tolist()) == [3, 4], values
    # the same shape dolfin writes: a mesh grid, then the marked entities
    with df.XDMFFile(df.MPI.comm_world, tmp('mvc_dolfin.xdmf')) as f:
        f.write(mvc)
    theirs = _scan(tmp('mvc_dolfin.xdmf'))
    assert [g['topology'] for g in theirs] == [g['topology'] for g in grids], \
        ([g['topology'] for g in theirs], [g['topology'] for g in grids])
    print('2 marked facets, same grid shape as dolfin')


def test_mesh_value_collection_needs_mesh():
    """a mesh value collection with no mesh in the file says so"""
    mesh = unit_mesh(2)
    mvc = df.MeshValueCollection('size_t', mesh, 2)
    mvc.set_value(0, 0, 3)
    with XDMFFile2(df.MPI.comm_world, tmp('mvc_bare.xdmf')) as f:
        try:
            f.write(mvc)
        except ValueError as exc:
            assert 'does not carry its mesh' in str(exc), str(exc)
        else:
            raise AssertionError('a MeshValueCollection was written with no mesh')
    print('raises, naming what to write first')


def test_points():
    """points are written, with and without values"""
    pts = [df.Point(0.0, 0.0, 0.0), df.Point(1.0, 0.0, 0.0), df.Point(0.0, 1.0, 0.5)]
    with XDMFFile2(df.MPI.comm_world, tmp('pts.xdmf')) as f:
        f.write(pts)
    grid = _scan(tmp('pts.xdmf'))[0]
    assert grid['topology'] == 'Polyvertex', grid['topology']
    coords, _ = _read_grid_geometry(tmp('pts.xdmf'), grid)
    assert_close(coords, [[0, 0, 0], [1, 0, 0], [0, 1, 0.5]], 1e-14, 'point positions')
    assert not grid['attributes'], grid['attributes']

    # the values are an array, and used to be mistaken for the trailing encoding
    with XDMFFile2(df.MPI.comm_world, tmp('ptsv.xdmf')) as f:
        f.write(pts, [1.0, 2.0, 3.0])
    grid = _scan(tmp('ptsv.xdmf'))[0]
    assert list(grid['attributes']) == ['values'], grid['attributes']
    got = _read_grid_attribute(tmp('ptsv.xdmf'), grid, 'values').ravel()
    assert_close(got, [1.0, 2.0, 3.0], 1e-14, 'point values')
    print('3 points, bare and with values')


@CELLS
def test_routes_identical(cell):
    """the standalone route writes the identical file

Both write vertices-then-edges now, so the two files agree bit for bit."""
    mesh = unit_mesh(3, cell)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('ident.xdmf')) as f:
        f.write(u)
    grid = _scan(tmp('ident.xdmf'))[0]
    coords, conn = _read_grid_geometry(tmp('ident.xdmf'), grid)
    # the node numbering is the mesh's vertices, then one per edge
    assert len(coords) == mesh.num_vertices() + mesh.num_edges(), (
        len(coords), mesh.num_vertices(), mesh.num_edges())
    assert_close(coords[:mesh.num_vertices()], mesh.coordinates(), 0.0,
                 'the first nodes are the mesh vertices, in order')
    # and the rest are the edge midpoints, in edge order
    mesh.init(1, 0)
    ends = mesh.coordinates()[np.array([e.entities(0) for e in df.edges(mesh)])]
    assert_close(coords[mesh.num_vertices():], ends.mean(axis=1), 1e-15,
                 'the edge nodes are the edge midpoints, in order')
    print('{} vertices then {} edge nodes'.format(mesh.num_vertices(), mesh.num_edges()))


def test_cpp_matches_fallback():
    """the caster-refused fallbacks write what the C++ helper writes

What is left to compare, now that the compiler is assumed.

    The connectivity fallback is gone -- `_cpp()` raises rather than returning None,
    so there is one path and nothing to hold it against. The two fallbacks that
    remain are not about the compiler: the pybind casters refuse *argument types*
    they have no overload for (only `size_t` MeshValueCollections, only real dolfin
    Points), and that happens with a perfectly good helper. So the helper is asked
    to refuse, rather than to be absent, and the Python paths must still write the
    same file.

    It also insists the helper actually built here, which is now what every other
    dolfin case in this file depends on too.
    """
    assert export_p2_xdmf._cpp() is not None, \
        'the C++ helper did not build (a compiler and libdolfin-dev are assumed)'
    mesh = unit_mesh(6)
    pts = [df.Point(0.1 * k, 0.2 * k, 0.3 * k) for k in range(5)]
    mvc = df.MeshValueCollection('size_t', mesh, 2)
    for k in range(0, mesh.num_cells(), 5):
        mvc.set_value(k, k % 4, k % 9)
    mvc.rename('c', 'c')

    class Refusing(object):
        """The helper with its two casters refusing, which is what a value type or
        a point type without an overload looks like from the writer's side."""

        def __init__(self, real):
            self._real = real

        def connectivity(self, *a):
            return self._real.connectivity(*a)

        def points_array(self, *a):
            raise TypeError('no overload')

        def mvc_values(self, *a):
            raise TypeError('no overload')

    def write_all(tag):
        with XDMFFile2(df.MPI.comm_world, tmp(tag + '_pts.xdmf')) as f:
            f.write(pts)
        with XDMFFile2(df.MPI.comm_world, tmp(tag + '_mvc.xdmf')) as f:
            f.write(mesh)
            f.write(mvc)

    saved = export_p2_xdmf._CPP_MODULE
    try:
        write_all('cpp_on')
        export_p2_xdmf._CPP_MODULE = Refusing(saved)
        write_all('cpp_off')
    finally:
        export_p2_xdmf._CPP_MODULE = saved

    pa, _ = _read_grid_geometry(tmp('cpp_on_pts.xdmf'), _scan(tmp('cpp_on_pts.xdmf'))[0])
    pb, _ = _read_grid_geometry(tmp('cpp_off_pts.xdmf'), _scan(tmp('cpp_off_pts.xdmf'))[0])
    assert_close(pa, pb, 0.0, 'points across the two paths')
    ma = [g for g in _scan(tmp('cpp_on_mvc.xdmf')) if 'c' in g['attributes']][0]
    mb = [g for g in _scan(tmp('cpp_off_mvc.xdmf')) if 'c' in g['attributes']][0]
    _, ea = _read_grid_geometry(tmp('cpp_on_mvc.xdmf'), ma)
    _, eb = _read_grid_geometry(tmp('cpp_off_mvc.xdmf'), mb)
    assert np.array_equal(ea, eb), 'MVC entities differ between the two paths'
    assert_close(_read_grid_attribute(tmp('cpp_on_mvc.xdmf'), ma, 'c'),
                 _read_grid_attribute(tmp('cpp_off_mvc.xdmf'), mb, 'c'), 0.0,
                 'MVC values across the two paths')
    print('{} points and {} marked facets, identical either way'.format(len(pa), len(ea)))


def test_checkpoint_signature():
    """checkpoint signatures are dolfin's: keywords, append, last-step default

dolfin's names are function_name and time_step, append must be forwarded
    (dolfin truncates the file on every non-appending call, so dropping it kept
    only the last step), and read_checkpoint's default counter is -1, the LAST
    step -- a drop-in defaulting to the first hands back the wrong field."""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('ck_sig.xdmf')) as f:
        f.write_checkpoint(u, function_name='u', time_step=0.0)
        u.vector()[:] = 2.0 * u.vector()[:]
        f.write_checkpoint(u, 'u', 1.0, append=True)
    back = df.Function(u.function_space())
    with XDMFFile2(df.MPI.comm_world, tmp('ck_sig.xdmf')) as f:
        f.read_checkpoint(back, 'u')            # dolfin's default: the last step
    assert_close(back.vector().get_local(), u.vector().get_local(), 1e-14,
                 'the appended (last) step')
    first = df.Function(u.function_space())
    with XDMFFile2(df.MPI.comm_world, tmp('ck_sig.xdmf')) as f:
        f.read_checkpoint(first, 'u', 0)
    assert_close(2.0 * first.vector().get_local(), back.vector().get_local(),
                 1e-14, 'step 0 must survive the append')
    print('append kept both steps; the default read is the last')


def test_write_without_close():
    """the XML is on disk after every write, close or no close

A caller with no context manager and no close() used to end with datasets
    in the .h5 and no .xdmf naming them; the loss was silent."""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    f = XDMFFile2(df.MPI.comm_world, tmp('noclose.xdmf'))
    f.write(u)
    assert os.path.exists(tmp('noclose.xdmf')), 'no .xdmf until close()'
    assert len(_scan(tmp('noclose.xdmf'))) == 1
    f.close()
    print('the file is complete before close')


def test_time_on_mesh_refused():
    """a time on anything but a Function is refused, as dolfin refuses it

dolfin has no write(Mesh, t) overload and says so; this used to accept the
    time and silently write a grid with no <Time> in it."""
    mesh = unit_mesh(2)
    mf = df.MeshFunction('size_t', mesh, 2, 0)
    with XDMFFile2(df.MPI.comm_world, tmp('mesh_t.xdmf')) as f:
        for obj in (mesh, mf):
            try:
                f.write(obj, 0.5)
            except TypeError:
                pass
            else:
                raise AssertionError(
                    'a time on {} was accepted'.format(type(obj).__name__))
    print('write(Mesh, t) and write(MeshFunction, t) both raise')


def test_numpy_time():
    """numpy scalar times and the time= keyword are accepted

`times[i]` off a float32 array is an np.float32, which is not a float
    subclass; it was misrouted to encoding and rejected. The old time= keyword
    must keep working too."""
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('np_time.xdmf')) as f:
        f.write(u, np.float32(0.5))
        f.write(u, time=np.int64(1))
    grids = _scan(tmp('np_time.xdmf'))
    times = [g['time'] for g in grids if g['time'] is not None]
    assert times == [0.5, 1.0], times
    print('np.float32 positionally and np.int64 by keyword both land as <Time>')


def test_p1_on_triangles():
    """a P1 field on a triangle mesh is written, as dolfin writes it

Degree 1 has no midsides and works on any simplex; refusing it was a
    regression against both dolfin and the older delegating implementation. A P2
    field on the same mesh is a `Triangle_6` grid, and P2 on an interval, which has
    no midside numbering here, is refused by name."""
    mesh = df.UnitSquareMesh(4, 4)
    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    u.rename('u', 'u')
    u.interpolate(df.Expression('x[0] + 2*x[1]', degree=1))
    with XDMFFile2(df.MPI.comm_world, tmp('tri_p1.xdmf')) as f:
        f.write(u)
    g = _scan(tmp('tri_p1.xdmf'))[0]
    assert g['topology'] == 'Triangle', g['topology']
    coords, conn = _read_grid_geometry(tmp('tri_p1.xdmf'), g)
    vals = _read_grid_attribute(tmp('tri_p1.xdmf'), g, 'u').ravel()
    assert_close(vals, coords[:, 0] + 2 * coords[:, 1], 1e-12, 'P1 values')
    assert conn.shape == (mesh.num_cells(), 3), conn.shape
    assert len(coords) == mesh.num_vertices(), (len(coords), mesh.num_vertices())
    w = df.Function(df.FunctionSpace(mesh, 'CG', 2))
    with XDMFFile2(df.MPI.comm_world, tmp('tri_p2.xdmf')) as f:
        f.write(w)
    assert _scan(tmp('tri_p2.xdmf'))[0]['topology'] == 'Triangle_6'
    line = df.Function(df.FunctionSpace(df.UnitIntervalMesh(4), 'CG', 2))
    with XDMFFile2(df.MPI.comm_world, tmp('int_p2.xdmf')) as f:
        try:
            f.write(line)
        except ValueError as exc:
            assert 'interval' in str(exc), str(exc)
        else:
            raise AssertionError('P2 on an interval was accepted')
    print('{} triangles written; P2 on them is Triangle_6, on an interval refused'
          .format(len(conn)))


def test_mvc_value_types():
    """double and int mesh value collections keep their values

Every value type was force-cast to uint64: a double collection's 1.5 came
    back as 1, and an int collection's -1 as 2**64-1. dolfin writes each type
    faithfully, so must this; only size_t has a C++ fast path, the rest take the
    Python loop and their own dtype."""
    mesh = unit_mesh(2)
    for kind, value, tag in (('double', 1.5, 'd'), ('int', -1, 'i')):
        mvc = df.MeshValueCollection(kind, mesh, 2)
        mvc.set_value(0, 0, value)
        mvc.rename('m', 'm')
        path = tmp('mvc_{}.xdmf'.format(tag))
        with XDMFFile2(df.MPI.comm_world, path) as f:
            f.write(mesh)
            f.write(mvc)
        grid = [g for g in _scan(path) if 'm' in g['attributes']][0]
        got = _read_grid_attribute(path, grid, 'm').ravel()
        assert got[0] == value, (kind, got[0])
    print('1.5 stays 1.5 and -1 stays -1')


PARALLEL_SCRIPT = """
import os, sys
import numpy as np, dolfin as df
from femtools.xdmf import XDMFFile2
df.set_log_level(40)
df.parameters['ghost_mode'] = os.environ.get('GHOST_MODE', 'none')
C = df.MPI.comm_world; R = df.MPI.rank(C)
mesh = df.UnitCubeMesh(5, 5, 5)
u = df.Function(df.FunctionSpace(mesh, 'CG', 2)); u.rename('u', 'u')
u.interpolate(df.Expression('x[0]*x[0] + 2*x[1]*x[1] - x[2]*x[2]', degree=2))
v = df.Function(df.VectorFunctionSpace(mesh, 'CG', 2)); v.rename('v', 'v')
v.interpolate(df.Expression(('x[0]*x[0]', 'x[1]*x[2]', 'x[2]*x[2]'), degree=2))
q = df.Function(df.FunctionSpace(mesh, 'CG', 1)); q.rename('q', 'q')
q.interpolate(df.Expression('x[0] + x[1]', degree=1))
mf = df.MeshFunction('size_t', mesh, 2, 0); mf.rename('m', 'm')
for fc in df.facets(mesh):
    p = fc.midpoint()
    mf[fc] = int(round(5*p.x()) + 6*round(5*p.y()) + 36*round(5*p.z()))
# cell data beside the nodal fields, marked geometrically so every partition marks
# the same physical cells -- _content_equal then matches it cell for cell
tags = df.MeshFunction('size_t', mesh, 3, 0); tags.rename('t', 't')
for c in df.cells(mesh):
    p = c.midpoint()
    tags[c] = int(round(10*p.x()) + 11*round(10*p.y()) + 121*round(10*p.z()))
pts = [df.Point(0.1*k, 0.2*k, 0.3*k) for k in range(7)]
# marked geometrically, so every partition marks the same physical entities --
# an MVC used to have no gather path at all and lost every rank's marks but zero's
mvc = df.MeshValueCollection('size_t', mesh, 2)
for c in df.cells(mesh):
    for j, fc in enumerate(df.facets(c)):
        p = fc.midpoint()
        if p.z() < 1e-9:
            mvc.set_value(c.index(), j, int(round(5*p.x()) + 6*round(5*p.y())))
mvc.rename('mv', 'mv')
with XDMFFile2(C, sys.argv[1]) as f:
    f.write(mesh)
    f.write(mf)
    f.write((v, q, tags))
    f.write(u)
    f.write(mvc)
    # rank zero holds all the points and the others write empty slabs, which is
    # the case that used to skip the collective write and deadlock the close
    f.write(pts, [float(k) for k in range(7)])
    driver = f._doc.driver
if R == 0:
    print('DRIVER=' + str(driver))
"""


def _run_parallel(path, ranks, force_serial_h5py=False, ghost_mode='none'):
    """Runs the script above under mpirun and returns the driver it used.

    `env=` is not cosmetic: MPI_Init sets 26 PMIX_* variables through the C setenv,
    which Python's os.environ never sees, so an inherited environment tells mpirun it
    is already inside a job and it exits 1 with nothing on either stream.
    """
    script = os.path.join(TMP, 'par_all.py')
    with open(script, 'w') as handle:
        handle.write(PARALLEL_SCRIPT.format(root=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
    env = support.clean_env()
    cmd = ['mpirun', '-n', str(ranks)]
    if force_serial_h5py:
        env['H5PY_NEVER_USE_MPI'] = '1'
        cmd += ['-x', 'H5PY_NEVER_USE_MPI']
    env['GHOST_MODE'] = ghost_mode
    cmd += ['-x', 'GHOST_MODE']
    proc = subprocess.run(cmd + [sys.executable, script, path],
                          capture_output=True, timeout=900, env=env)
    if proc.returncode != 0:
        raise AssertionError('mpirun -n {} failed: {}'.format(
            ranks, proc.stderr.decode()[-800:] or proc.stdout.decode()[-800:]))
    for line in proc.stdout.decode().splitlines():
        if line.startswith('DRIVER='):
            return line.split('=', 1)[1]
    raise AssertionError('the run did not report which driver it used')


def _content_equal(a, b):
    """Same file, compared the only way a parallel one can be.

    dolfin's mesh partitioning is **not reproducible**: three runs of one program over
    two ranks gave three different partitions, three different global edge numberings
    and three different dof ownership splits. So two parallel files cannot be compared
    byte for byte -- their node order legitimately differs -- and everything is matched
    by position instead. Global *vertex* numbering is stable, which is why a Mesh and a
    P1 field do come out byte-identical; edge nodes, numbered after partitioning by
    `init_global(1)`, do not.
    """
    from scipy.spatial import cKDTree
    grids_a, grids_b = _scan(a), _scan(b)
    assert len(grids_a) == len(grids_b), (len(grids_a), len(grids_b))
    for ga, gb in zip(grids_a, grids_b):
        assert ga['topology'] == gb['topology'], (ga['topology'], gb['topology'])
        xa, ca = _read_grid_geometry(a, ga)
        xb, cb = _read_grid_geometry(b, gb)
        assert xa.shape == xb.shape and ca.shape == cb.shape, \
            (xa.shape, xb.shape, ca.shape, cb.shape)
        distance, index = cKDTree(xb).query(xa)
        assert distance.max() < 1e-12, 'node sets differ by {:.2e}'.format(
            distance.max())
        assert len(set(index.tolist())) == len(index), 'the node match is not 1-1'
        back = np.empty(len(index), dtype=np.int64)
        back[index] = np.arange(len(index))
        for name in ga['attributes']:
            va = _read_grid_attribute(a, ga, name)
            vb = _read_grid_attribute(b, gb, name)
            if len(va) == len(xa):
                assert_close(va, vb[index], 1e-12, 'nodal field ' + name)
            else:
                ka = np.hstack([np.sort(ca, axis=1), va])
                kb = np.hstack([np.sort(back[cb], axis=1), vb])
                assert np.array_equal(ka[np.lexsort(ka.T)], kb[np.lexsort(kb.T)]), \
                    'cell field {} differs'.format(name)
        assert np.array_equal(np.sort(np.sort(ca, axis=1), axis=0),
                              np.sort(np.sort(back[cb], axis=1), axis=0)), \
            'the cells differ'


EMPTY_RANK_SCRIPT = """
import sys
import numpy as np, dolfin as df
from femtools.xdmf import XDMFFile2
df.set_log_level(40)
C = df.MPI.comm_world; R = df.MPI.rank(C); N = df.MPI.size(C)

# A mesh built by hand, every cell on rank zero: nothing dolfin partitions leaves a
# rank empty, so this shape only arises from a mesh a caller assembles itself -- the
# no-slip surface of a flow, whose facets can miss a rank entirely.
n_cell_global, n_vert_global = 2, 4
pts = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
tris = np.array([[0,1,2],[1,3,2]])
mine = range(4) if R == 0 else range(0)
mesh = df.Mesh(C)
ed = df.MeshEditor()
ed.open(mesh, 'triangle', 2, 3)
ed.init_vertices_global(len(list(mine)), n_vert_global)
ed.init_cells_global(len(tris) if R == 0 else 0, n_cell_global)
if R == 0:
    for i, x in enumerate(pts):
        ed.add_vertex_global(i, i, x)
    for i, t in enumerate(tris):
        ed.add_cell(i, t.astype(np.uintp))
ed.close()

V = df.VectorFunctionSpace(mesh, 'CG', 1)
u = df.Function(V); u.rename('u', 'u')
u.interpolate(df.Expression(('x[0]', 'x[1]', 'x[0]+x[1]'), degree=1))
tags = df.MeshFunction('size_t', mesh, 2); tags.rename('t', 't')
for c in df.cells(mesh):
    tags[c] = 5 + c.index()
with XDMFFile2(C, sys.argv[1]) as f:
    f.write((u, tags))
if R == 0:
    print('OK dim={{}} vertices={{}}'.format(V.dim(), mesh.num_entities_global(0)))
"""


TRIANGLE_SCRIPT = """
import os, sys
import numpy as np, dolfin as df
from femtools.xdmf import XDMFFile2
df.set_log_level(40)
C = df.MPI.comm_world
out = sys.argv[1]


class Periodic(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


mesh = df.UnitSquareMesh(C, 8, 8)
V = df.VectorFunctionSpace(mesh, 'CG', 2, constrained_domain=Periodic())
P = df.FunctionSpace(mesh, 'CG', 1, constrained_domain=Periodic())
Q = df.FunctionSpace(mesh, 'CG', 2)
u = df.Function(V); u.rename('u', 'u')
p = df.Function(P); p.rename('p', 'p')
q = df.Function(Q); q.rename('q', 'q')
# marked geometrically, so every partition marks the same physical cells
tags = df.MeshFunction('size_t', mesh, 2, 0); tags.rename('t', 't')
for c in df.cells(mesh):
    m = c.midpoint()
    tags[c] = int(round(48*m.x()) + 49*round(48*m.y()))
# the series the way twoasis writes them: a file per field, the mesh once, flushed
files = {}
for name in ('u', 'p'):
    files[name] = XDMFFile2(C, os.path.join(out, name + '.xdmf'))
    files[name].parameters['rewrite_function_mesh'] = False
    files[name].parameters['flush_output'] = True
for t in (0.0, 0.5, 1.0):
    u.interpolate(df.Expression(%r, t=t, degree=4))
    p.interpolate(df.Expression('t*cos(2*pi*x[0]) + x[1]', t=t, degree=2))
    files['u'].write(u, t)
    files['p'].write(p, t)
driver = files['u']._doc.driver
for f in files.values():
    f.close()
q.interpolate(df.Expression(%r, degree=2))
with XDMFFile2(C, os.path.join(out, 'uq.xdmf')) as f:
    f.write((u, q, tags))
# and back, the last step into the periodic space
back = df.Function(V)
with XDMFFile2(C, os.path.join(out, 'u.xdmf')) as f:
    f.read_checkpoint(back, 'u')
err = df.MPI.max(C, float(np.abs(back.vector().get_local()
                                 - u.vector().get_local()).max()))
if df.MPI.rank(C) == 0:
    print('DRIVER=' + driver)
    print('ERR=%%.3e' %% err)
""" % (PERIODIC_VELOCITY, QUADRATIC_EXPRESSION[2])


@pytest.mark.mpi
def test_triangles_rank_counts():
    """P2 on triangles: one and three ranks write the same files

MPI-IO on three, and rank zero's writes (serial h5py) on three, against serial.

    A periodic vector series written the way twoasis writes it, and a P2 scalar
    beside it on one grid with cell data, read back on the same ranks into the
    periodic space. dolfin numbers the edges after partitioning, so the files are
    compared by node position (`_content_equal`), not byte for byte.
    """
    script = os.path.join(TMP, 'tri_ranks.py')
    with open(script, 'w') as handle:
        handle.write(TRIANGLE_SCRIPT)
    folders = {}
    for tag, ranks, gather in (('1', 1, False), ('3', 3, False), ('3g', 3, True)):
        folder = folders[tag] = tmp('tri_ranks_' + tag)
        os.makedirs(folder)
        argv = (['-x', 'H5PY_NEVER_USE_MPI'] if gather else []) + \
            [sys.executable, script, folder]
        proc = support.run(argv, ranks, timeout=900,
                           env={'H5PY_NEVER_USE_MPI': '1'} if gather else None)
        said = dict(line.split('=', 1) for line in proc.stdout.splitlines()
                    if line.startswith(('DRIVER=', 'ERR=')))
        assert said.get('DRIVER') == ('mpio' if ranks > 1 and not gather
                                      else 'serial'), (tag, said)
        assert float(said['ERR']) < 1e-12, (tag, said)
    for name, topology in (('u', 'Triangle_6'), ('p', 'Triangle'), ('uq', 'Triangle_6')):
        serial = os.path.join(folders['1'], name + '.xdmf')
        assert {g['topology'] for g in _scan(serial)} == {topology}, name
        for tag in ('3', '3g'):
            _content_equal(os.path.join(folders[tag], name + '.xdmf'), serial)
    assert _scan(os.path.join(folders['1'], 'uq.xdmf'))[0]['centers'] == \
        {'u': 'Node', 'q': 'Node', 't': 'Cell'}
    print('serial, 3 ranks (mpio) and 3 (gather) agree; read back exact')


@pytest.mark.mpi
def test_empty_rank():
    """a rank owning nothing does not hang the collective write

This deadlocked, and the deadlock is the point.

    `dofmap.entity_dofs` hands back a Python list; `np.asarray([])` is float64, which
    cannot index -- so the rank owning no entities raised while every other rank sat
    inside the collective HDF5 open, at full CPU, forever. A test that only checked
    the values would have passed on one rank and hung on three, so this one asserts
    the run *finishes*.
    """
    script = os.path.join(TMP, 'empty_rank.py')
    with open(script, 'w') as handle:
        handle.write(EMPTY_RANK_SCRIPT.format(root=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
    out = tmp('empty_rank.xdmf')
    proc = support.run([sys.executable, script, out], 3, timeout=300)
    assert 'OK dim=12' in proc.stdout, proc.stdout[-400:]
    grids = _scan(out)
    assert len(grids) == 1, len(grids)
    x, conn = _read_grid_geometry(out, grids[0])
    u = _read_grid_attribute(out, grids[0], 'u')
    assert len(x) == 4 and len(conn) == 2, (len(x), len(conn))
    assert_close(u[:, 0], x[:, 0], 1e-14, 'u0 against x')
    assert_close(u[:, 2], x[:, 0] + x[:, 1], 1e-14, 'u2 against x+y')
    assert grids[0]['centers']['t'] == 'Cell', grids[0]['centers']
    print('3 ranks, all 2 cells on one of them: finished and correct')


@pytest.fixture(scope='module')
def parallel_runs(_scratch):
    """The script above, once each: serially, on three ranks through MPI-IO and through
    rank zero (h5py's serial build), and on three with ghost cells -- the files every
    test below compares, written once for all of them. `(path, driver)` by name."""
    runs = {}
    for tag, ranks, serial_h5py, ghosts in (('1', 1, False, 'none'),
                                            ('3', 3, False, 'none'),
                                            ('3g', 3, True, 'none'),
                                            ('3gh', 3, False, 'shared_facet')):
        path = tmp('par_{}.xdmf'.format(tag))
        runs[tag] = (path, _run_parallel(path, ranks, serial_h5py, ghosts))
    return runs


@pytest.mark.mpi
def test_mpio_is_used(parallel_runs):
    """a parallel write really goes through MPI-IO

Without this, a silent fall back to rank zero's writes would pass every other case."""
    import h5py
    driver = parallel_runs['3'][1]
    if not h5py.get_config().mpi and driver != 'mpio':
        # the serial h5py is all that is installed; say so rather than pass quietly
        raise AssertionError(
            'no parallel HDF5: install python3-h5py-mpi, or rank zero writes it all')
    assert driver == 'mpio', 'the collective path was not taken (driver={})'.format(
        driver)
    assert parallel_runs['3g'][1] == 'serial'


@pytest.mark.mpi
@pytest.mark.parametrize('tag', ['3', '3g', '3gh'])
def test_parallel_matches_serial(parallel_runs, tag):
    """three ranks write the serial file, every format: through MPI-IO, through rank
    zero (H5PY_NEVER_USE_MPI), and with ghost cells (`shared_facet`), each cell then
    written once, by its owner"""
    _content_equal(parallel_runs[tag][0], parallel_runs['1'][0])


@pytest.mark.mpi
def test_mesh_rank_count_independent(parallel_runs):
    """a mesh written in parallel is byte-identical to the serial one

The property dolfin has and this keeps: global vertex numbering is stable.

    It holds for a Mesh and for P1 because their nodes are vertices. It does *not*
    hold for P2, whose edge nodes are numbered after partitioning -- and that is
    dolfin's non-determinism, not a choice made here.
    """
    one, three = parallel_runs['1'][0], parallel_runs['3'][0]
    a = [g for g in _scan(one) if g['topology'] == 'Tetrahedron'][0]
    b = [g for g in _scan(three) if g['topology'] == 'Tetrahedron'][0]
    xa, _ = _read_grid_geometry(one, a)
    xb, _ = _read_grid_geometry(three, b)
    assert_close(xa, xb, 0.0, 'mesh vertices across rank counts')


def test_points_twice():
    """two point writes into one file keep their own arrays

They used to share the group names `Points/*`, with no counter.

    `dataset` recreates a group that already exists, so the second write deleted the
    first one's arrays while the first grid's XML went on pointing at them: a grid
    declaring three points against a dataset holding two, and those two the *other*
    write's.
    """
    a = [df.Point(0.0, 0.0, 0.0), df.Point(1.0, 0.0, 0.0), df.Point(0.0, 1.0, 0.0)]
    b = [df.Point(0.0, 0.0, 1.0), df.Point(1.0, 1.0, 1.0)]
    with XDMFFile2(df.MPI.comm_world, tmp('twice.xdmf')) as f:
        f.write(a)
        f.write(b)
    grids = _scan(tmp('twice.xdmf'))
    assert len(grids) == 2, len(grids)
    first, _ = _read_grid_geometry(tmp('twice.xdmf'), grids[0])
    second, _ = _read_grid_geometry(tmp('twice.xdmf'), grids[1])
    assert_close(first, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e-14, 'the first set')
    assert_close(second, [[0, 0, 1], [1, 1, 1]], 1e-14, 'the second set')
    print('3 points then 2, both intact')


def test_untimed_grid_placement():
    """an untimed grid stays out of the temporal collection

A grid with no <Time> inside a temporal collection is ambiguous.

    dolfin does something worse with the same sequence -- measured, `write(mesh)`
    followed by `write(u, t)` drops the mesh grid altogether -- so this is a
    deliberate departure rather than a parity gap.
    """
    import xml.etree.ElementTree as ET
    mesh = unit_mesh(2)
    u = quadratic_field(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('untimed.xdmf')) as f:
        f.write(mesh)
        f.write(u, 0.0)
        f.write(u, 1.0)
    root = ET.parse(tmp('untimed.xdmf')).getroot()
    series = [g for g in root.iter('Grid') if g.get('CollectionType') == 'Temporal']
    assert len(series) == 1, len(series)
    inside = [g for g in series[0] if g.tag == 'Grid']
    assert len(inside) == 2, 'only the timed grids belong in the collection'
    for g in inside:
        assert g.find('Time') is not None, 'a collection member with no time'
    # and the mesh grid is still in the file, at the top level
    top = [g for g in root.find('Domain') if g.tag == 'Grid'
           and g.get('GridType') == 'Uniform']
    assert len(top) == 1, 'the mesh grid was lost'
    print('2 timed grids inside, the mesh grid outside')


def _pad(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] < 3:
        coords = np.column_stack([coords] + [np.zeros(len(coords))]
                                 * (3 - coords.shape[1]))
    return coords


@pytest.fixture(scope='module', autouse=True)
def _scratch():
    """The scratch directory every case in this file writes into.

    A module global rather than a fixture argument because fifty cases use it and the
    generated MPI scripts interpolate it; autouse so none of them has to ask.
    """
    global TMP
    TMP = tempfile.mkdtemp(prefix='xdmf2_')
    try:
        yield TMP
    finally:
        shutil.rmtree(TMP, ignore_errors=True)






QUADRATIC_BLOCKS = """
import sys
import numpy as np
from mpi4py import MPI
from femtools.xdmf import write_quadratic, write_quadratic_blocks
C = MPI.COMM_WORLD
rng = np.random.default_rng(4)
n_nodes, n_cells = 300, 80
coords = rng.random((n_nodes, 3))
conn = rng.integers(0, n_nodes, (n_cells, 10))
u, v = rng.random(n_nodes), rng.random((n_nodes, 3))
if C.rank == 0:
    write_quadratic(sys.argv[1], coords, conn, [u, v], ['u', 'v'])
cells = np.arange(n_cells)[C.rank::C.size]
nodes = np.arange(n_nodes)[C.rank::C.size]
write_quadratic_blocks(sys.argv[2], C, cells, conn[cells], n_cells, nodes, coords[nodes],
                       [u[nodes], v[nodes]], ['u', 'v'], n_nodes)
try:
    write_quadratic_blocks(sys.argv[2] + '.bad.xdmf', C, cells, conn[cells], n_cells, nodes,
                           coords[nodes], [rng.random((len(nodes), 4))], ['w'], n_nodes)
    refused = False
except ValueError:
    refused = True
if C.allreduce(int(refused), op=MPI.MIN) and C.rank == 0:
    print('refused everywhere')
"""


@pytest.mark.mpi
def test_quadratic_blocks_write_what_one_rank_writes():
    """`write_quadratic_blocks` from three ranks' shares is `write_quadratic`'s file;
    a field of four components is refused on every rank, none left in a barrier"""
    path = os.path.join(TMP, 'qblocks.py')
    with open(path, 'w') as handle:
        handle.write(QUADRATIC_BLOCKS)
    one, three = tmp('qb_one.xdmf'), tmp('qb_three.xdmf')
    proc = support.run([sys.executable, path, one, three], 3, timeout=300)
    assert 'refused everywhere' in proc.stdout, proc.stdout + proc.stderr
    a, b = _scan(one), _scan(three)
    assert len(a) == len(b) == 1
    xa, ca = _read_grid_geometry(one, a[0])
    xb, cb = _read_grid_geometry(three, b[0])
    assert np.array_equal(xa, xb) and np.array_equal(ca, cb)
    for name in ('u', 'v'):
        assert np.array_equal(_read_grid_attribute(one, a[0], name),
                              _read_grid_attribute(three, b[0], name))


def test_a_2d_mesh_reads_back_2d():
    """a 2D mesh is written XY, as dolfin writes it, so it reads back with two
    coordinates: from a linear file (dolfin's reader) and from a quadratic one"""
    mesh = df.UnitSquareMesh(3, 3)
    with XDMFFile2(df.MPI.comm_world, tmp('m2d.xdmf')) as f:
        f.write(mesh)
    with XDMFFile2(df.MPI.comm_world, tmp('q2d.xdmf')) as f:
        f.write(quadratic_field(mesh))
    for name in ('m2d.xdmf', 'q2d.xdmf'):
        back = df.Mesh()
        with XDMFFile2(df.MPI.comm_world, tmp(name)) as f:
            f.read(back)
        assert back.geometry().dim() == 2, name
        assert back.num_cells() == mesh.num_cells(), name


@pytest.mark.parametrize('degree', [1, 2])
def test_a_moved_mesh_is_written_where_it_is(degree):
    """the node layout is kept between writes, its coordinates are not: a mesh moved
    between two steps is stored at each step where it was"""
    mesh = df.UnitSquareMesh(3, 3)
    u = df.Function(df.FunctionSpace(mesh, 'CG', degree))
    u.rename('u', 'u')
    with XDMFFile2(df.MPI.comm_world, tmp('moved{}.xdmf'.format(degree))) as f:
        f.write(u, 0.0)
        mesh.coordinates()[:] *= 2.0
        f.write(u, 1.0)
    steps = _scan(tmp('moved{}.xdmf'.format(degree)))
    first, second = (_read_grid_geometry(tmp('moved{}.xdmf'.format(degree)), g)[0]
                     for g in steps)
    assert first.max() == 1.0 and second.max() == 2.0


def test_an_interval_mesh_is_written_xy():
    """XDMF has no one-column geometry: a 1D mesh's goes out as XY, y = 0"""
    mesh = df.UnitIntervalMesh(4)
    u = df.interpolate(df.Expression('x[0]', degree=1), df.FunctionSpace(mesh, 'CG', 1))
    u.rename('u', 'u')
    with XDMFFile2(df.MPI.comm_world, tmp('line.xdmf')) as f:
        f.write(mesh)
        f.write(u)
    for g in _scan(tmp('line.xdmf')):
        x, _ = _read_grid_geometry(tmp('line.xdmf'), g)
        assert x.shape == (5, 2) and not x[:, 1].any()
    assert 'GeometryType="XY"' in open(tmp('line.xdmf')).read()
