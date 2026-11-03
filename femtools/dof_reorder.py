"""`femtools-build-dof-reorder`: dolfin's dof ordering without its construction peak, as a
library to preload.

dolfin 2019 orders each rank's owned dofs with a graph that joins every dof of a cell to every
other -- for Taylor-Hood every dof is a node, so the graph has the mixed matrix's pattern --
and hands a copy to SCOTCH. That graph is built and dropped inside the function space's
construction, and can be its peak. `cpp/dof_reorder_star.cpp` is dolfin's
`DofMapBuilder::compute_node_reordering` with the graph a star a cell; any permutation of a
rank's owned dofs is a valid numbering, so only locality can change, and a reader of a
stored Function has to go through the cell dofmap stored beside it.

It cannot be switched from Python: `DofMapBuilder::build` calls the function inside
libdolfin. Where that call goes through the PLT, a library defining the same symbol and
loaded first (`LD_PRELOAD`) replaces it for the whole process. So this builds the library
against the dolfin it runs under -- inside the image, since int32 and int64 builds differ
in the index type the symbol's name carries -- and checks the three things that make the
preload take effect, refusing a library that would load and change nothing:

- it links libdolfin alone, and every library it needs resolves (it loads before python has
  loaded dolfin, so libdolfin's own runpath does not serve it: its runpath names libdolfin's
  directory). MPI is not linked: its calls resolve through libdolfin's own MPI library,
  which in a container may be the host's bind-mounted over it under another soname;
- it exports exactly the symbol libdolfin exports;
- libdolfin calls that symbol through its PLT. Debian's dolfin does not (it binds the call
  inside the library); some builds do.

    femtools-build-dof-reorder --out run/libdof_reorder_star.so
    LD_PRELOAD=run/libdof_reorder_star.so python3 solve.py ...

`DOF_REORDER_GRAPH=clique` in the environment restores dolfin's graph under the same preload.
"""
import os
import shlex
import subprocess

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp', 'dof_reorder_star.cpp')
SYMBOL_PART = 'DofMapBuilder23compute_node_reordering'


def _run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _exported(lib):
    """The mangled compute_node_reordering a shared library exports, or None."""
    names = [line.split()[-1] for line in _run(['nm', '-D', '--defined-only', lib]).splitlines()
             if SYMBOL_PART in line]
    return names[0] if len(names) == 1 else None


def build(out, compiler='c++'):
    """Compile `dof_reorder_star.cpp` into `out` and check the preload would take effect.
    Returns the libdolfin it replaces the function in; raises with the reason otherwise."""
    cflags = shlex.split(_run(['pkg-config', '--cflags', 'dolfin']))
    libdir = _run(['pkg-config', '--variable=libdir', 'dolfin']).strip()
    cmd = ([compiler, '-O2', '-fPIC', '-shared', '-std=c++17'] + cflags + [SOURCE, '-o', out,
           '-L' + libdir, '-ldolfin', '-Wl,-rpath,' + libdir])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError('build-dof-reorder: the compile failed:\n' + proc.stderr[-4000:])

    deps = {}
    for line in _run(['ldd', out]).splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == '=>':
            deps[parts[0]] = parts[2]
    missing = sorted(n for n, p in deps.items() if p == 'not')
    if missing:
        raise RuntimeError('build-dof-reorder: {} does not resolve {}'.format(out, missing))
    dolfin = next((os.path.realpath(p) for n, p in deps.items() if n.startswith('libdolfin')),
                  None)
    if dolfin is None:
        raise RuntimeError('build-dof-reorder: {} does not link libdolfin'.format(out))

    ours, theirs = _exported(out), _exported(dolfin)
    if ours is None or ours != theirs:
        raise RuntimeError('build-dof-reorder: {} exports {}, {} exports {}: the preload would '
                           'not replace it'.format(out, ours, dolfin, theirs))
    # a call through the PLT reads `call ... <symbol@plt>` in the disassembly
    disasm = subprocess.run(['objdump', '-d', '--no-show-raw-insn', dolfin],
                            capture_output=True, text=True, check=True).stdout
    if '<{}@plt>'.format(ours) not in disasm:
        raise RuntimeError('build-dof-reorder: {} calls compute_node_reordering inside itself, '
                           'not through its PLT: a preload changes nothing under this dolfin'
                           .format(dolfin))
    return dolfin


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='femtools-build-dof-reorder', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True,
                    help='the shared library to write, e.g. run/libdof_reorder_star.so')
    ap.add_argument('--compiler', default='c++',
                    help='the C++ compiler (default c++; MPI\'s headers come from dolfin\'s '
                         'pkg-config flags, and MPI is deliberately not linked)')
    a = ap.parse_args(argv)
    try:
        dolfin = build(a.out, a.compiler)
    except RuntimeError as e:
        raise SystemExit(str(e))
    print('build-dof-reorder: {} replaces compute_node_reordering in {}; preload it with '
          'LD_PRELOAD'.format(a.out, dolfin))
    return 0
