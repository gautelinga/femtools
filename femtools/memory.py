"""Hand freed heap back to the operating system, and say how much a process holds.

Once petsc4py -- and so dolfin -- is imported, glibc stops returning freed large blocks to
the operating system: a large numpy array freed afterwards leaves the resident size where
it was, where plain Python gives it back. So every phase of a solve keeps the high-water mark of all the phases
before it, a later allocation cannot reuse the fragmented free space, and a resident
size read off `ru_maxrss` or `top` counts heap nobody holds. `malloc_trim(0)` returns
it, in milliseconds.
"""
import ctypes

_LIBC = []


def release():
    """Return free heap to the operating system (glibc `malloc_trim(0)`). A no-op
    where there is no glibc."""
    if not _LIBC:
        try:
            _LIBC.append(ctypes.CDLL('libc.so.6').malloc_trim)
        except (OSError, AttributeError):
            _LIBC.append(None)
    if _LIBC[0] is not None:
        _LIBC[0](0)


def drop():
    """Free what the caller has just let go of, and hand the heap back (`release`).

    A dolfin mesh that a function space was ever built on is in a reference cycle (its
    UFL domain holds it as `ufl_cargo`, and it holds the domain), so dropping its last
    name does not free it: only Python's cyclic collector does, when its allocation
    counts say so. Called wherever a mesh is let go."""
    import gc
    gc.collect()
    release()


def resident_gb():
    """This process's resident size now, in GiB (Linux; 0 elsewhere)."""
    try:
        with open('/proc/self/statm') as f:
            import os
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30
    except OSError:
        return 0.0
