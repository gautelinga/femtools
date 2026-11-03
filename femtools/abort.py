"""Every command's entry point: one rank's failure ends the job instead of hanging it.

Under MPI an exception on one rank leaves the others waiting in the next collective
until the job's time limit runs out (a refusal raised on one rank, a full disk on rank
zero's write). `run` turns any uncaught exception into `MPI_Abort` when the job has
more than one rank, after printing it with its rank, so every launch form -- `mpirun
-n P <command>`, the module form, `python -m mpi4py -m ...` -- stops the job with the
message. Serially it changes nothing: the exception propagates as it always did.

If `MPI_Abort` itself does not end the process within `ABORT_GRACE` seconds, the kernel
does: `SIGALRM` at its default action terminates the process whatever holds the
interpreter's lock or has installed a handler (PETSc does), and `srun` and `mpirun` end a
job whose task dies.

A deliberate exit that every rank takes together (a time-budget stop, say) raises
`CollectiveExit`, which `run` lets through as an ordinary exit with its status.

Neither dolfin nor mpi4py is imported here: a command that never initialised MPI has no
collective to hang in.
"""
import contextlib
import signal
import sys
import traceback

#: seconds `MPI_Abort` is given to end the process before the kernel does (`_abort`)
ABORT_GRACE = 60


class CollectiveExit(SystemExit):
    """A `SystemExit` every rank raises together, decided on values they all share:
    `run` exits with its status rather than aborting the job."""


def _world():
    """MPI's world communicator if MPI is up with more than one rank, else None."""
    if 'mpi4py' not in sys.modules and 'dolfin' not in sys.modules:
        return None
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    if not MPI.Is_initialized() or MPI.Is_finalized():
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def run(main, argv=None):
    """`main(argv)` as a command (argv None, as a console script calls it, is sys.argv's):
    its return value is the exit status. Under MPI with more than one rank, an
    uncaught exception other than `CollectiveExit` -- a `SystemExit` with a message or a
    non-zero status included -- is printed with the rank and aborts every rank."""
    try:
        return main(argv)
    except CollectiveExit:
        raise
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        comm = _world()
        if comm is None:
            raise
        message = exc.code if isinstance(exc.code, str) else 'exit status {}'.format(exc.code)
        _abort(comm, message, exc.code if isinstance(exc.code, int) else 1)
    except BaseException:
        comm = _world()
        if comm is None:
            raise
        _abort(comm, traceback.format_exc().rstrip(), 1)


@contextlib.contextmanager
def aborting(comm, cleanup=None):
    """A region the ranks of `comm` run collectively: an exception on any rank is
    printed with its rank and aborts the job, since the others wait for it in the
    region's collectives; `cleanup()` is called first (what the job leaves behind). With
    no comm or one rank it propagates as usual.

    Put it *inside* a file opened collectively (MPI-IO): the file's close is itself
    collective, so an exception leaving through it first would block the failing rank
    there, the others waiting in the write, and never reach the abort."""
    try:
        yield
    except BaseException:
        if comm is None or comm.Get_size() == 1:
            raise
        message = traceback.format_exc().rstrip()
        if cleanup is not None:
            try:
                cleanup()
            except Exception:       # noqa: BLE001 -- the abort is what matters now
                pass
        _abort(comm, message, 1)


def _abort(comm, message, status):
    sys.stdout.flush()
    sys.stderr.write('rank {} of {}: {}\nrank {}: aborting the job (femtools.abort)\n'
                     .format(comm.Get_rank(), comm.Get_size(), message, comm.Get_rank()))
    sys.stderr.flush()
    try:                                    # the default action, not a handler: see above
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(ABORT_GRACE)
    except ValueError:                      # not the main thread: no alarm to set
        pass
    comm.Abort(status)
