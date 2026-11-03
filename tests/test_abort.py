"""`femtools.abort`: one rank's failure ends an MPI job instead of hanging it.

    python3 -m pytest tests/test_abort.py

Serially `run` changes nothing. Under `mpirun -n 2`, a failure on rank 1 while rank 0
waits in a collective ends the job promptly with the message, a `SystemExit` with a
message is treated the same, and a `CollectiveExit` every rank raises together keeps its
status. The MPI cases need mpi4py and mpirun.
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from femtools.abort import CollectiveExit, run
from tests import support


def test_serially_nothing_changes():
    assert run(lambda argv: 7) == 7
    assert run(lambda argv: len(argv), ['a', 'b']) == 2
    with pytest.raises(SystemExit, match='refused'):
        run(lambda argv: (_ for _ in ()).throw(SystemExit('refused')))
    with pytest.raises(ValueError):
        run(lambda argv: int('x'))
    with pytest.raises(CollectiveExit) as e:
        run(lambda argv: (_ for _ in ()).throw(CollectiveExit(3)))
    assert e.value.code == 3


def test_an_abort_that_never_returns_is_ended_by_the_kernel():
    """`_abort` with an `MPI_Abort` that never returns:
    the kernel ends the process by SIGALRM after ABORT_GRACE, the message printed first."""
    hang = (
        "import time\n"
        "from femtools import abort\n"
        "abort.ABORT_GRACE = 2\n"
        "class Comm:\n"
        "    def Get_rank(self): return 0\n"
        "    def Get_size(self): return 2\n"
        "    def Abort(self, status):\n"
        "        time.sleep(600)\n"
        "abort._abort(Comm(), 'it failed', 1)\n")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", hang], capture_output=True, text=True,
                          timeout=60, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == -signal.SIGALRM, (proc.returncode, proc.stderr[-500:])
    assert time.time() - t0 < 30 and "rank 0 of 2: it failed" in proc.stderr


SCRIPT = r"""
import sys
from mpi4py import MPI
from femtools.abort import CollectiveExit, run

def main(argv):
    comm = MPI.COMM_WORLD
    kind = sys.argv[1]
    if kind == 'collective':
        comm.Barrier()
        raise CollectiveExit(3)
    if comm.Get_rank() == 1:
        if kind == 'raise':
            raise RuntimeError('rank 1 cannot go on')
        raise SystemExit('rank 1 refuses')
    comm.Barrier()          # rank 0 waits here for a rank that never comes
    return 0

sys.exit(run(main))
"""


@pytest.mark.mpi
@pytest.mark.parametrize('kind, status, text', [
    ('raise', None, 'rank 1 cannot go on'),
    ('exit', None, 'rank 1 refuses'),
    ('collective', 3, None),
])
def test_one_rank_failing_ends_the_job(tmp_path, kind, status, text):
    pytest.importorskip('mpi4py')
    if not shutil.which('mpirun'):
        pytest.skip('no mpirun on this machine')
    script = tmp_path / 'job.py'
    script.write_text(SCRIPT)
    env = support.clean_env()
    env['OMP_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(['mpirun', '-n', '2', sys.executable, str(script), kind],
                          capture_output=True, text=True, timeout=120, env=env)
    out = proc.stdout + proc.stderr
    if status is None:
        assert proc.returncode != 0, out
        assert text in out and 'aborting the job' in out, out
    else:
        assert proc.returncode == status, out
        assert 'aborting' not in out, out
