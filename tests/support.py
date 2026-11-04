"""What the suites share: a child process on one rank or under `mpirun`."""
import os
import shutil
import subprocess


def clean_env():
    """The environment without MPI_Init's own variables, which a nested mpirun would
    read as being inside a job already; Open MPI's settings (`OMPI_MCA_*`, such as
    oversubscription) are kept."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_")) or k.startswith("OMPI_MCA_")}


def run(argv, ranks=1, env=None, timeout=900, check=True, cwd=None, text=True):
    """`argv` in a child, under `mpirun -n ranks` when `ranks` > 1 (skipped where there
    is no mpirun, failed instead with FEMTOOLS_REQUIRE_MPI set, as CI sets it), one
    OpenMP thread a rank; raises with the output's tails on failure."""
    if ranks > 1 and shutil.which("mpirun") is None:
        import pytest
        if os.environ.get("FEMTOOLS_REQUIRE_MPI"):
            pytest.fail("no mpirun on this machine, and FEMTOOLS_REQUIRE_MPI is set")
        pytest.skip("no mpirun on this machine")
    full = clean_env()
    full["OMP_NUM_THREADS"] = "1"
    full.update(env or {})
    cmd = (["mpirun", "-n", str(ranks)] if ranks > 1 else []) + list(argv)
    proc = subprocess.run(cmd, capture_output=True, text=text, timeout=timeout, env=full,
                          cwd=cwd)
    if check and proc.returncode:
        out, err = proc.stdout, proc.stderr
        if not text:
            out, err = out.decode(errors="replace"), err.decode(errors="replace")
        raise AssertionError("{} exited {}:\n{}{}".format(
            " ".join(cmd), proc.returncode, out[-3000:], err[-3000:]))
    return proc


def lines(stdout):
    """A child's stdout as lines, without dolfin's notices (its JIT announces each
    compile on a cold cache, as on a fresh CI runner, and on no warm one)."""
    return [line for line in stdout.strip().splitlines()
            if not line.startswith("Calling FFC just-in-time")]
