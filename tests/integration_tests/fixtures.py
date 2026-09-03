import os
import shutil
import subprocess

import pytest
from tests.utils import get_abs_path


@pytest.fixture
def run_local_pipeline():
    def _run_local_pipeline(**kwargs):
        # Construct the command with keyword arguments
        script_path = get_abs_path(
            __file__, "../simulators_and_runners/run_local_synthetic.sh"
        )
        # Ensure the script runs from the repository root so outputs land in the correct location
        run_location = os.path.dirname(script_path)

        # Empty the simulated_data folder
        sim_data_dir = os.path.join(run_location, "simulated_data")
        try:
            os.makedirs(sim_data_dir, exist_ok=True)
            for entry in os.listdir(sim_data_dir):
                path = os.path.join(sim_data_dir, entry)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
        except Exception as e:
            raise RuntimeError(f"Failed to empty simulated_data: {e}") from e

        # Default off FOR TESTS ONLY (the CLI's own default is on): these integration tests
        # assert numeric correctness of the analysis outputs, and whether a toy config's
        # diagnostics happen to be flagged (exit status 3) is incidental to that -- a
        # diagnostic-threshold recalibration must not spuriously fail them. A test about the
        # exit behavior itself can override by passing fail_on_flagged_diagnostics="1".
        kwargs.setdefault("fail_on_flagged_diagnostics", "0")
        args = [f"--{key}={value}" for key, value in kwargs.items()]

        # subprocess, NOT the sh library, which this fixture used until 2026-09-02.
        #
        # sh runs a bare os.fork() and then executes a substantial amount of PYTHON in the
        # child before execv -- signal handling, os.setsid(), dup2, fd closes, and on macOS a
        # BLOCKING os.read on a sync pipe. Forking a multithreaded process and then running
        # non-async-signal-safe code in the child is the classic deadlock pattern: the child
        # inherits mutexes held by threads that do not exist in it. By the time these tests
        # run, JAX's thread pool is live (every unit test in the same pytest process warms it
        # up), and JAX installs an os.register_at_fork hook that warned about exactly this on
        # every one of these tests: "os.fork() was called. os.fork() is incompatible with
        # multithreaded code, and JAX is multithreaded, so this will likely lead to a
        # deadlock."
        #
        # subprocess never runs Python in the child: CPython either uses posix_spawn (no fork
        # at all -- _USE_POSIX_SPAWN is True on this platform) or _posixsubprocess.fork_exec,
        # which forks and execs entirely in async-signal-safe C. Verified: the fork warning is
        # gone.
        completed = subprocess.run(
            [script_path, *args],
            cwd=run_location,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Bash script failed with exit status {completed.returncode}: "
                f"{completed.stderr}"
            )
        # The script's STDOUT, so that str(result) still yields it -- sh's RunningCommand
        # stringified to stdout, and test_RL_diagnostics_smoke asserts on that text.
        return completed.stdout

    return _run_local_pipeline
