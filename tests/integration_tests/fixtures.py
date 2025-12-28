import pytest
import sh
import os

from tests.utils import get_abs_path
import shutil


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

        command = sh.Command(script_path)
        args = [f"--{key}={value}" for key, value in kwargs.items()]
        try:
            result = command(*args, _cwd=run_location)
        except sh.ErrorReturnCode as e:
            raise RuntimeError(
                f"Bash script failed with error: {e.stderr.decode()}"
            ) from e
        return result

    return _run_local_pipeline
