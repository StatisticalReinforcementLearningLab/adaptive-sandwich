import pytest
import sh

from tests.integration_tests.utils import get_abs_path


@pytest.fixture
def run_local_pipeline():
    def _run_local_pipeline(**kwargs):
        # Construct the command with keyword arguments
        script_path = get_abs_path(__file__, "../../run_local.sh")
        command = sh.Command(script_path)
        args = [f"--{key}={value}" for key, value in kwargs.items()]
        try:
            result = command(*args)
        except sh.ErrorReturnCode as e:
            raise RuntimeError(
                f"Bash script failed with error: {e.stderr.decode()}"
            ) from e
        return result

    return _run_local_pipeline
