"""
Generates (or regenerates) the committed benchmark fixtures used by
tests/benchmarks/test_analyze_dataset_benchmark.py, at each scale in SCALES.

This runs the existing synthetic simulation + analysis CLI pipeline
(run_local_synthetic.sh) once per scale, and copies:
  - the three analyze_dataset *input* pickles (study_df.pkl, pi_args.pkl,
    rl_update_args.pkl) into tests/benchmarks/fixtures/<scale>/, so the
    benchmark can call analyze_dataset directly without paying simulation
    cost or subprocess overhead on every run.
  - the CLI's analysis.pkl *output* into
    tests/benchmarks/fixtures/<scale>/golden_analysis.pkl, to serve as the
    numerical regression oracle.

Only run this deliberately, when you want to regenerate a fixture (e.g. the
synthetic parameters below changed) or refresh a golden output after an
intentional, independently-verified change to the numerical results. Do not
run it just to "make a failing benchmark test pass" -- a failing numerical
comparison in the benchmark means something changed the answer, which needs
to be understood before the golden file is updated.

Usage: python tests/benchmarks/generate_fixture.py [scale ...]
       (default: all scales in SCALES; pass e.g. "small" to regenerate just one)
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SIMULATORS_DIR = REPO_ROOT / "tests" / "simulators_and_runners"
SIMULATED_DATA_DIR = SIMULATORS_DIR / "simulated_data"
FIXTURES_ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"

# "small" is fast to iterate on for correctness but too small to show a
# meaningful before/after for optimizations whose cost scales with problem
# size (input_checks.py, the jax.jacrev hot path) -- both scale with
# subjects x decision times. "medium" matches the scale already used by
# tests/integration_tests (n=100, T=10, recruit_n=20), so timings here are
# comparable to those tests' ~2-3 minute runtimes. Everything else uses
# run_local_synthetic.sh's own defaults (see that script for the full list).
SCALES = {
    "small": {
        "T": "6",
        "n": "20",
        "recruit_n": "10",
        "suppress_interactive_data_checks": "1",
    },
    "medium": {
        "T": "10",
        "n": "100",
        "recruit_n": "20",
        "suppress_interactive_data_checks": "1",
    },
}


def _find_run_output_dir() -> pathlib.Path:
    # run_local_synthetic.sh derives its output folder name from its own
    # shell variables (see its own comment: "This should really be output by
    # that script or passed into it as an arg, but alas"). Rather than
    # hand-reconstructing that naming convention here too -- a third place
    # that would silently drift out of sync with the script -- we empty
    # simulated_data/ first (matching tests/integration_tests/fixtures.py's
    # run_local_pipeline) and then just look for whatever one directory the
    # script produced.
    candidates = [p for p in SIMULATED_DATA_DIR.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one output directory under {SIMULATED_DATA_DIR} "
            f"after running the simulation, found {len(candidates)}: {candidates}. "
        )
    run_output_dir = candidates[0] / "exp=1"
    if not run_output_dir.is_dir():
        raise FileNotFoundError(f"Expected experiment output directory not found: {run_output_dir}.")
    return run_output_dir


def _generate_one(scale: str) -> None:
    simulation_args = SCALES[scale]
    fixture_dir = FIXTURES_ROOT / scale

    if SIMULATED_DATA_DIR.is_dir():
        shutil.rmtree(SIMULATED_DATA_DIR)

    args = [f"--{key}={value}" for key, value in simulation_args.items()]
    subprocess.run(
        ["./run_local_synthetic.sh", *args],
        cwd=SIMULATORS_DIR,
        check=True,
    )

    run_output_dir = _find_run_output_dir()

    fixture_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dest_name in [
        ("study_df.pkl", "study_df.pkl"),
        ("pi_args.pkl", "pi_args.pkl"),
        ("rl_update_args.pkl", "rl_update_args.pkl"),
        ("analysis.pkl", "golden_analysis.pkl"),
    ]:
        shutil.copyfile(run_output_dir / src_name, fixture_dir / dest_name)

    print(f"Wrote '{scale}' fixture files to {fixture_dir}")


def main() -> None:
    requested = sys.argv[1:] or list(SCALES.keys())
    unknown = set(requested) - set(SCALES.keys())
    if unknown:
        raise ValueError(f"Unknown scale(s) {unknown}; choose from {list(SCALES.keys())}")

    for scale in requested:
        _generate_one(scale)


if __name__ == "__main__":
    main()
