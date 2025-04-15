#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The expected worfklow here is the following, which allows one to run Kelly's and
Nowell's analysis code on the same data from a single experiment for comparison. The experiment
should be run from a branch containing Kelly's code (mostly), while the comparison happens here,
a branch containing Nowell's code (mostly).

git checkout pre_nowell_state_mostly_but_with_inputs_collected_for_package
deactivate; source .kelly_venv/bin/activate
./run_for_comparison_with_lifejacket.sh
<copy the suggested lifejacket command>
git checkout main
deactivate; source .venv/bin/activate
<run the suggested lifejacket command>
./compare_kelly_nowell_analyses.py --containing_folder <path to the folder containing the two analyses>
"""

import pickle
import click


@click.command()
@click.option(
    "--containing_folder",
    type=click.Path(exists=True),
    required=True,
    help="Path to a single experiment folder which should contain analyses using both Nowell's and Kelly's code",
)
def compare_kelly_nowell_analyses(containing_folder) -> None:
    """
    Compare the results of Kelly's and Nowell's analyses on the same experiment.
    Assumes the experiment and analyses have already been run (see above for instructions.)

    Args:
        containing_folder (str): Path to the folder containing the analyses.
    """
    with open(f"{containing_folder}/kelly_analysis.pkl", "rb") as f:
        kelly_analysis = pickle.load(f)
    with open(f"{containing_folder}/analysis.pkl", "rb") as f:
        nowell_analysis = pickle.load(f)
    with open(f"{containing_folder}/debug_pieces.pkl", "rb") as f:
        nowell_debug_pieces = pickle.load(f)
    with open(f"{containing_folder}/study_RLalg.pkl", "rb") as f:
        rl_alg_obj = pickle.load(f)

    print("*** High-level results ***")
    print("\nTheta estimates:")
    print("Kelly:\n", kelly_analysis["LS_estimator"])
    print("Nowell:\n", nowell_analysis["theta_est"])

    print("\nClassical sandwich estimates:")
    print("Kelly:\n", kelly_analysis["sandwich_var"])
    print("Nowell:\n", nowell_analysis["classical_sandwich_var_estimate"])

    print("\nAdaptive sandwich estimates:")
    print("Kelly:\n", kelly_analysis["adaptive_sandwich"])
    print("Nowell:\n", nowell_analysis["adaptive_sandwich_var_estimate"])


if __name__ == "__main__":
    compare_kelly_nowell_analyses()  # pylint: disable=no-value-for-parameter
