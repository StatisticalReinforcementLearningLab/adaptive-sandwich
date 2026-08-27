"""
Direct old-path-vs-new-path equivalence + timing/memory measurement for
lifejacket.batched_weighted_estimating_function_stack's
combine_updates_into_one_vmap opt-in, against this repo's own real
tests/benchmarks fixtures (see docs/adr/0001-adaptive-sandwich-performance-plan.md
and MEMORY.md's ADS-139 performance-audit notes for the wider context this
opt-in was built for).

The shipped fixtures' own alg_update_func (RL_least_squares_loss_regularized.py)
does not accept a validity mask, so it cannot exercise
alg_update_func_args_mask_index (a prerequisite for combine_updates_into_one_vmap)
at all. RL_least_squares_loss_regularized_masked.py (same directory as the
other functions_to_pass_to_analysis fixtures) is a mask-aware variant that is
algebraically IDENTICAL to the original on real, unpadded data (mask
all-ones) -- the appended mask argument is the only difference -- so it can
be run through analyze_dataset three ways on the exact same underlying
fixture data and compared:

  1. "baseline": original alg_update_func, no mask, no combine -- exactly
     today's shipped path (same as test_analyze_dataset_benchmark.py).
  2. "masked": masked alg_update_func, alg_update_func_args_mask_index >= 0,
     combine_updates_into_one_vmap=False -- the existing (already-shipped)
     mask-only shape-bucket-consolidation feature, run for the first time
     through a full analyze_dataset call rather than only the lower-level
     unit tests.
  3. "combined": masked alg_update_func, same mask config, PLUS
     combine_updates_into_one_vmap=True -- the new feature under test.
  4. "auto": masked alg_update_func, same mask config,
     combine_updates_into_one_vmap left UNSET (None) -- the auto default,
     which must resolve to True on this eligible (masked, no
     previous_betas) fixture and produce results identical to the explicit
     "combined" variant, end to end through analyze_dataset.

Both fixtures' rl_update_args carry genuinely ragged, staggered-recruitment
per-subject/per-update history lengths (confirmed by inspection -- not a
synthetic scenario), so this exercises the real self-pad/global-pad/
update_fill_index machinery on real data, not a toy.

All three must agree with the existing golden fixture (golden_analysis.pkl)
to float32 noise -- this both confirms combine_updates_into_one_vmap is a
true no-op on the answer, AND (per constraint 2 in the task this was built
for) directly compares the OLD and NEW paths on the same real input.

Also prints, honestly, the forward_vjp / backward_vmap_chunks phase timing
and peak-RSS log lines already emitted by
construct_classical_and_adjusted_sandwiches for the baseline vs. combined
runs at both fixture scales -- see this module's test function docstrings.
"""

from __future__ import annotations

import logging
import pickle
import re
import time

import numpy as np
import pytest
from tests.utils import get_abs_path

from lifejacket.helper_functions import load_function_from_same_named_file
from lifejacket.post_deployment_analysis import analyze_dataset

FUNCTIONS_DIR = get_abs_path(
    __file__, "../simulators_and_runners/functions_to_pass_to_analysis"
)

PHASE_LOG_PATTERN = re.compile(
    r"^Phase '(?P<phase>.+)' took (?P<seconds>[\d.]+) seconds\.$"
)
PEAK_RSS_PATTERN = re.compile(r"^Peak RSS after (?P<phase>\S+): (?P<mb>[\d.]+) MB$")

# RL_least_squares_loss_regularized_masked's argument positions (see that
# file's own docstring): base_states, treat_states, actions, rewards,
# action1probs, action1probtimes are the genuinely ragged (per-decision-time
# history) positions; the mask is appended as a new, 11th (index 10) last
# argument.
MASKED_RAGGED_INDICES = (1, 2, 3, 4, 5, 6)
MASKED_MASK_INDEX = 10


def _fixture_dir(scale):
    return get_abs_path(__file__, f"fixtures/{scale}")


def _load_fixture_inputs(scale):
    fixture_dir = _fixture_dir(scale)
    with open(f"{fixture_dir}/study_df.pkl", "rb") as f:
        study_df = pickle.load(f)
    with open(f"{fixture_dir}/pi_args.pkl", "rb") as f:
        action_prob_func_args = pickle.load(f)
    with open(f"{fixture_dir}/rl_update_args.pkl", "rb") as f:
        alg_update_func_args = pickle.load(f)
    return study_df, action_prob_func_args, alg_update_func_args


def _run(tmp_path, scale, variant, caplog):
    """
    variant is one of "baseline", "masked", "combined", "auto" -- see this
    module's own docstring.
    """
    study_df, action_prob_func_args, alg_update_func_args = _load_fixture_inputs(scale)

    action_prob_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_get_action_1_prob_generalized_logistic.py"
    )
    inference_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_get_least_squares_loss_inference_no_action_centering.py"
    )
    theta_calculation_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_estimate_theta_least_squares_no_action_centering.py"
    )

    if variant == "baseline":
        alg_update_func = load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/RL_least_squares_loss_regularized.py"
        )
        mask_kwargs = {}
    else:
        alg_update_func = load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/RL_least_squares_loss_regularized_masked.py"
        )
        mask_kwargs = {
            "alg_update_func_args_mask_index": MASKED_MASK_INDEX,
            "alg_update_func_args_ragged_indices": MASKED_RAGGED_INDICES,
        }
        if variant != "auto":
            # "auto" deliberately leaves combine_updates_into_one_vmap unset
            # (the None default), exercising the auto resolution end to end.
            mask_kwargs["combine_updates_into_one_vmap"] = variant == "combined"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        start = time.perf_counter()
        result = analyze_dataset(
            output_dir=tmp_path,
            analysis_df=study_df,
            action_prob_func=action_prob_func,
            action_prob_func_args=action_prob_func_args,
            action_prob_func_args_beta_index=0,
            alg_update_func=alg_update_func,
            alg_update_func_type="loss",
            alg_update_func_args=alg_update_func_args,
            alg_update_func_args_beta_index=0,
            alg_update_func_args_action_prob_index=5,
            alg_update_func_args_action_prob_times_index=6,
            alg_update_func_args_previous_betas_index=-1,
            inference_func=inference_func,
            inference_func_type="loss",
            inference_func_args_theta_index=0,
            theta_calculation_func=theta_calculation_func,
            active_col_name="in_study",
            action_col_name="action",
            policy_num_col_name="policy_num",
            calendar_t_col_name="calendar_t",
            subject_id_col_name="user_id",
            action_prob_col_name="action1prob",
            reward_col_name="reward",
            suppress_interactive_data_checks=True,
            suppress_all_data_checks=True,
            form_adjusted_meat_adjustments_explicitly=False,
            **mask_kwargs,
        )
        total_seconds = time.perf_counter() - start

    phase_timings = [
        (match.group("phase"), float(match.group("seconds")))
        for record in caplog.records
        for match in [PHASE_LOG_PATTERN.match(record.message)]
        if match
    ]
    peak_rss = [
        (match.group("phase"), float(match.group("mb")))
        for record in caplog.records
        for match in [PEAK_RSS_PATTERN.match(record.message)]
        if match
    ]
    return result, total_seconds, phase_timings, peak_rss


def _print_report(label, total_seconds, phase_timings, peak_rss):
    print(f"\n--- combine_updates_into_one_vmap benchmark: {label} ---")
    print(f"Total wall-clock time: {total_seconds:.3f}s")
    for phase, seconds in sorted(phase_timings, key=lambda p: -p[1]):
        if "jacrev" in phase or "batched_components" in phase:
            print(f"  {seconds:8.3f}s  {phase}")
    for phase, mb in peak_rss:
        print(f"  {mb:8.1f} MB  peak RSS after {phase}")


@pytest.mark.parametrize(
    "scale", ["small", pytest.param("medium", marks=pytest.mark.slow)]
)
def test_combine_updates_into_one_vmap_matches_golden_and_baseline(
    tmp_path, caplog, scale
):
    """
    All three variants (baseline / masked-uncombined / masked-combined) must
    agree with the existing golden fixture to float32 noise, on the exact
    same real, genuinely-ragged/staggered-recruitment fixture data -- this is
    both the "opted-in result matches golden" check and the direct
    old-path-vs-new-path comparison on the same real input.
    """
    with open(f"{_fixture_dir(scale)}/golden_analysis.pkl", "rb") as f:
        golden = pickle.load(f)

    results = {}
    timings = {}
    for variant in ("baseline", "masked", "combined", "auto"):
        result, total_seconds, phase_timings, peak_rss = _run(
            tmp_path, scale, variant, caplog
        )
        if variant == "auto":
            # The auto default must have RESOLVED to the combined path on
            # this eligible (masked, no previous_betas) fixture -- not merely
            # produced matching numbers.
            assert any(
                "combine_updates_into_one_vmap=None (auto) resolved to True"
                in record.message
                for record in caplog.records
            ), "auto variant did not log an auto-resolution to True"
        results[variant] = result
        timings[variant] = (total_seconds, phase_timings, peak_rss)
        _print_report(
            f"scale={scale}, variant={variant}", total_seconds, phase_timings, peak_rss
        )

        np.testing.assert_allclose(
            np.asarray(result["theta_est"]),
            np.asarray(golden["theta_est"]),
            rtol=1e-5,
            err_msg=f"theta_est diverged from golden for variant={variant!r}.",
        )
        np.testing.assert_allclose(
            np.asarray(result["adjusted_sandwich_var_estimate"]),
            np.asarray(golden["adjusted_sandwich_var_estimate"]),
            rtol=1e-5,
            err_msg=(
                "adjusted_sandwich_var_estimate diverged from golden for "
                f"variant={variant!r}."
            ),
        )
        np.testing.assert_allclose(
            np.asarray(result["classical_sandwich_var_estimate"]),
            np.asarray(golden["classical_sandwich_var_estimate"]),
            rtol=1e-5,
            err_msg=(
                "classical_sandwich_var_estimate diverged from golden for "
                f"variant={variant!r}."
            ),
        )

    # Direct pairwise comparison too (not just "both match golden" --
    # constraint 2 of the task this was built for asks for a DIRECT
    # old-path-vs-new-path comparison on the same real input).
    for key in (
        "theta_est",
        "adjusted_sandwich_var_estimate",
        "classical_sandwich_var_estimate",
    ):
        np.testing.assert_allclose(
            np.asarray(results["baseline"][key]),
            np.asarray(results["combined"][key]),
            rtol=1e-5,
            err_msg=f"{key} differs between baseline and combined variants.",
        )
        # The auto default resolves to the exact same combined code path as
        # an explicit True on this fixture, so the two runs must agree
        # bitwise, not just to tolerance.
        np.testing.assert_array_equal(
            np.asarray(results["combined"][key]),
            np.asarray(results["auto"][key]),
            err_msg=f"{key} differs between explicit-combined and auto variants.",
        )
