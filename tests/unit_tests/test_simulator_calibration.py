import math

import numpy as np
import pytest

from lifejacket import simulator_calibration
from lifejacket.constants import DiagnosticClassifications


def test_clopper_pearson_zero_failures_59_directions_matches_closed_form():
    # From the task write-up: 0 failures in 59 directions bounds the failure probability below
    # approximately 5% at 95% confidence, via the closed form 1 - alpha**(1/J).
    bound = simulator_calibration.clopper_pearson_upper_bound(
        0, 59, confidence_level=0.95
    )
    closed_form = 1 - 0.05 ** (1 / 59)
    assert math.isclose(bound, closed_form, rel_tol=1e-9)
    assert math.isclose(bound, 0.05, abs_tol=0.005)


def test_clopper_pearson_zero_failures_299_directions_matches_closed_form():
    # 0 failures in 299 directions bounds the failure probability below approximately 1%.
    bound = simulator_calibration.clopper_pearson_upper_bound(
        0, 299, confidence_level=0.95
    )
    closed_form = 1 - 0.05 ** (1 / 299)
    assert math.isclose(bound, closed_form, rel_tol=1e-9)
    assert math.isclose(bound, 0.01, abs_tol=0.002)


def test_clopper_pearson_all_failures_returns_one():
    assert (
        simulator_calibration.clopper_pearson_upper_bound(10, 10, confidence_level=0.95)
        == 1.0
    )


def test_clopper_pearson_zero_trials_returns_nan():
    assert math.isnan(
        simulator_calibration.clopper_pearson_upper_bound(0, 0, confidence_level=0.95)
    )


def test_clopper_pearson_rejects_negative_failures():
    with pytest.raises(ValueError):
        simulator_calibration.clopper_pearson_upper_bound(-1, 10, confidence_level=0.95)


def test_clopper_pearson_bound_increases_with_more_observed_failures():
    lower = simulator_calibration.clopper_pearson_upper_bound(
        0, 100, confidence_level=0.95
    )
    higher = simulator_calibration.clopper_pearson_upper_bound(
        5, 100, confidence_level=0.95
    )
    assert higher > lower


def _make_toy_replay(
    seed, num_subjects=50, theta_dim=1, ground_truth_theta=0.0, inject_failure=False
):
    rng = np.random.default_rng(seed)
    b_true = np.array([[2.0]])
    theta_hat = np.array([ground_truth_theta]) + rng.normal(scale=0.1, size=theta_dim)
    per_subject_stacks = rng.normal(size=(num_subjects, theta_dim))
    per_subject_stacks = per_subject_stacks - per_subject_stacks.mean(axis=0)
    m_hat = (per_subject_stacks.T @ per_subject_stacks) / num_subjects
    joint_sandwich = (
        np.linalg.solve(b_true, m_hat) @ np.linalg.inv(b_true).T / num_subjects
    )
    if inject_failure:
        # Artificially shrink the variance estimate so the resulting CI is too tight to reliably
        # cover ground_truth_theta given the actual dispersion of theta_hat around it -- this is
        # what "inferential failure" (under-coverage) looks like to the default predicate.
        joint_sandwich = joint_sandwich * 1e-6

    def g_tilde(eta):
        return b_true @ (eta - theta_hat)

    diagnostic_kwargs = dict(
        g_tilde=g_tilde,
        eta_hat=theta_hat,
        B_hat=b_true,
        M_hat=m_hat,
        joint_sandwich_matrix=joint_sandwich,
        per_subject_stacks=per_subject_stacks,
        beta_dim=0,
        theta_dim=theta_dim,
        num_subjects=num_subjects,
    )
    return simulator_calibration.DeploymentReplay(
        diagnostic_kwargs=diagnostic_kwargs,
        theta_hat=theta_hat,
        theta_variance_estimate=joint_sandwich,
        ground_truth_theta=np.array([ground_truth_theta]),
    )


def test_calibrate_and_classify_returns_locally_supported_without_enough_evidence():
    from lifejacket.diagnostics import DiagnosticConfig

    def replay_fn(seed):
        return _make_toy_replay(seed)

    result = simulator_calibration.calibrate_and_classify(
        replay_fn,
        train_seeds=range(3),
        holdout_seeds=range(3, 8),
        config=DiagnosticConfig(compute_influence_and_overlap_checks=False),
        risk_tolerance=0.05,
    )
    # Five holdout replays can never certify a <5% conditional failure rate.
    assert result.classification == DiagnosticClassifications.LOCALLY_SUPPORTED
    assert result.num_holdout_replays == 5


def test_calibrate_and_classify_flags_failures_via_predicate():
    from lifejacket.diagnostics import DiagnosticConfig

    def replay_fn(seed):
        return _make_toy_replay(seed, inject_failure=(seed % 2 == 0))

    result = simulator_calibration.calibrate_and_classify(
        replay_fn,
        train_seeds=[],
        holdout_seeds=range(10),
        config=DiagnosticConfig(compute_influence_and_overlap_checks=False),
        failure_predicate=simulator_calibration.default_failure_predicate,
    )
    assert result.num_inferential_failures_among_passed >= 1
    assert result.classification != DiagnosticClassifications.SUPPORTED
