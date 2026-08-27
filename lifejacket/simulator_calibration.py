from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.stats

from .constants import DiagnosticClassifications

logger = logging.getLogger(__name__)


def clopper_pearson_upper_bound(
    num_failures: int, num_trials: int, confidence_level: float = 0.95
) -> float:
    """
    One-sided Clopper-Pearson upper confidence bound on a failure probability given
    `num_failures` observed out of `num_trials` independent trials.

    For num_failures == 0, this reduces exactly to `1 - alpha**(1/num_trials)` with
    `alpha = 1 - confidence_level` (e.g. 0 failures in 59 directions bounds the failure
    probability below ~5% at 95% confidence; 0 failures in 299 directions bounds it below ~1%).
    """
    if num_trials <= 0:
        return math.nan
    if num_failures >= num_trials:
        return 1.0
    if num_failures < 0:
        raise ValueError("num_failures must be non-negative.")
    return float(
        scipy.stats.beta.ppf(
            confidence_level, num_failures + 1, num_trials - num_failures
        )
    )


###############################################################################
# Section 11: simulator-agnostic calibration interface.
#
# This module does not know about, or invent, any particular deployment/environment model. It
# only defines the shape of a "deployment replay" and a calibration/classification procedure
# that a caller-supplied simulator can be plugged into. See tests/integration_tests for a
# concrete (small-scale, non-production) adapter built on this repository's own synthetic RL
# simulator under tests/simulators_and_runners.
###############################################################################


@dataclasses.dataclass
class DeploymentReplay:
    """
    Everything needed to run the diagnostic suite on one simulated deployment, plus whatever
    ground truth is available to judge inferential failure (e.g. bias/coverage against a known
    generative theta*). `diagnostic_kwargs` is unpacked directly into
    lifejacket.diagnostics.run_diagnostic_suite.
    """

    diagnostic_kwargs: dict[str, Any]
    theta_hat: np.ndarray
    theta_variance_estimate: np.ndarray
    ground_truth_theta: np.ndarray | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CalibrationResult:
    classification: str
    num_holdout_replays: int
    num_holdout_passed_diagnostics: int
    num_inferential_failures_among_passed: int
    conditional_failure_rate_upper_bound: float
    risk_tolerance: float
    confidence_level: float
    per_replay_records: list[dict[str, Any]]


def default_failure_predicate(
    replay: DeploymentReplay,
    se_distortion_tolerance: float = 0.05,
    coverage_tolerance: float = 0.05,
    nominal_coverage: float = 0.95,
) -> bool:
    """
    A minimal, explicitly non-authoritative example failure predicate: flags nonfinite/extreme
    variance estimates, or (when ground truth is available) a contrast falling outside its
    nominal-coverage interval more often than tolerance allows. Callers should generally supply
    their own predicate matched to what "inferential failure" means for their deployment.
    """
    variance = np.asarray(replay.theta_variance_estimate, dtype=np.float64)
    if not np.all(np.isfinite(variance)) or np.any(np.diag(variance) < 0):
        return True
    if replay.ground_truth_theta is None:
        return False
    theta_hat = np.asarray(replay.theta_hat, dtype=np.float64)
    ground_truth = np.asarray(replay.ground_truth_theta, dtype=np.float64)
    se = np.sqrt(np.clip(np.diag(variance), 0.0, None))
    z = scipy.stats.norm.ppf(1 - (1 - nominal_coverage) / 2)
    covered = np.abs(theta_hat - ground_truth) <= z * se
    return bool(np.mean(~covered) > coverage_tolerance)


def calibrate_and_classify(
    replay_fn: Callable[[int], DeploymentReplay],
    train_seeds: Sequence[int],
    holdout_seeds: Sequence[int],
    config: Any,
    failure_predicate: Callable[[DeploymentReplay, Any], bool] | None = None,
    risk_tolerance: float = 0.05,
    confidence_level: float = 0.95,
) -> CalibrationResult:
    """
    Runs the diagnostic suite (lifejacket.diagnostics.run_diagnostic_suite) on every replay
    produced by `replay_fn` for `train_seeds` (available for the caller's own threshold tuning --
    not used by this function beyond running the suite once per seed) and every replay for
    `holdout_seeds`. Among held-out replays whose diagnostics pass (classification is not
    `failed`/`indeterminate`), computes `P(inferential failure | diagnostics pass)` via
    `failure_predicate` and its one-sided Clopper-Pearson upper confidence bound.

    Returns `DiagnosticClassifications.SUPPORTED` only when that upper bound is below
    `risk_tolerance`; otherwise `DiagnosticClassifications.LOCALLY_SUPPORTED`. This is the ONLY
    place in the package that classification can ever be `SUPPORTED`.
    """
    from . import (
        diagnostics as diagnostics_module,  # deferred: avoids a circular import
    )

    failure_predicate = failure_predicate or default_failure_predicate

    train_records = []
    for seed in train_seeds:
        replay = replay_fn(seed)
        report = diagnostics_module.run_diagnostic_suite(
            config=config, **replay.diagnostic_kwargs
        )
        train_records.append({"seed": seed, "classification": report.classification})

    holdout_records = []
    num_passed = 0
    num_failures_among_passed = 0
    for seed in holdout_seeds:
        replay = replay_fn(seed)
        report = diagnostics_module.run_diagnostic_suite(
            config=config, **replay.diagnostic_kwargs
        )
        diagnostics_passed = report.classification not in (
            DiagnosticClassifications.FAILED,
            DiagnosticClassifications.INDETERMINATE,
        )
        inferential_failure = None
        if diagnostics_passed:
            num_passed += 1
            inferential_failure = bool(failure_predicate(replay, config))
            if inferential_failure:
                num_failures_among_passed += 1
        holdout_records.append(
            {
                "seed": seed,
                "classification": report.classification,
                "diagnostics_passed": diagnostics_passed,
                "inferential_failure": inferential_failure,
            }
        )

    upper_bound = clopper_pearson_upper_bound(
        num_failures_among_passed, num_passed, confidence_level
    )

    if num_passed > 0 and upper_bound < risk_tolerance:
        classification = DiagnosticClassifications.SUPPORTED
    else:
        classification = DiagnosticClassifications.LOCALLY_SUPPORTED

    logger.info(
        "Calibration: %d/%d held-out replays passed diagnostics; %d inferential failures among "
        "those; P(failure|pass) upper bound %.4g (risk tolerance %.4g) -> %s.",
        num_passed,
        len(holdout_seeds),
        num_failures_among_passed,
        upper_bound,
        risk_tolerance,
        classification,
    )

    return CalibrationResult(
        classification=classification,
        num_holdout_replays=len(holdout_seeds),
        num_holdout_passed_diagnostics=num_passed,
        num_inferential_failures_among_passed=num_failures_among_passed,
        conditional_failure_rate_upper_bound=upper_bound,
        risk_tolerance=risk_tolerance,
        confidence_level=confidence_level,
        per_replay_records=train_records + holdout_records,
    )
