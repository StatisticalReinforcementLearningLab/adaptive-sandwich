import functools
import inspect
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
    seed,
    num_subjects=50,
    theta_dim=1,
    ground_truth_theta=0.0,
    inject_failure=False,
    theta_hat_noise_scale=0.1,
):
    # theta_hat_noise_scale controls how far theta_hat lands from ground truth relative to its own
    # sandwich SE (~0.07 at the default num_subjects). At the default 0.1 the toy replays are
    # genuinely under-covering, so the default predicate flags a realistic minority of them; shrink
    # it to make coverage certain for every seed.
    rng = np.random.default_rng(seed)
    b_true = np.array([[2.0]])
    theta_hat = np.array([ground_truth_theta]) + rng.normal(
        scale=theta_hat_noise_scale, size=theta_dim
    )
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


def _make_stub_replay(
    seed, theta_hat=(0.0,), variance_diagonal=(1.0,), ground_truth_theta=(0.0,)
):
    """
    A replay that carries no simulation at all: its `diagnostic_kwargs` is nothing but the seed,
    which is what `_install_stub_diagnostic_suite`'s stand-in suite reads. Only the three fields a
    failure predicate looks at are real, so certification arithmetic can be exercised over a
    hundred held-out replays without a hundred real diagnostic suites.
    """
    return simulator_calibration.DeploymentReplay(
        diagnostic_kwargs={"seed": seed},
        theta_hat=np.asarray(theta_hat, dtype=float),
        theta_variance_estimate=np.diag(np.asarray(variance_diagonal, dtype=float)),
        ground_truth_theta=(
            None
            if ground_truth_theta is None
            else np.asarray(ground_truth_theta, dtype=float)
        ),
    )


def _install_stub_diagnostic_suite(monkeypatch, classification_overrides=None):
    """
    Replaces run_diagnostic_suite with a stand-in that reports a fixed classification per seed
    (`locally_supported` unless overridden). calibrate_and_classify imports the diagnostics module
    lazily and resolves the suite off the module object at call time, so patching the attribute is
    enough to intercept it.
    """
    from lifejacket import diagnostics

    classification_overrides = classification_overrides or {}

    def stub_run_diagnostic_suite(*, config, seed):
        return diagnostics.DiagnosticReport(
            classification=classification_overrides.get(
                seed, DiagnosticClassifications.LOCALLY_SUPPORTED
            ),
            check_results={},
            input_check_results={},
            metrics={},
            tolerances_used={},
            warnings=[],
            monte_carlo_counts={},
            target_labels=[],
            rank_diagnostics={},
        )

    monkeypatch.setattr(diagnostics, "run_diagnostic_suite", stub_run_diagnostic_suite)


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


@pytest.mark.parametrize(
    "seeds_failing_inference, expected_bound, expected_classification",
    [
        # The one-sided 95% Clopper-Pearson upper bounds for 1 and 2 failures in 100 trials: the p
        # solving sum_{k<=x} C(100,k) p^k (1-p)^(100-k) = 0.05. Written out rather than recomputed
        # with clopper_pearson_upper_bound so a miscounted denominator cannot satisfy this
        # assertion tautologically by feeding the same wrong counts to both sides.
        ((0,), 0.04655981, DiagnosticClassifications.SUPPORTED),
        ((0, 1), 0.06161920, DiagnosticClassifications.LOCALLY_SUPPORTED),
    ],
)
def test_calibrate_and_classify_certifies_supported_only_when_the_bound_clears_tolerance(
    monkeypatch, seeds_failing_inference, expected_bound, expected_classification
):
    from lifejacket.diagnostics import DiagnosticConfig

    # Seeds 100-104 fail their diagnostics and 105-109 come back indeterminate; neither group may
    # enter the conditional-failure-rate denominator, so it is exactly the 100 seeds 0-99.
    classification_overrides = {
        seed: DiagnosticClassifications.FAILED for seed in range(100, 105)
    }
    classification_overrides.update(
        {seed: DiagnosticClassifications.INDETERMINATE for seed in range(105, 110)}
    )
    _install_stub_diagnostic_suite(
        monkeypatch, classification_overrides=classification_overrides
    )

    result = simulator_calibration.calibrate_and_classify(
        _make_stub_replay,
        train_seeds=[],
        holdout_seeds=range(110),
        config=DiagnosticConfig(compute_influence_and_overlap_checks=False),
        failure_predicate=lambda replay, _config: (
            replay.diagnostic_kwargs["seed"] in seeds_failing_inference
        ),
        risk_tolerance=0.05,
    )

    assert result.num_holdout_replays == 110
    assert result.num_holdout_passed_diagnostics == 100
    assert result.num_inferential_failures_among_passed == len(seeds_failing_inference)
    assert result.conditional_failure_rate_upper_bound == pytest.approx(expected_bound)
    assert (expected_bound < result.risk_tolerance) == (
        expected_classification == DiagnosticClassifications.SUPPORTED
    )
    assert result.classification == expected_classification


def test_calibrate_and_classify_certifies_supported_end_to_end_on_clean_replays():
    from lifejacket.diagnostics import DiagnosticConfig

    def replay_fn(seed):
        # theta_hat sits essentially on top of the ground truth, so the shipped default predicate
        # finds every held-out replay covered and records no inferential failure.
        return _make_toy_replay(seed, theta_hat_noise_scale=1e-3)

    result = simulator_calibration.calibrate_and_classify(
        replay_fn,
        train_seeds=[],
        holdout_seeds=range(10),
        # num_directions is trimmed purely to keep ten real diagnostic suites fast; nothing about
        # the certification arithmetic under test depends on it.
        config=DiagnosticConfig(
            compute_influence_and_overlap_checks=False, num_directions=5
        ),
        risk_tolerance=0.30,
    )

    assert result.num_holdout_passed_diagnostics == 10
    assert result.num_inferential_failures_among_passed == 0
    # 0 failures in 10 replays bounds the conditional failure rate below ~25.9%, which clears a
    # 30% risk tolerance -- the one branch in the package that can return SUPPORTED.
    assert result.conditional_failure_rate_upper_bound == pytest.approx(
        1 - 0.05 ** (1 / 10)
    )
    assert result.classification == DiagnosticClassifications.SUPPORTED


def test_calibrate_and_classify_invokes_predicate_with_replay_and_config_positionally(
    monkeypatch,
):
    from lifejacket.diagnostics import DiagnosticConfig

    _install_stub_diagnostic_suite(
        monkeypatch,
        classification_overrides={2: DiagnosticClassifications.FAILED},
    )
    config = DiagnosticConfig(compute_influence_and_overlap_checks=False)
    calls = []

    def recording_predicate(*args, **kwargs):
        calls.append((args, kwargs))
        return False

    simulator_calibration.calibrate_and_classify(
        _make_stub_replay,
        train_seeds=[],
        holdout_seeds=range(4),
        config=config,
        failure_predicate=recording_predicate,
    )

    # The contract default_failure_predicate's docstring fixes: exactly two positional arguments
    # and no keywords, the second being the very DiagnosticConfig the suite was run with. A
    # predicate written to that signature must never find a tolerance in the config's slot.
    assert [kwargs for _args, kwargs in calls] == [{}, {}, {}]
    assert [len(args) for args, _kwargs in calls] == [2, 2, 2]
    assert all(args[1] is config for args, _kwargs in calls)
    # Seed 2's diagnostics failed, so it is never judged for inferential failure at all.
    assert [args[0].diagnostic_kwargs["seed"] for args, _kwargs in calls] == [0, 1, 3]


def test_default_failure_predicate_signature_pins_the_positional_argument_contract():
    parameters = inspect.signature(
        simulator_calibration.default_failure_predicate
    ).parameters

    assert list(parameters) == [
        "replay",
        "config",
        "coverage_tolerance",
        "nominal_coverage",
    ]
    # The removed se_distortion_tolerance knob was never read, yet it occupied the slot
    # calibrate_and_classify passes the DiagnosticConfig into: a predicate copying this signature
    # bound the config to something named (and defaulted) like a float tolerance.
    assert "se_distortion_tolerance" not in parameters
    assert parameters["config"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["config"].default is None
    for name in ("coverage_tolerance", "nominal_coverage"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    # No tolerance may reoccupy a positional slot, so binding one positionally is an error rather
    # than a silent reinterpretation of the argument list.
    with pytest.raises(TypeError):
        simulator_calibration.default_failure_predicate(
            _make_stub_replay(0), None, 0.05
        )


def test_default_failure_predicate_verdict_does_not_depend_on_the_config_it_receives():
    replay = _make_stub_replay(0, theta_hat=(0.5,), variance_diagonal=(1.0,))
    standalone = simulator_calibration.default_failure_predicate(replay)

    assert standalone is False
    assert simulator_calibration.default_failure_predicate(replay, None) == standalone
    assert (
        simulator_calibration.default_failure_predicate(replay, object()) == standalone
    )


def test_partial_bound_coverage_tolerance_changes_the_verdict_through_calibrate(
    monkeypatch,
):
    from lifejacket.diagnostics import DiagnosticConfig

    _install_stub_diagnostic_suite(monkeypatch)
    config = DiagnosticConfig(compute_influence_and_overlap_checks=False)

    def replay_fn(seed):
        # Four-component theta whose last component sits five SEs from the truth: a within-replay
        # non-coverage fraction of exactly 0.25.
        return _make_stub_replay(
            seed,
            theta_hat=(0.0, 0.0, 0.0, 5.0),
            variance_diagonal=(1.0, 1.0, 1.0, 1.0),
            ground_truth_theta=(0.0, 0.0, 0.0, 0.0),
        )

    def calibrate_with(predicate):
        return simulator_calibration.calibrate_and_classify(
            replay_fn,
            train_seeds=[],
            holdout_seeds=range(4),
            config=config,
            failure_predicate=predicate,
        )

    strict = calibrate_with(simulator_calibration.default_failure_predicate)
    # functools.partial is the documented way to override the keyword-only tolerances while
    # keeping the required two-positional-argument surface.
    loose = calibrate_with(
        functools.partial(
            simulator_calibration.default_failure_predicate, coverage_tolerance=0.5
        )
    )

    assert strict.num_inferential_failures_among_passed == 4
    assert loose.num_inferential_failures_among_passed == 0


@pytest.mark.parametrize("variance_diagonal", [(np.nan,), (np.inf,), (-1.0,)])
def test_default_failure_predicate_flags_unusable_variance_without_ground_truth(
    variance_diagonal,
):
    replay = _make_stub_replay(
        0, variance_diagonal=variance_diagonal, ground_truth_theta=None
    )
    assert simulator_calibration.default_failure_predicate(replay, None) is True


def test_default_failure_predicate_passes_a_usable_variance_without_ground_truth():
    replay = _make_stub_replay(0, variance_diagonal=(1.0,), ground_truth_theta=None)
    assert simulator_calibration.default_failure_predicate(replay, None) is False
