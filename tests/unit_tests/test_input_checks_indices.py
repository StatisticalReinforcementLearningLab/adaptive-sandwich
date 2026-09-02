"""
Unit tests for the structural/wiring checks in lifejacket.input_checks that validate the
SUPPLIED FUNCTIONS and the ARGUMENT INDICES that address their argument tuples:

    require_valid_function_types
    _get_positional_parameter_bounds
    _describe_positional_bounds
    _get_declared_positional_parameter_count
    require_arg_tuple_lengths_consistent_and_callable
    require_arg_indices_supplied
    require_arg_indices_in_range
    require_arg_indices_distinct
    require_mask_index_appends_after_supplied_args
    require_ragged_indices_valid

The signature-facing half of this file is deliberately PERMISSIVE about wrapper shapes. The
production argument-batching code derives the argument count from the DATA, not from the
signature (vmap_helpers.build_batched_arg_lists_by_subject says so in its own docstring), and
both a `def w(beta_est, *rest)` forwarding shim and a function carrying an extra defaulted
hyperparameter were verified end to end on this repo's fixture to produce estimates
BIT-IDENTICAL to the unwrapped function. So the tests below pin that those are ACCEPTED, and
that what is actually enforced is (a) every supplied tuple having the same width and (b) the
function being callable with that many positional arguments.

None of these checks touch analysis_df or call confirm_input_check_result, so none of them can
block on builtins.input.
"""

import inspect
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lifejacket import input_checks
from lifejacket.constants import FunctionTypes

### Module-level helper builders. Suffixed with _indices so this file can be concatenated with
### the other input-check test modules without colliding.


def _alg_update_func_indices(beta, previous_betas, action_probs, action_prob_times):
    """
    A mask-free algorithm update function: FOUR declared parameters, so callers must supply
    four-entry argument tuples.
    """
    return (
        jnp.sum(beta)
        + jnp.sum(previous_betas)
        + jnp.sum(action_probs)
        + jnp.sum(action_prob_times)
    )


def _mask_aware_alg_update_func_indices(beta, rewards, action_probs, validity_mask):
    """
    A mask-aware algorithm update function: FOUR declared parameters, but callers supply only
    THREE, because self_pad_ragged_args_and_build_mask appends the validity mask as a new last
    argument (mask_index == 3).
    """
    return jnp.sum(beta) + jnp.sum(rewards * validity_mask) + jnp.sum(action_probs)


def _action_prob_func_indices(beta, features):
    """Two declared parameters; the shape action_prob_func_args tuples must match."""
    return jnp.dot(beta, features)


def _forwarding_shim_func_indices(beta_est, *rest):
    """
    The forwarding shim shape the adversarial review vindicated: it names only the parameter it
    touches and passes everything else straight through, so its positional maximum is
    math.inf and any supplied tuple width is callable.
    """
    return jnp.sum(beta_est) + sum(jnp.sum(argument) for argument in rest)


def _required_pair_then_variadic_func_indices(beta, features, *rest):
    """A shim with TWO required parameters, so its unbounded maximum still has a floor of 2."""
    return jnp.dot(beta, features)


def _defaulted_hyperparameter_func_indices(
    beta, previous_betas, action_probs, n, ridge=0.0
):
    """
    The other vindicated wrapper shape: FIVE declared parameters, the last of them defaulted,
    so four-entry supplied tuples are callable and five-entry ones are too.
    """
    return (
        jnp.sum(beta)
        + jnp.sum(previous_betas)
        + jnp.sum(action_probs)
        + n * (1.0 + ridge)
    )


def _positional_only_func_indices(beta, features, /):
    """Positional-only parameters, which count toward the positional bounds like any other."""
    return jnp.dot(beta, features)


def _variadic_keyword_func_indices(beta, **options):
    return jnp.sum(beta)


def _doubly_variadic_func_indices(beta, *extra_args, **options):
    return jnp.sum(beta)


def _keyword_only_func_indices(beta, features, *, scale):
    return jnp.dot(beta, features) * scale


def _no_parameter_func_indices():
    return jnp.array(0.0)


def _build_supplied_args_indices(
    *,
    tuple_length=4,
    keys=(2, 3),
    subject_ids=(0, 1),
    blank_cells=(),
    length_overrides=None,
):
    """
    A small valid alg_update_func_args-shaped mapping: {policy_num: {subject_id: args_tuple}}.

    Policy numbers deliberately start at 2, matching the real fixtures where the INITIAL policy
    number is 1 and never appears in alg_update_func_args, so nothing here may assume a
    zero-based first key.

    blank_cells: (key, subject_id) pairs handed the blank () tuple, which the arity check must
        skip ("not applicable at this update").
    length_overrides: (key, subject_id) -> tuple length, to introduce one wrong-length tuple.
    """
    length_overrides = dict(length_overrides or {})
    args_by_subject_id_by_key = {}
    for key in keys:
        args_by_subject_id_by_key[key] = {}
        for subject_id in subject_ids:
            if (key, subject_id) in blank_cells:
                args_by_subject_id_by_key[key][subject_id] = ()
                continue
            length = length_overrides.get((key, subject_id), tuple_length)
            args_by_subject_id_by_key[key][subject_id] = tuple(
                jnp.arange(3.0) + position for position in range(length)
            )
    return args_by_subject_id_by_key


def _build_alg_update_indices_by_name_indices(
    *,
    beta_index=0,
    previous_betas_index=1,
    action_prob_index=2,
    action_prob_times_index=3,
):
    """
    The four alg_update_func argument indices, wired exactly as
    perform_first_wave_input_checks builds them, defaulting to the valid distinct-and-in-range
    assignment for a four-entry supplied tuple.
    """
    return {
        "alg_update_func_args_beta_index": beta_index,
        "alg_update_func_args_previous_betas_index": previous_betas_index,
        "alg_update_func_args_action_prob_index": action_prob_index,
        "alg_update_func_args_action_prob_times_index": action_prob_times_index,
    }


def _build_shared_indices_by_name_indices(*, beta_index=0, previous_betas_index=1):
    """
    The subset of indices require_ragged_indices_valid treats as SHARED across subjects, wired
    the way perform_first_wave_input_checks passes them.
    """
    return {
        "alg_update_func_args_beta_index": beta_index,
        "alg_update_func_args_previous_betas_index": previous_betas_index,
    }


### require_valid_function_types


def test_require_valid_function_types_passes_for_recognized_types_indices():
    """Both FunctionTypes members are accepted, in either slot."""
    input_checks.require_valid_function_types(
        FunctionTypes.LOSS, FunctionTypes.ESTIMATING
    )
    input_checks.require_valid_function_types(
        FunctionTypes.ESTIMATING, FunctionTypes.LOSS
    )
    input_checks.require_valid_function_types(FunctionTypes.LOSS, FunctionTypes.LOSS)
    input_checks.require_valid_function_types(
        FunctionTypes.ESTIMATING, FunctionTypes.ESTIMATING
    )


def test_require_valid_function_types_rejects_capitalized_alg_update_type_indices():
    """
    A near-miss typo in alg_update_func_type ("Loss" for "loss") must be caught here by name,
    not as a bare "Unknown update function type." from inside the derivative precompute.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func_type='Loss' is not a recognized function type; it must be one of "
            "['loss', 'estimating']"
        ),
    ):
        input_checks.require_valid_function_types("Loss", FunctionTypes.ESTIMATING)


def test_require_valid_function_types_rejects_bad_inference_type_indices():
    """The inference slot is checked too, and the message names that slot rather than the other."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "inference_func_type='estimating_function' is not a recognized"
        ),
    ):
        input_checks.require_valid_function_types(
            FunctionTypes.LOSS, "estimating_function"
        )


def test_require_valid_function_types_rejects_none_indices():
    """An unset (None) function type is not silently treated as a default."""
    with pytest.raises(
        AssertionError, match=re.escape("alg_update_func_type=None is not a recognized")
    ):
        input_checks.require_valid_function_types(None, FunctionTypes.LOSS)


def test_require_valid_function_types_checks_alg_update_slot_first_indices():
    """
    With both slots wrong the alg_update slot is reported, pinning the loop order so the
    message is deterministic rather than dict/argument-order dependent.
    """
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_valid_function_types("bogus_alg", "bogus_inference")
    assert "alg_update_func_type='bogus_alg'" in str(excinfo.value)
    assert "bogus_inference" not in str(excinfo.value)


### _get_positional_parameter_bounds


def test_get_positional_parameter_bounds_counts_required_positionals_indices():
    """The plain case: four required positional parameters bound the call at exactly four."""
    assert input_checks._get_positional_parameter_bounds(
        _alg_update_func_indices, "alg_update_func"
    ) == (4, 4)
    assert input_checks._get_positional_parameter_bounds(
        _action_prob_func_indices, "action_prob_func"
    ) == (2, 2)


def test_get_positional_parameter_bounds_accepts_var_positional_shim_indices():
    """
    The headline case the adversarial review established: a forwarding shim declaring *args is
    ACCEPTED, with an UNBOUNDED maximum, rather than rejected as it used to be. The production
    batching code takes its argument count from the DATA, not the signature (see
    vmap_helpers.build_batched_arg_lists_by_subject), and `def w(beta_est, *rest)` was verified
    on this repo's own fixture to return estimates bit-identical to the function it forwards to.
    """
    assert input_checks._get_positional_parameter_bounds(
        _forwarding_shim_func_indices, "alg_update_func"
    ) == (1, math.inf)


def test_get_positional_parameter_bounds_var_positional_does_not_raise_the_minimum_indices():
    """
    *args itself is optional, so it contributes to the maximum only: a two-required-parameter
    shim still has a minimum of two, not three.
    """
    assert input_checks._get_positional_parameter_bounds(
        _required_pair_then_variadic_func_indices, "alg_update_func"
    ) == (2, math.inf)


def test_get_positional_parameter_bounds_counts_defaulted_parameter_in_maximum_only_indices():
    """
    A trailing DEFAULTED hyperparameter widens the maximum without raising the minimum, which
    is what lets tuples one entry shorter than the declared count through -- the second wrapper
    shape the review proved bit-identical to the unwrapped function.
    """
    assert input_checks._get_positional_parameter_bounds(
        _defaulted_hyperparameter_func_indices, "alg_update_func"
    ) == (4, 5)


def test_get_positional_parameter_bounds_ignores_var_keyword_indices():
    """
    **kwargs accepts no POSITIONAL argument, so it leaves the bounds exactly where the declared
    positionals put them (it used to be rejected outright).
    """
    assert input_checks._get_positional_parameter_bounds(
        _variadic_keyword_func_indices, "inference_func"
    ) == (1, 1)


def test_get_positional_parameter_bounds_handles_both_variadic_kinds_indices():
    """
    With both *args and **kwargs only the *args moves the maximum; the pair is accepted rather
    than reported as variadic.
    """
    assert input_checks._get_positional_parameter_bounds(
        _doubly_variadic_func_indices, "alg_update_func"
    ) == (1, math.inf)


def test_get_positional_parameter_bounds_excludes_keyword_only_parameters_indices():
    """
    Changed behavior: a keyword-only parameter is NOT a positional position, so
    `(beta, features, *, scale)` bounds the call at two, not three. The batching code can only
    fill positions positionally, so demanding a supplied value for `scale` was wrong.
    """
    assert input_checks._get_positional_parameter_bounds(
        _keyword_only_func_indices, "alg_update_func"
    ) == (2, 2)


def test_get_positional_parameter_bounds_counts_positional_only_parameters_indices():
    """Positional-only parameters (declared before /) count like ordinary positionals."""
    assert input_checks._get_positional_parameter_bounds(
        _positional_only_func_indices, "action_prob_func"
    ) == (2, 2)


def test_get_positional_parameter_bounds_counts_zero_parameter_function_indices():
    """Edge case: a function declaring no parameters bounds the call at exactly zero."""
    assert input_checks._get_positional_parameter_bounds(
        _no_parameter_func_indices, "alg_update_func"
    ) == (0, 0)


def test_get_positional_parameter_bounds_resolves_jit_wrapped_signature_indices():
    """
    Regression test: a jax.jit-wrapped function is a PjitFunction with NO __code__ attribute,
    so the bounds must come from inspect.signature (which follows functools.wraps to the
    wrapped function) rather than func.__code__.co_argcount.
    """
    jitted = jax.jit(_alg_update_func_indices)
    assert not hasattr(jitted, "__code__")
    assert input_checks._get_positional_parameter_bounds(jitted, "alg_update_func") == (
        4,
        4,
    )


def test_get_positional_parameter_bounds_resolves_grad_wrapped_signature_indices():
    """
    A jax.grad-wrapped loss function resolves to the WRAPPED signature too -- and here
    co_argcount actively lies (the grad wrapper is a *args/**kwargs closure reporting 0), so
    signature resolution is the only thing that works.
    """
    gradded = jax.grad(_alg_update_func_indices)
    assert gradded.__code__.co_argcount == 0
    assert input_checks._get_positional_parameter_bounds(
        gradded, "alg_update_func"
    ) == (4, 4)


def test_get_positional_parameter_bounds_resolves_jit_wrapped_shim_indices():
    """
    Wrapping a *args shim in jax.jit keeps it accepted with an unbounded maximum: signature
    resolution sees through the wrapper, and there is no longer anything to reject.
    """
    assert input_checks._get_positional_parameter_bounds(
        jax.jit(_forwarding_shim_func_indices), "alg_update_func"
    ) == (1, math.inf)


def test_get_positional_parameter_bounds_rejects_uninspectable_callable_indices():
    """
    A C-implemented callable whose signature inspect cannot recover (numpy ufunc) raises
    ValueError inside inspect.signature; that is still converted into a named AssertionError.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape("Could not inspect the signature of alg_update_func"),
    ):
        input_checks._get_positional_parameter_bounds(np.add, "alg_update_func")


def test_get_positional_parameter_bounds_rejects_non_callable_indices():
    """
    A non-callable passed where a function belongs (a common mis-wire: passing the args instead
    of the function) raises TypeError inside inspect.signature and is reported as an
    AssertionError naming the slot.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape("Could not inspect the signature of inference_func (3)"),
    ):
        input_checks._get_positional_parameter_bounds(3, "inference_func")


### _describe_positional_bounds


def test_describe_positional_bounds_says_at_least_for_unbounded_maximum_indices():
    """The *args branch: an infinite maximum is described by its floor alone."""
    assert input_checks._describe_positional_bounds(1, math.inf) == "at least 1"
    assert input_checks._describe_positional_bounds(0, math.inf) == "at least 0"


def test_describe_positional_bounds_says_exactly_for_a_fixed_signature_indices():
    """The all-required branch: one number, phrased as an exact demand."""
    assert input_checks._describe_positional_bounds(4, 4) == "exactly 4"
    assert input_checks._describe_positional_bounds(0, 0) == "exactly 0"


def test_describe_positional_bounds_says_a_range_when_defaults_widen_it_indices():
    """The defaulted-parameter branch: a closed range, low end first."""
    assert input_checks._describe_positional_bounds(4, 5) == "4 to 5"
    assert input_checks._describe_positional_bounds(0, 3) == "0 to 3"


def test_describe_positional_bounds_matches_the_real_functions_bounds_indices():
    """The three branches as the real helper produces them, so the phrasing stays paired."""
    assert (
        input_checks._describe_positional_bounds(
            *input_checks._get_positional_parameter_bounds(
                _forwarding_shim_func_indices, "alg_update_func"
            )
        )
        == "at least 1"
    )
    assert (
        input_checks._describe_positional_bounds(
            *input_checks._get_positional_parameter_bounds(
                _alg_update_func_indices, "alg_update_func"
            )
        )
        == "exactly 4"
    )
    assert (
        input_checks._describe_positional_bounds(
            *input_checks._get_positional_parameter_bounds(
                _defaulted_hyperparameter_func_indices, "alg_update_func"
            )
        )
        == "4 to 5"
    )


### _get_declared_positional_parameter_count


def test_get_declared_positional_parameter_count_matches_co_argcount_indices():
    """
    On a plain function the count is the declared positional count, which is exactly what
    post_deployment_analysis.process_inference_func_args reads via __code__.co_argcount.
    """
    assert (
        input_checks._get_declared_positional_parameter_count(
            _alg_update_func_indices, "inference_func"
        )
        == _alg_update_func_indices.__code__.co_argcount
        == 4
    )
    assert (
        input_checks._get_declared_positional_parameter_count(
            _action_prob_func_indices, "inference_func"
        )
        == 2
    )


def test_get_declared_positional_parameter_count_excludes_var_positional_indices():
    """
    *args is EXCLUDED, matching co_argcount: a shim declaring one named parameter plus *rest
    counts as one, not as an unbounded position, so the theta index is bounded by something
    finite.
    """
    assert (
        input_checks._get_declared_positional_parameter_count(
            _forwarding_shim_func_indices, "inference_func"
        )
        == _forwarding_shim_func_indices.__code__.co_argcount
        == 1
    )
    assert (
        input_checks._get_declared_positional_parameter_count(
            _doubly_variadic_func_indices, "inference_func"
        )
        == 1
    )


def test_get_declared_positional_parameter_count_includes_defaulted_parameters_indices():
    """
    A defaulted parameter IS a declared position -- process_inference_func_args builds one
    inference argument per declared parameter -- so it counts, unlike in the bounds helper's
    minimum.
    """
    assert (
        input_checks._get_declared_positional_parameter_count(
            _defaulted_hyperparameter_func_indices, "inference_func"
        )
        == _defaulted_hyperparameter_func_indices.__code__.co_argcount
        == 5
    )


def test_get_declared_positional_parameter_count_excludes_keyword_only_indices():
    """Keyword-only parameters are not declared POSITIONS, and co_argcount agrees."""
    assert (
        input_checks._get_declared_positional_parameter_count(
            _keyword_only_func_indices, "inference_func"
        )
        == _keyword_only_func_indices.__code__.co_argcount
        == 2
    )


def test_get_declared_positional_parameter_count_counts_zero_parameter_function_indices():
    """Edge case: no declared parameters means no position a theta index could address."""
    assert (
        input_checks._get_declared_positional_parameter_count(
            _no_parameter_func_indices, "inference_func"
        )
        == 0
    )


def test_get_declared_positional_parameter_count_resolves_jit_wrapped_signature_indices():
    """The jit wrapper has no __code__ at all, so this count comes from the signature too."""
    assert (
        input_checks._get_declared_positional_parameter_count(
            jax.jit(_alg_update_func_indices), "inference_func"
        )
        == 4
    )
    assert (
        input_checks._get_declared_positional_parameter_count(
            jax.jit(_forwarding_shim_func_indices), "inference_func"
        )
        == 1
    )


def test_get_declared_positional_parameter_count_rejects_uninspectable_callable_indices():
    """It inherits the bounds helper's named AssertionError for an uninspectable callable."""
    with pytest.raises(
        AssertionError,
        match=re.escape("Could not inspect the signature of inference_func"),
    ):
        input_checks._get_declared_positional_parameter_count(np.add, "inference_func")


### require_arg_tuple_lengths_consistent_and_callable


def test_require_arg_tuple_lengths_consistent_and_callable_returns_common_length_indices():
    """
    Valid mask-free wiring: four-entry supplied tuples against a four-parameter function, and
    the returned common SUPPLIED length is what the caller range-checks its indices against.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=4),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 4
    )


def test_require_arg_tuple_lengths_consistent_and_callable_passes_for_mask_aware_func_indices():
    """
    Under mask padding it is supplied_length + 1 that must be callable, since
    self_pad_ragged_args_and_build_mask appends the validity mask as a new last argument: a
    mask-aware function whose arity is exactly one more than the supplied length passes, and
    the returned length is the SUPPLIED 3, not the callable 4.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _mask_aware_alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=3),
            func_description="alg_update_func",
            key_description="policy_num",
            mask_index=3,
        )
        == 3
    )


def test_require_arg_tuple_lengths_consistent_and_callable_mask_off_reserves_no_position_indices():
    """
    The mask flag is what buys that extra position: the SAME four-parameter mask-aware function
    fed three-entry tuples with padding OFF is called with three arguments and is rejected.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 3 positional argument(s), but its signature "
            "accepts exactly 4."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _mask_aware_alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=3),
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_accepts_variadic_shim_indices():
    """
    The relaxation that matters: a `def w(beta_est, *rest)` forwarding shim is ACCEPTED at any
    supplied width, because its maximum is math.inf. The old equality check rejected it even
    though it produces bit-identical estimates.
    """
    for tuple_length in (1, 4, 9):
        assert (
            input_checks.require_arg_tuple_lengths_consistent_and_callable(
                _forwarding_shim_func_indices,
                _build_supplied_args_indices(tuple_length=tuple_length),
                func_description="alg_update_func",
                key_description="policy_num",
            )
            == tuple_length
        )


def test_require_arg_tuple_lengths_consistent_and_callable_accepts_variadic_shim_with_mask_indices():
    """A *args shim is callable with the appended mask too, whatever the supplied width."""
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _forwarding_shim_func_indices,
            _build_supplied_args_indices(tuple_length=3),
            func_description="alg_update_func",
            key_description="policy_num",
            mask_index=3,
        )
        == 3
    )


def test_require_arg_tuple_lengths_consistent_and_callable_accepts_defaulted_parameter_indices():
    """
    The second accepted wrapper shape: a function carrying an extra defaulted hyperparameter
    (`..., n, ridge=0.0`) fed tuples ONE ENTRY SHORTER than its declared count passes, and the
    returned length is the supplied 4 rather than the declared 5.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _defaulted_hyperparameter_func_indices,
            _build_supplied_args_indices(tuple_length=4),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 4
    )


def test_require_arg_tuple_lengths_consistent_and_callable_accepts_filled_default_indices():
    """
    Supplying the defaulted parameter explicitly (five entries for the same function) is
    equally callable: the check is a RANGE, so both ends of it are valid wiring.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _defaulted_hyperparameter_func_indices,
            _build_supplied_args_indices(tuple_length=5),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 5
    )


def test_require_arg_tuple_lengths_consistent_and_callable_ignores_var_keyword_indices():
    """
    **kwargs alone does not widen the positional bounds, so a one-parameter **kwargs function
    still takes exactly one supplied entry -- and two are rejected.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _variadic_keyword_func_indices,
            _build_supplied_args_indices(tuple_length=1),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 1
    )
    with pytest.raises(
        AssertionError, match=re.escape("but its signature accepts exactly 1.")
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _variadic_keyword_func_indices,
            _build_supplied_args_indices(tuple_length=2),
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_passes_for_jit_wrapped_func_indices():
    """A jax.jit-wrapped update function has its bounds resolved, not treated as uninspectable."""
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            jax.jit(_alg_update_func_indices),
            _build_supplied_args_indices(tuple_length=4),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 4
    )


def test_require_arg_tuple_lengths_consistent_and_callable_skips_blank_tuples_indices():
    """
    Blank () argument tuples mean "not applicable at this update" and must be skipped by the
    consistency check, not reported as a second, zero-length width.
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=4, blank_cells=((2, 0), (3, 1))
    )
    assert supplied_args[2][0] == ()
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 4
    )


def test_require_arg_tuple_lengths_consistent_and_callable_returns_none_on_empty_mapping_indices():
    """
    An empty args mapping supplies no width, so the return is None -- the signal
    perform_first_wave_input_checks guards the index/mask/ragged checks behind. A mapping of
    empty per-subject dicts is the same degenerate case.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _action_prob_func_indices,
            {},
            func_description="action_prob_func",
            key_description="decision_time",
        )
        is None
    )
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _action_prob_func_indices,
            {2: {}, 3: {}},
            func_description="action_prob_func",
            key_description="decision_time",
        )
        is None
    )


def test_require_arg_tuple_lengths_consistent_and_callable_skips_the_signature_when_empty_indices():
    """
    With no supplied width there is nothing to check the signature against, so the function is
    never inspected at all -- even a non-callable in the function slot returns None here rather
    than raising, which is why the uninspectable-function rejection needs a non-blank tuple.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            3,
            {},
            func_description="alg_update_func",
            key_description="policy_num",
        )
        is None
    )


def test_require_arg_tuple_lengths_consistent_and_callable_returns_none_when_all_blank_indices():
    """
    A mapping whose every tuple is blank also yields None rather than 0: there is no supplied
    width, and the emptiness itself is other checks' finding (a policy blank for every subject
    is require_every_update_policy_has_at_least_one_nonblank_arg_tuple's).
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=2, blank_cells=((2, 0), (2, 1), (3, 0), (3, 1))
    )
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _action_prob_func_indices,
            supplied_args,
            func_description="action_prob_func",
            key_description="decision_time",
        )
        is None
    )


def test_require_arg_tuple_lengths_consistent_and_callable_passes_on_float_keys_indices():
    """
    Real fixtures carry float64 policy numbers (1.0..7.0) with the initial policy 1.0 absent
    from alg_update_func_args, so float keys starting above zero must be accepted unchanged.
    """
    assert (
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=4, keys=(2.0, 3.0)),
            func_description="alg_update_func",
            key_description="policy_num",
        )
        == 4
    )


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_inconsistent_widths_indices():
    """
    The substantive half: one three-entry tuple among four-entry ones is the real wiring error,
    and the message names the distinct lengths present plus one example
    (policy_num, subject_id) key per length.
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=4, length_overrides={(2, 0): 3}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Supplied alg_update_func argument tuples do not all have the same length; "
            "lengths [3, 4] are all present. One example (policy_num, subject_id) per length: "
            "{3: (2, 0), 4: (2, 1)}. Every non-blank tuple must carry the same arguments in "
            "the same positions."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_names_the_key_description_indices():
    """
    The action-prob slot keys tuples by DECISION TIME, and the message must say so rather than
    naming policy numbers.
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=2, keys=(0, 1), length_overrides={(1, 1): 3}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Supplied action_prob_func argument tuples do not all have the same length; "
            "lengths [2, 3] are all present. One example (decision_time, subject_id) per "
            "length: {2: (0, 0), 3: (1, 1)}"
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _action_prob_func_indices,
            supplied_args,
            func_description="action_prob_func",
            key_description="decision_time",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_reports_three_widths_indices():
    """Every distinct length is listed, each with its own example key, sorted ascending."""
    supplied_args = _build_supplied_args_indices(
        tuple_length=4, length_overrides={(2, 0): 3, (3, 1): 5}
    )
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )
    message = str(excinfo.value)
    assert "lengths [3, 4, 5] are all present" in message
    assert "per length: {3: (2, 0), 4: (2, 1), 5: (3, 1)}" in message


def test_require_arg_tuple_lengths_consistent_and_callable_picks_one_example_per_length_indices():
    """
    With many offenders the example per length is chosen deterministically (the str()-smallest
    key), so the message does not depend on dict ordering: nine length-3 cells and two length-4
    ones yield exactly one example each.
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=3,
        keys=(2, 3, 4),
        subject_ids=(0, 1, 2),
        length_overrides={(3, 1): 4, (4, 2): 4},
    )
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )
    message = str(excinfo.value)
    assert "per length: {3: (2, 0), 4: (3, 1)}" in message
    assert "(4, 2)" not in message


def test_require_arg_tuple_lengths_consistent_and_callable_handles_unorderable_keys_indices():
    """
    Studies mix key types (int or float policy numbers, arbitrary hashable subject ids), which
    are not mutually orderable, so the example keys are picked by str() -- an int key and a str
    key together must not raise TypeError from inside the checker.
    """
    supplied_args = {
        2: {0: (jnp.arange(3.0),) * 3},
        "a": {0: (jnp.arange(3.0),) * 4},
    }
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )
    assert "per length: {3: (2, 0), 4: ('a', 0)}" in str(excinfo.value)


def test_require_arg_tuple_lengths_consistent_and_callable_checks_consistency_first_indices():
    """
    Ordering is pinned: inconsistent widths are reported even for a *args shim that could
    accept any of them, because the inconsistency is the finding and the shim is not.
    """
    supplied_args = _build_supplied_args_indices(
        tuple_length=4, length_overrides={(2, 0): 3}
    )
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _forwarding_shim_func_indices,
            supplied_args,
            func_description="alg_update_func",
            key_description="policy_num",
        )
    message = str(excinfo.value)
    assert "do not all have the same length; lengths [3, 4] are all present" in message
    assert "will be called with" not in message


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_too_many_args_indices():
    """
    Callability, upper end: four consistent entries against a two-parameter function cannot be
    called, and the message states the call width and the accepted band.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "action_prob_func will be called with 4 positional argument(s), but its signature "
            "accepts exactly 2. Please see the contract for details."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _action_prob_func_indices,
            _build_supplied_args_indices(tuple_length=4),
            func_description="action_prob_func",
            key_description="decision_time",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_too_few_args_indices():
    """
    Callability, lower end: two entries against four REQUIRED parameters would raise a
    TypeError at call time, so it is rejected here by name.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 2 positional argument(s), but its signature "
            "accepts exactly 4."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=2),
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_below_shim_minimum_indices():
    """
    Even an unbounded *args shim has a floor: one entry against `(beta, features, *rest)` is
    rejected, and the "at least N" phrasing is what the message carries.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 1 positional argument(s), but its signature "
            "accepts at least 2."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _required_pair_then_variadic_func_indices,
            _build_supplied_args_indices(tuple_length=1),
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_reports_a_bounded_range_indices():
    """
    Overshooting a defaulted-parameter signature reports the closed range, so the reader can
    see that both 4 and 5 would have been fine and 6 is not.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 6 positional argument(s), but its signature "
            "accepts 4 to 5."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _defaulted_hyperparameter_func_indices,
            _build_supplied_args_indices(tuple_length=6),
            func_description="alg_update_func",
            key_description="policy_num",
        )


def test_require_arg_tuple_lengths_consistent_and_callable_explains_the_mask_argument_indices():
    """
    Under padding the call width is supplied + 1, and the message must explain WHERE the extra
    argument comes from so a four-entry tuple against a four-parameter mask-aware function does
    not look like an off-by-one in the checker.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 5 positional argument(s) (the 4 supplied, "
            "plus the validity mask that will be appended at mask index 4), but its signature "
            "accepts exactly 4."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _mask_aware_alg_update_func_indices,
            _build_supplied_args_indices(tuple_length=4),
            func_description="alg_update_func",
            key_description="policy_num",
            mask_index=4,
        )


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_no_room_for_mask_indices():
    """
    Edge case: a zero-parameter function leaves no room for a requested validity mask, and the
    mask note names the mask index that asked for it.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 2 positional argument(s) (the 1 supplied, "
            "plus the validity mask that will be appended at mask index 0), but its signature "
            "accepts exactly 0."
        ),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            _no_parameter_func_indices,
            _build_supplied_args_indices(tuple_length=1),
            func_description="alg_update_func",
            key_description="policy_num",
            mask_index=0,
        )


def test_require_arg_tuple_lengths_consistent_and_callable_rejects_uninspectable_func_indices():
    """
    Once there IS a supplied width the function must be inspectable, so a non-callable in the
    function slot is reported as a named AssertionError rather than as a bare TypeError.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape("Could not inspect the signature of alg_update_func (3)"),
    ):
        input_checks.require_arg_tuple_lengths_consistent_and_callable(
            3,
            _build_supplied_args_indices(tuple_length=4),
            func_description="alg_update_func",
            key_description="policy_num",
        )


### require_arg_indices_supplied


def test_require_arg_indices_supplied_passes_for_non_negative_indices_indices():
    """Index 0 counts as supplied: only a NEGATIVE index means "absent"."""
    input_checks.require_arg_indices_supplied(
        {"alg_update_func_args_beta_index": 0}, "alg_update_func"
    )
    input_checks.require_arg_indices_supplied(
        {
            "alg_update_func_args_beta_index": 3,
            "inference_func_args_theta_index": 0,
        },
        "alg_update_func",
    )


def test_require_arg_indices_supplied_passes_on_empty_mapping_indices():
    """Edge case: nothing required means nothing to complain about."""
    input_checks.require_arg_indices_supplied({}, "alg_update_func")


def test_require_arg_indices_supplied_rejects_absent_beta_index_indices():
    """
    beta is ALWAYS threaded, so a negative beta index is a mis-wire; the message must name the
    parameter and explain that negative means absent.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func argument indices are required but were not supplied (a "
            "negative index means absent): {'alg_update_func_args_beta_index': -1}"
        ),
    ):
        input_checks.require_arg_indices_supplied(
            {"alg_update_func_args_beta_index": -1}, "alg_update_func"
        )


def test_require_arg_indices_supplied_rejects_absent_theta_index_indices():
    """
    An absent theta index is the SILENT failure this check exists for: inference would be
    differentiated with respect to a theta that was never substituted.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These inference_func argument indices are required but were not supplied"
        ),
    ):
        input_checks.require_arg_indices_supplied(
            {"inference_func_args_theta_index": -1}, "inference_func"
        )


def test_require_arg_indices_supplied_reports_every_absent_index_indices():
    """All absent required indices are reported together, not one per run."""
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_indices_supplied(
            {
                "alg_update_func_args_beta_index": -1,
                "inference_func_args_theta_index": -5,
                "some_supplied_index": 2,
            },
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert "'alg_update_func_args_beta_index': -1" in message
    assert "'inference_func_args_theta_index': -5" in message
    assert "some_supplied_index" not in message


### require_arg_indices_in_range


def test_require_arg_indices_in_range_passes_for_addressable_indices_indices():
    """Positions 0..length-1 all address a real entry in the supplied tuples."""
    input_checks.require_arg_indices_in_range(
        _build_alg_update_indices_by_name_indices(), 4, "alg_update_func"
    )


def test_require_arg_indices_in_range_skips_absent_negative_indices_indices():
    """
    Negative indices mark genuinely optional positions (action probabilities, their times,
    previous betas) as absent and must be SKIPPED rather than reported as out of range.
    """
    input_checks.require_arg_indices_in_range(
        _build_alg_update_indices_by_name_indices(
            beta_index=0,
            previous_betas_index=-1,
            action_prob_index=-1,
            action_prob_times_index=-1,
        ),
        1,
        "alg_update_func",
    )


def test_require_arg_indices_in_range_skips_negative_indices_on_empty_tuples_indices():
    """
    Edge case: with an expected tuple length of 0 every absent index is still fine, because
    absence is decided before range.
    """
    input_checks.require_arg_indices_in_range(
        {"alg_update_func_args_action_prob_index": -1}, 0, "alg_update_func"
    )


def test_require_arg_indices_in_range_rejects_index_equal_to_length_indices():
    """
    The classic off-by-one: index 4 into four-entry tuples. Naming the parameter and the valid
    range beats the bare IndexError raised by whichever precompute step reaches it first.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func argument indices do not address a position in argument "
            "tuples of length 4 (valid positions are 0 through 3): "
            "{'alg_update_func_args_action_prob_times_index': 4}"
        ),
    ):
        input_checks.require_arg_indices_in_range(
            _build_alg_update_indices_by_name_indices(action_prob_times_index=4),
            4,
            "alg_update_func",
        )


def test_require_arg_indices_in_range_rejects_multiple_out_of_range_indices_indices():
    """Every out-of-range index is reported at once."""
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_indices_in_range(
            _build_alg_update_indices_by_name_indices(
                action_prob_index=7, action_prob_times_index=9
            ),
            4,
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert "'alg_update_func_args_action_prob_index': 7" in message
    assert "'alg_update_func_args_action_prob_times_index': 9" in message
    assert "beta_index" not in message


def test_require_arg_indices_in_range_rejects_supplied_index_against_empty_tuples_indices():
    """
    Edge case: when the expected supplied-tuple length is 0, any supplied (non-negative) index
    is out of range -- there is no position to address.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "argument tuples of length 0 (valid positions are 0 through -1): "
            "{'alg_update_func_args_beta_index': 0}"
        ),
    ):
        input_checks.require_arg_indices_in_range(
            {"alg_update_func_args_beta_index": 0}, 0, "alg_update_func"
        )


### require_arg_indices_distinct


def test_require_arg_indices_distinct_passes_for_distinct_positions_indices():
    """The valid wiring: four parameters on four different positions."""
    input_checks.require_arg_indices_distinct(
        _build_alg_update_indices_by_name_indices(), "alg_update_func"
    )


def test_require_arg_indices_distinct_passes_on_empty_mapping_indices():
    """Edge case: no indices, no collisions."""
    input_checks.require_arg_indices_distinct({}, "alg_update_func")


def test_require_arg_indices_distinct_ignores_shared_negative_indices_indices():
    """
    Several parameters may all be ABSENT (-1) at once; only SUPPLIED indices are required to be
    distinct, so shared negative values must not be flagged as a collision.
    """
    input_checks.require_arg_indices_distinct(
        _build_alg_update_indices_by_name_indices(
            beta_index=0,
            previous_betas_index=-1,
            action_prob_index=-1,
            action_prob_times_index=-1,
        ),
        "alg_update_func",
    )


def test_require_arg_indices_distinct_rejects_beta_action_prob_collision_indices():
    """
    beta_index == action_prob_index means thread_update_func_args writes the reconstructed
    action probabilities over the beta position, so the beta being differentiated never reaches
    the update function. The message maps position -> the colliding parameter names, sorted.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func argument indices collide -- position -> the parameters that "
            "claim it: {0: ['alg_update_func_args_action_prob_index', "
            "'alg_update_func_args_beta_index']}"
        ),
    ):
        input_checks.require_arg_indices_distinct(
            _build_alg_update_indices_by_name_indices(
                beta_index=0, action_prob_index=0
            ),
            "alg_update_func",
        )


def test_require_arg_indices_distinct_rejects_two_separate_collisions_indices():
    """Two independent collisions are both reported, keyed by the position they fight over."""
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_indices_distinct(
            _build_alg_update_indices_by_name_indices(
                beta_index=0,
                previous_betas_index=1,
                action_prob_index=0,
                action_prob_times_index=1,
            ),
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert (
        "0: ['alg_update_func_args_action_prob_index', 'alg_update_func_args_beta_index']"
        in message
    )
    assert (
        "1: ['alg_update_func_args_action_prob_times_index', "
        "'alg_update_func_args_previous_betas_index']" in message
    )


def test_require_arg_indices_distinct_rejects_three_way_collision_indices():
    """A three-way collision lists all three claimants for the position."""
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_arg_indices_distinct(
            {"a_index": 2, "b_index": 2, "c_index": 2, "d_index": 0},
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert "2: ['a_index', 'b_index', 'c_index']" in message
    assert "d_index" not in message


### require_mask_index_appends_after_supplied_args


def test_require_mask_index_appends_after_supplied_args_no_ops_when_padding_off_indices():
    """
    Padding off (mask index -1) makes this a no-op, so the supplied tuple length is irrelevant
    and nothing raises.
    """
    assert (
        input_checks.require_mask_index_appends_after_supplied_args(
            -1, 4, "alg_update_func"
        )
        is None
    )
    input_checks.require_mask_index_appends_after_supplied_args(
        -7, 0, "alg_update_func"
    )


def test_require_mask_index_appends_after_supplied_args_passes_when_appended_indices():
    """
    The valid mask wiring: the mask index is exactly the supplied tuple length, i.e. one past
    the last supplied position.
    """
    input_checks.require_mask_index_appends_after_supplied_args(3, 3, "alg_update_func")
    input_checks.require_mask_index_appends_after_supplied_args(0, 0, "alg_update_func")


def test_require_mask_index_appends_after_supplied_args_rejects_inserted_mask_indices():
    """
    A mask index INSIDE the supplied range would overwrite a real argument; the mask is always
    appended, never inserted.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func mask index 1 must equal the supplied argument tuple length (3): "
            "the validity mask is appended as a new last argument, so it belongs at position 3"
        ),
    ):
        input_checks.require_mask_index_appends_after_supplied_args(
            1, 3, "alg_update_func"
        )


def test_require_mask_index_appends_after_supplied_args_rejects_gap_after_args_indices():
    """A mask index past the end of the supplied tuples leaves an unfilled hole and is rejected."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func mask index 5 must equal the supplied argument tuple length (3)"
        ),
    ):
        input_checks.require_mask_index_appends_after_supplied_args(
            5, 3, "alg_update_func"
        )


### require_ragged_indices_valid


def test_require_ragged_indices_valid_no_ops_when_padding_off_indices():
    """
    With padding off the stacking code ignores ragged_indices entirely, so even indices that
    are out of range AND name shared parameters must not raise.
    """
    assert (
        input_checks.require_ragged_indices_valid(
            (0, 99),
            -1,
            3,
            _build_shared_indices_by_name_indices(beta_index=0),
            "alg_update_func",
        )
        is None
    )


def test_require_ragged_indices_valid_passes_for_per_subject_positions_indices():
    """
    The valid wiring: ragged positions are in range and none of them is a shared parameter
    (beta at 0, previous betas at 1 here).
    """
    input_checks.require_ragged_indices_valid(
        (2, 3),
        4,
        4,
        _build_shared_indices_by_name_indices(beta_index=0, previous_betas_index=1),
        "alg_update_func",
    )


def test_require_ragged_indices_valid_passes_when_previous_betas_absent_indices():
    """
    An ABSENT shared parameter (previous betas at -1) cannot collide with a ragged position, so
    it must not be matched against the ragged set.
    """
    input_checks.require_ragged_indices_valid(
        (1, 2),
        3,
        3,
        _build_shared_indices_by_name_indices(beta_index=0, previous_betas_index=-1),
        "alg_update_func",
    )


def test_require_ragged_indices_valid_rejects_empty_ragged_indices_indices():
    """
    Requesting mask padding with no positions to pad is a mis-wire: nothing would be padded and
    the mask would describe rows that were never added.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func requests mask padding (mask index 3) but supplied no ragged "
            "argument positions to pad"
        ),
    ):
        input_checks.require_ragged_indices_valid(
            (),
            3,
            3,
            _build_shared_indices_by_name_indices(),
            "alg_update_func",
        )


def test_require_ragged_indices_valid_rejects_out_of_range_ragged_index_indices():
    """A ragged position past the end of the supplied tuples addresses nothing."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func ragged argument positions do not address a supplied "
            "position in argument tuples of length 3: [3]"
        ),
    ):
        input_checks.require_ragged_indices_valid(
            (1, 3),
            3,
            3,
            _build_shared_indices_by_name_indices(
                beta_index=0, previous_betas_index=-1
            ),
            "alg_update_func",
        )


def test_require_ragged_indices_valid_rejects_negative_ragged_index_indices():
    """
    A negative ragged index is NOT read as "absent" here (unlike the shared-parameter indices):
    ragged_indices is a list of positions to pad, so a negative entry is out of range, and the
    offenders are reported sorted for determinism.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func ragged argument positions do not address a supplied "
            "position in argument tuples of length 3: [-1, 5]"
        ),
    ):
        input_checks.require_ragged_indices_valid(
            (5, 1, -1),
            3,
            3,
            _build_shared_indices_by_name_indices(
                beta_index=0, previous_betas_index=-1
            ),
            "alg_update_func",
        )


def test_require_ragged_indices_valid_rejects_beta_as_ragged_index_indices():
    """
    beta is a (beta_dim,) vector SHARED across subjects: self-padding it would append copies of
    its last component and change its dimension rather than adding padding rows.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func parameters are shared across subjects but were listed as "
            "ragged (self-padding) positions: {'alg_update_func_args_beta_index': 0}"
        ),
    ):
        input_checks.require_ragged_indices_valid(
            (0, 2),
            3,
            3,
            _build_shared_indices_by_name_indices(
                beta_index=0, previous_betas_index=-1
            ),
            "alg_update_func",
        )


def test_require_ragged_indices_valid_rejects_previous_betas_as_ragged_index_indices():
    """Previous betas are shared across subjects the same way beta is."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "ragged (self-padding) positions: "
            "{'alg_update_func_args_previous_betas_index': 1}"
        ),
    ):
        input_checks.require_ragged_indices_valid(
            (1, 2),
            3,
            3,
            _build_shared_indices_by_name_indices(beta_index=0, previous_betas_index=1),
            "alg_update_func",
        )


def test_require_ragged_indices_valid_reports_both_shared_parameters_indices():
    """Both shared parameters are reported together when both were listed as ragged."""
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_ragged_indices_valid(
            (0, 1),
            2,
            2,
            _build_shared_indices_by_name_indices(beta_index=0, previous_betas_index=1),
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert "'alg_update_func_args_beta_index': 0" in message
    assert "'alg_update_func_args_previous_betas_index': 1" in message


def test_require_ragged_indices_valid_checks_range_before_shared_parameters_indices():
    """
    Ordering is pinned: an out-of-range ragged index is reported even when a shared-parameter
    collision is also present, so the cheaper structural problem is surfaced first.
    """
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_ragged_indices_valid(
            (0, 9),
            3,
            3,
            _build_shared_indices_by_name_indices(
                beta_index=0, previous_betas_index=-1
            ),
            "alg_update_func",
        )
    message = str(excinfo.value)
    assert "ragged argument positions do not address a supplied position" in message
    assert "shared across subjects" not in message


### Cross-function wiring: the consistency check's return value feeds the index checks


def test_supplied_length_return_value_range_checks_mask_aware_indices_end_to_end_indices():
    """
    The pipeline perform_first_wave_input_checks runs: the returned common SUPPLIED length
    (arity - 1 under padding) is what the range, distinctness, mask-position and ragged checks
    are all measured against, so index 3 -- a real parameter of the function but the MASK's
    position -- must be rejected as out of range for the supplied tuples.
    """
    supplied_args = _build_supplied_args_indices(tuple_length=3)
    supplied_length = input_checks.require_arg_tuple_lengths_consistent_and_callable(
        _mask_aware_alg_update_func_indices,
        supplied_args,
        func_description="alg_update_func",
        key_description="policy_num",
        mask_index=3,
    )
    assert supplied_length == 3

    indices_by_name = _build_alg_update_indices_by_name_indices(
        beta_index=0,
        previous_betas_index=-1,
        action_prob_index=1,
        action_prob_times_index=2,
    )
    input_checks.require_arg_indices_supplied(
        {"alg_update_func_args_beta_index": 0}, "alg_update_func"
    )
    input_checks.require_arg_indices_in_range(
        indices_by_name, supplied_length, "alg_update_func"
    )
    input_checks.require_arg_indices_distinct(indices_by_name, "alg_update_func")
    input_checks.require_mask_index_appends_after_supplied_args(
        3, supplied_length, "alg_update_func"
    )
    input_checks.require_ragged_indices_valid(
        (1, 2),
        3,
        supplied_length,
        _build_shared_indices_by_name_indices(beta_index=0, previous_betas_index=-1),
        "alg_update_func",
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "argument tuples of length 3 (valid positions are 0 through 2)"
        ),
    ):
        input_checks.require_arg_indices_in_range(
            {"alg_update_func_args_action_prob_times_index": 3},
            supplied_length,
            "alg_update_func",
        )


def test_supplied_length_return_value_range_checks_variadic_shim_end_to_end_indices():
    """
    The same pipeline with the accepted *args forwarding shim in the function slot: the
    downstream checks are measured against the DATA's width (four here), not against the shim's
    single declared parameter, so nothing downstream notices that the function is variadic.
    """
    supplied_length = input_checks.require_arg_tuple_lengths_consistent_and_callable(
        _forwarding_shim_func_indices,
        _build_supplied_args_indices(tuple_length=4),
        func_description="alg_update_func",
        key_description="policy_num",
    )
    assert supplied_length == 4

    indices_by_name = _build_alg_update_indices_by_name_indices()
    input_checks.require_arg_indices_in_range(
        indices_by_name, supplied_length, "alg_update_func"
    )
    input_checks.require_arg_indices_distinct(indices_by_name, "alg_update_func")
    input_checks.require_mask_index_appends_after_supplied_args(
        -1, supplied_length, "alg_update_func"
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "argument tuples of length 4 (valid positions are 0 through 3)"
        ),
    ):
        input_checks.require_arg_indices_in_range(
            {"alg_update_func_args_beta_index": 4},
            supplied_length,
            "alg_update_func",
        )


def test_none_supplied_length_is_why_the_caller_guards_the_index_checks_indices():
    """
    Pins the reason perform_first_wave_input_checks wraps the range/mask/ragged checks in
    `if <length> is not None:`: the None return cannot be compared against an index, so handing
    it straight to require_arg_indices_in_range raises a bare TypeError instead of a named
    finding.
    """
    supplied_length = input_checks.require_arg_tuple_lengths_consistent_and_callable(
        _alg_update_func_indices,
        {},
        func_description="alg_update_func",
        key_description="policy_num",
    )
    assert supplied_length is None
    with pytest.raises(TypeError):
        input_checks.require_arg_indices_in_range(
            {"alg_update_func_args_beta_index": 0},
            supplied_length,
            "alg_update_func",
        )


def test_inspect_signature_is_the_positional_bound_source_of_truth_indices():
    """
    Guards the invariant the docstrings rest on: the bounds come from inspect.signature, which
    agrees with co_argcount for a plain function but is the ONLY thing that works for a
    jax.jit-wrapped one (no __code__ at all) or a jax.grad-wrapped one (whose __code__ belongs
    to the wrapper and reports 0).
    """
    minimum, maximum = input_checks._get_positional_parameter_bounds(
        _alg_update_func_indices, "alg_update_func"
    )
    assert (
        len(inspect.signature(_alg_update_func_indices).parameters)
        == _alg_update_func_indices.__code__.co_argcount
        == input_checks._get_declared_positional_parameter_count(
            _alg_update_func_indices, "alg_update_func"
        )
        == minimum
        == maximum
        == 4
    )

    jitted = jax.jit(_alg_update_func_indices)
    assert not hasattr(jitted, "__code__")
    gradded = jax.grad(_alg_update_func_indices)
    assert gradded.__code__.co_argcount == 0
    for wrapped in (jitted, gradded):
        assert input_checks._get_positional_parameter_bounds(
            wrapped, "alg_update_func"
        ) == (4, 4)
