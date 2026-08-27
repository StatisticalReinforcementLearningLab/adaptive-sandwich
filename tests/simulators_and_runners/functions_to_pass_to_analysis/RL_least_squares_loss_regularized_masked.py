import jax
from jax import numpy as jnp


@jax.jit
def RL_least_squares_loss_regularized_masked(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action1probtimes,
    action_centering,
    lambda_,
    n,
    mask,
):
    """
    Mask-aware variant of RL_least_squares_loss_regularized.py, for testing
    lifejacket.batched_weighted_estimating_function_stack's
    alg_update_func_args_mask_index / combine_updates_into_one_vmap opt-ins
    against this repo's own real tests/benchmarks fixtures (which have
    genuinely ragged, staggered-recruitment per-subject/per-update history
    lengths -- see tests/benchmarks/test_combine_updates_into_one_vmap_benchmark.py).

    Identical to the original function except: the per-decision-time
    squared-residual term is multiplied by `mask` (1.0 = real row, 0.0 =
    self-padded row -- see self_pad_ragged_args_and_build_mask's own
    docstring) before being summed, so a self-padded (repeated last real
    row) row contributes exactly zero regardless of its (real, non-zero,
    never-fabricated) content. On real, unpadded data (mask all-ones) this
    is bit-for-bit identical to the original function -- the appended mask
    argument is the only difference.
    """
    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )
    residual_sq = (
        rewards
        - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
        - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
    ) ** 2
    masked_sum = jnp.einsum("ij,i->", residual_sq, mask)
    return masked_sum + jnp.dot(beta_est, beta_est) * lambda_ / n
