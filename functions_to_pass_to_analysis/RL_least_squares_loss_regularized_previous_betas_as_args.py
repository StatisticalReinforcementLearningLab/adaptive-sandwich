import jax
from jax import numpy as jnp

from functions_to_pass_to_analysis.synthetic_get_action_1_prob_generalized_logistic import (
    synthetic_get_action_1_prob_generalized_logistic,
)


@jax.jit
def RL_least_squares_loss_regularized_previous_betas_as_args(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    real_action1probs,
    pre_update_action1probs,
    previous_post_update_betas,
    # gives policy nums for each post-update decision time for which
    # arguments are being provided.
    post_update_policy_nums,
    lower_clip,
    steepness,
    upper_clip,
    action_centering,
    lambda_,
    n,
):

    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    # TODO: Use vmap to speed this up if necessary.
    action1probs = jnp.concatenate(
        [
            pre_update_action1probs,
            jnp.array(
                [
                    synthetic_get_action_1_prob_generalized_logistic(
                        # note the policy_num - ... since post update betas
                        # correspond to the policies starting from the first
                        # post-update policy
                        previous_post_update_betas[
                            policy_num - post_update_policy_nums[0]
                        ].flatten(),
                        lower_clip,
                        steepness,
                        upper_clip,
                        # treat states start from the first decision time,
                        # limit to post update to align
                        treat_states[i + pre_update_action1probs.shape[0]],
                    )
                    for i, policy_num in enumerate(post_update_policy_nums)
                ]
            ).reshape(-1, 1),
        ]
    )

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )
    return (
        jnp.einsum(
            "ij->",
            (
                rewards
                - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
                - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
            )
            ** 2,
        )
        + jnp.dot(
            beta_est,
            beta_est,
        )
        * lambda_
        / n
    )
