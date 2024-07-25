from jax import numpy as jnp


def get_least_squares_loss_rl(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action_centering,
):
    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )

    return jnp.sum(
        (
            rewards
            - jnp.matmul(base_states, beta_0_est)
            - jnp.matmul(actions * treat_states, beta_1_est)
        )
        ** 2,
    )
