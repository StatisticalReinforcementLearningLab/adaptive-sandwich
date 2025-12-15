from jax import numpy as jnp


def synthetic_get_least_squares_loss_inference_action_centering(
    theta_est,
    intercept,
    past_reward,
    action,
    reward,
    action1prob,
):
    state = jnp.hstack((intercept, past_reward))
    theta_0 = theta_est[: state.shape[1]].reshape(-1, 1)
    theta_1 = theta_est[state.shape[1] :].reshape(-1, 1)

    action = action.astype(jnp.float32) - action1prob

    return jnp.sum(
        (reward - jnp.matmul(state, theta_0) - jnp.matmul(action * state, theta_1)) ** 2
    )
