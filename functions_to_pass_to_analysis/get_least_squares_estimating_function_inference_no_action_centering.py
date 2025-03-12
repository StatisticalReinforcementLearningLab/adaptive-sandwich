from jax import numpy as jnp


def get_least_squares_estimating_function_inference_no_action_centering(
    theta_est,
    intercept,
    past_reward,
    action,
    reward,
):
    state = jnp.hstack((intercept, past_reward))
    theta_0 = theta_est[: state.shape[1]].reshape(-1, 1)
    theta_1 = theta_est[state.shape[1] :].reshape(-1, 1)

    return -2 * jnp.sum(
        (reward - jnp.matmul(state, theta_0) - jnp.matmul(action * state, theta_1))
        * jnp.hstack((state, action * state)),
        axis=0,
    )
