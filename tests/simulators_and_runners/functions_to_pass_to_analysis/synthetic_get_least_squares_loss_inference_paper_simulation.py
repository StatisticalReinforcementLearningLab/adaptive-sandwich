from jax import numpy as jnp


def synthetic_get_least_squares_loss_inference_paper_simulation(
    theta_est,
    intercept,
    past_reward,
    action,
    reward,
):
    features = jnp.hstack((intercept, past_reward, action))

    return jnp.sum((reward - jnp.matmul(features, theta_est.reshape(-1, 1))) ** 2)
