import jax.numpy as jnp


def oralytics_primary_analysis_avg_reward_sum_loss_debug(
    theta_est,
    oscb,
):
    return 0.5 * (jnp.sum(oscb) - theta_est[0]) ** 2
