import jax.numpy as jnp


def primary_analysis_avg_reward_sum_loss(
    theta_est,
    reward,
    calendar_t,
    user_id,
):

    num_users = jnp.unique(user_id).size
    # Sort of assumes no incremental recruitment, though it doesn't really
    # matter
    num_decision_times = jnp.unique(calendar_t).size

    return 0.5 * (jnp.sum(reward) / num_users / num_decision_times - theta_est[0]) ** 2
