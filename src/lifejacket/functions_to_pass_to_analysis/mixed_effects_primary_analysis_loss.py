import jax.numpy as jnp


def mixed_effects_primary_analysis_loss(
    theta_est,
    action,
    outcome,
    action_probability,
):

    nu_1 = theta_est[0]
    delta = theta_est[1]

    weight = 1 / (action_probability * (1 - action_probability))

    return jnp.sum(
        weight
        * (outcome - action_probability * nu_1 - (action - action_probability) * delta)
        ** 2
    )
