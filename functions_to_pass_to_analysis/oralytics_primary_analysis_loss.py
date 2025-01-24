import jax.numpy as jnp


def oralytics_primary_analysis_loss(
    theta_est,
    tod,
    bbar,
    abar,
    appengage,
    bias,
    action,
    oscb,
    act_prob,
):

    features = jnp.hstack(
        (tod, bbar, abar, appengage, bias, act_prob, action - act_prob)
    )

    weight = 1 / (act_prob * (1 - act_prob))

    return 0.5 * jnp.sum(
        weight * ((oscb - jnp.matmul(features, theta_est).reshape(-1, 1)) ** 2)
    )
