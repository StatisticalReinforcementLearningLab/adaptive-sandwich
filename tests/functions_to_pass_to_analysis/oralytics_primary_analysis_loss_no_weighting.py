import jax.numpy as jnp


def oralytics_primary_analysis_loss_no_weighting(
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

    return 0.5 * jnp.sum((oscb - jnp.matmul(features, theta_est).reshape(-1, 1)) ** 2)
