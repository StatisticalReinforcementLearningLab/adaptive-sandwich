import jax.numpy as jnp


def oralytics_primary_analysis_loss_no_action_probs(
    theta_est,
    tod,
    bbar,
    abar,
    appengage,
    bias,
    oscb,
    act_prob,
):

    features = jnp.hstack((tod, bbar, abar, appengage, bias, act_prob))

    return 0.5 * jnp.sum((oscb - jnp.matmul(features, theta_est).reshape(-1, 1)) ** 2)
