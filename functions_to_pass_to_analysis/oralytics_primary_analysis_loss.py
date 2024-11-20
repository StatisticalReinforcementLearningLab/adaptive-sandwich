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

    nu_1 = theta_est[:5]
    nu_2 = theta_est[5]
    delta = theta_est[6]

    state = jnp.hstack(
        (
            tod,
            bbar,
            abar,
            appengage,
            bias,
        )
    )

    # weight = 1 / (act_prob * (1 - act_prob))
    weight = 1

    return jnp.sum(
        weight
        * (
            oscb
            - jnp.matmul(state, nu_1)
            - act_prob * nu_2
            - (action - act_prob) * delta
        )
        ** 2
    )
