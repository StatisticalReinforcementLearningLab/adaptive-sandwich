import jax
from jax import numpy as jnp


def synthetic_get_action_1_prob_generalized_logistic(
    beta_est, lower_clip, steepness, upper_clip, treat_states
):
    treat_est = beta_est[-len(treat_states) :]
    lin_est = jnp.matmul(treat_states, treat_est)

    raw_prob = jax.scipy.special.expit(steepness * lin_est)

    return lower_clip + (upper_clip - lower_clip) * raw_prob
