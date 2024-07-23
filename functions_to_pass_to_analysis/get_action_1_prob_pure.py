import jax
from jax import numpy as jnp


def get_action_1_prob_pure(beta_est, lower_clip, steepness, upper_clip, treat_states):
    treat_est = beta_est[-len(treat_states) :]
    lin_est = jnp.matmul(treat_states, treat_est)
    raw_prob = jax.scipy.special.expit(steepness * lin_est)

    return jnp.clip(raw_prob, lower_clip, upper_clip)[()]
