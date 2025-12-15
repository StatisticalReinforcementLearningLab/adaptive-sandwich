import jax
from jax import numpy as jnp


@jax.jit
def RL_least_squares_loss_regularized(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action1probtimes,  # pylint: disable=unused-argument
    action_centering,
    lambda_,
    n,
):

    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )
    return (
        jnp.einsum(
            "ij->",
            (
                rewards
                - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
                - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
            )
            ** 2,
        )
        + jnp.dot(
            beta_est,
            beta_est,
        )
        * lambda_
        / n
    )
