import jax.numpy as jnp


def oralytics_RL_loss(
    beta: jnp.array,
    n_users: int,
    state: jnp.array,
    action: jnp.array,
    act_prob: jnp.array,
    decision_times: jnp.array,  # pylint: disable=unused-argument
    rewards: jnp.array,
    prior_mu: jnp.array,
    prior_sigma_inv: jnp.array,
    init_noise_var: float,
) -> float:
    """
    Compute the loss function of the thompson sampling update
    """

    dim = 15

    mu = beta[:dim].reshape(-1, 1)
    V = beta[dim:].reshape(-1, dim)

    prior_mu = prior_mu.reshape(-1, 1)

    # Construct phi by the following:
    # phi = [state, (act_prob * state), (action - act_prob) * state] by stacking vertically
    # state is a 2D array of shape (num_decision_times, 5)
    # act_prob is a 2D array of shape (num_decision_times, 1)
    # action is a 2D array of shape (num_decision_times, 1)
    # phi should be a 2D array of shape (num_decision_times, 15)
    phi = jnp.hstack([state, (act_prob * state), (action - act_prob) * state])

    term1 = jnp.sum((rewards - jnp.einsum("ij,jk->i", phi, mu)) ** 2) / (
        2 * init_noise_var
    )

    term2 = (
        (prior_mu - mu).T @ prior_sigma_inv @ (prior_mu - mu) / (2 * n_users)
    ).squeeze()

    term3 = (
        jnp.sum(
            (
                jnp.triu(phi.T @ phi)
                + (jnp.triu(prior_sigma_inv) / n_users)
                - jnp.triu(V)
            )
        )
        ** 2
    ) / (2 * init_noise_var)

    total_sum = term1 + term2 + term3

    print(total_sum)

    return total_sum
