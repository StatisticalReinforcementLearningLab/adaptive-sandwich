import jax.numpy as jnp


def oralytics_RL_loss(
    beta: jnp.array,
    n_users: int,
    # phi: jnp.array,
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
    Compute the loss function of the Bayesian linear regression update for the RL model.
    """

    dim = 15

    mu = beta[:dim].reshape(-1, 1)

    utv_terms = beta[dim:]
    idx = jnp.triu_indices(dim)
    UTV = jnp.zeros((dim, dim), dtype=jnp.float32).at[idx].set(utv_terms)

    V = UTV + UTV.T - jnp.diag(jnp.diag(UTV))

    prior_mu = prior_mu.reshape(-1, 1)

    # Construct phi by the following:
    # phi = [state, (act_prob * state), (action - act_prob) * state] by stacking vertically
    # state is a 2D array of shape (num_decision_times, 5)
    # act_prob is a 2D array of shape (num_decision_times, 1)
    # action is a 2D array of shape (num_decision_times, 1)
    # phi should be a 2D array of shape (num_decision_times, 15)

    # Check done, this is correct
    phi = jnp.hstack([state, (act_prob * state), (action - act_prob) * state])

    term1 = jnp.sum((rewards - jnp.einsum("ij,jk->i", phi, mu)) ** 2) / (
        2 * init_noise_var
    )

    term2 = (
        (prior_mu - mu).T @ prior_sigma_inv @ (prior_mu - mu) / (2 * n_users)
    ).squeeze()

    term3 = jnp.sum(
        (jnp.triu(phi.T @ phi) + (jnp.triu(prior_sigma_inv) / n_users) - jnp.triu(V))
        ** 2
    ) / (2 * init_noise_var)

    total_sum = term1 + term2 + term3

    return total_sum
