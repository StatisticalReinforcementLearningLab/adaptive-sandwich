import jax.numpy as jnp


def oralytics_RL_estimating_function(
    beta: jnp.array,
    n_users: int,  # Note this is the number of users that have entered the study *so far*
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
    Estimating function for the Bayesian Linear Regression update
    """

    dim = 15

    mu = beta[:dim].reshape(-1, 1)

    utv_terms = beta[dim:]
    # Note this is row-major, moving left to right across rows in sequence
    idx = jnp.triu_indices(dim)
    UTV = jnp.zeros((dim, dim), dtype=jnp.float32).at[idx].set(utv_terms)
    V = UTV + UTV.T - jnp.diag(jnp.diag(UTV))

    prior_mu = prior_mu.reshape(-1, 1)

    # Construct stacked_phis by the following:
    # stacked_phis = [state, (act_prob * state), (action - act_prob) * state] by stacking vertically
    # state is a 2D array of shape (num_decision_times, 5)
    # act_prob is a 2D array of shape (num_decision_times, 1)
    # action is a 2D array of shape (num_decision_times, 1)
    # stacked_phis should be a 2D array of shape (num_decision_times, 15)

    stacked_phis = jnp.hstack([state, (act_prob * state), (action - act_prob) * state])

    # Note that the sum over t in the math is captured by matrix multiplication
    # with phis stacked together.
    vector_1 = (
        stacked_phis.T @ (rewards.reshape(-1, 1) - stacked_phis @ mu)
    ).flatten() / init_noise_var

    vector_2 = prior_sigma_inv @ (prior_mu - mu).flatten() / n_users

    matrix_3 = jnp.triu(stacked_phis.T @ stacked_phis) / init_noise_var + (
        jnp.triu(prior_sigma_inv) / n_users - jnp.triu(V)
    )
    triu_indices = jnp.triu_indices_from(matrix_3)
    vector_3 = matrix_3[triu_indices].flatten()

    # Must be same shape as beta input (so 1D)
    # The negative sign is not necessary, but makes this strictly the derivative
    # of the corresponding loss function as implemented.  This is useful for testing.
    return -jnp.concatenate([vector_1 + vector_2, vector_3])
