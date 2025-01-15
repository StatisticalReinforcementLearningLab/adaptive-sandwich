import jax.numpy as jnp


def miwaves_RL_estimating_function(
    beta: jnp.array,
    n_users: int,
    state: jnp.array,
    A_beta: jnp.array,
    A_beta_gamma: jnp.array,
    A_gamma: jnp.array,
    B_beta: jnp.array,
    B_gamma: jnp.array,
    action: jnp.array,
    act_prob: jnp.array,
    decision_times: jnp.array,
    rewards: jnp.array,
    mu_beta: jnp.array,
    sigma_beta_inv: jnp.array,
    sigma_gamma_inv: jnp.array,
    init_noise_var: float,
) -> jnp.ndarray:
    """
    Estimating function for the Bayesian Linear Mixed Effects Regression update
    """

    dim = 1

    lambda_t = beta[:dim].reshape(-1, 1)

    UEt_flat = beta[dim:]
    UEt = (
        jnp.zeros((dim, dim), dtype=jnp.float32).at[jnp.triu_indices(dim)].set(UEt_flat)
    )
    E_t = UEt + UEt.T - jnp.diag(jnp.diag(UEt))

    term1 = sigma_beta_inv @ mu_beta / n_users
    term2 = B_beta / init_noise_var
    term3 = (
        A_beta_gamma
        @ jnp.linalg.inv(A_gamma + init_noise_var * sigma_gamma_inv)
        @ B_gamma
        / init_noise_var
    )
    term4 = E_t @ lambda_t

    upper = term1 + term2 - term3 - term4
    upper = upper.flatten()

    term5 = jnp.triu(
        (sigma_beta_inv / n_users)
        + (A_beta / init_noise_var)
        - (
            A_beta_gamma
            @ jnp.linalg.inv(A_gamma + init_noise_var * sigma_gamma_inv)
            @ A_beta_gamma.T
        )
        / init_noise_var
    )
    term6 = jnp.triu(E_t)
    difference = term5 - term6

    triu_indices = jnp.triu_indices_from(difference)
    lower = difference[triu_indices].flatten()

    return -jnp.concatenate([upper, lower])
