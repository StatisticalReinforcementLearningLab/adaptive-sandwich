import jax.numpy as jnp
from jax import lax

def synthetic_BLR_estimating_function_no_action_centering_partial(
    beta: jnp.array,
    n_users: int,  # Note this is the number of users that have entered the study *so far*
    state: jnp.array,
    action: jnp.array,
    rewards: jnp.array,
    prior_mu: jnp.array,
    prior_sigma_inv: jnp.array,
    init_noise_var: float,
    Z_id: jnp.array,
) -> float:
    """
    Estimating function for the Bayesian Linear Regression update;
    this function is used only in the inference phase
    """
    def zero_branch(_):
        # zero beta estiamte and 0 gradient
        return jnp.zeros_like(beta)

    # print('state shape:', state.shape)
    # print('Z_id in active branch:', Z_id)
    def active_branch(_):
        dim = 4

        mu = beta[:dim].reshape(-1, 1)

        utv_terms = beta[dim:]
        # Note this is row-major, moving left to right across rows in sequence
        idx = jnp.triu_indices(dim)
        UTV = jnp.zeros((dim, dim), dtype=jnp.float32).at[idx].set(utv_terms)
        V = UTV + UTV.T - jnp.diag(jnp.diag(UTV))

        pmu = prior_mu.reshape(-1, 1)

        # Construct stacked_phis by the following:
        # stacked_phis = [state, (act_prob * state), (action - act_prob) * state] by stacking vertically
        # state is a 2D array of shape (num_decision_times, 5)
        # act_prob is a 2D array of shape (num_decision_times, 1)
        # action is a 2D array of shape (num_decision_times, 1)
        # stacked_phis should be a 2D array of shape (num_decision_times, 15)

        stacked_phis = jnp.hstack([state, action * state])

        # Note that the sum over t in the math is captured by matrix multiplication
        # with phis stacked together.
        vector_1 = (
            stacked_phis.T @ (rewards.reshape(-1, 1) - stacked_phis @ mu)
        ).flatten() / init_noise_var

        vector_2 = prior_sigma_inv @ (pmu - mu).flatten() / n_users

        matrix_3 = jnp.triu(stacked_phis.T @ stacked_phis) / init_noise_var + (
            jnp.triu(prior_sigma_inv) / n_users - jnp.triu(V)
        )
        triu_indices = jnp.triu_indices_from(matrix_3)
        vector_3 = matrix_3[triu_indices].flatten()
        # print('vector_1 shape:', vector_1.shape)
        # print('vector_2 shape:', vector_2.shape)
        # print('vector_3 shape:', vector_3.shape)
        # Must be same shape as beta input (so 1D)
        # The negative sign is not necessary, but makes this strictly the derivative
        # of the corresponding loss function as implemented.  This is useful for testing.
        return -jnp.concatenate([vector_1 + vector_2, vector_3])

    
    # z = jnp.asarray(Z_id).reshape(())  
    # pred = (z == 0)
    z = jnp.asarray(Z_id)
    pred = jnp.all(z == 0)
    return lax.cond(pred, zero_branch, active_branch, operand=None)
