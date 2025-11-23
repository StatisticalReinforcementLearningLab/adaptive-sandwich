import jax
import jax.numpy as jnp
import jax.scipy.special as special

RANDOM_VARS = jax.random.normal(jax.random.PRNGKey(0), (50,))
L_max = 0.8
L_min = 0.2
C_logistic = 3
B = 0.515


def logistic_function(x: float) -> float:

    numerator = L_max - L_min
    denominator_inverse = special.expit(B * x - jnp.log(C_logistic))
    return L_min + numerator * denominator_inverse


def allocation_function_monte_carlo(mean: float, var: float) -> float:

    std = jnp.sqrt(var)
    samples = mean + (RANDOM_VARS * std)
    prob = jnp.mean(logistic_function(samples))

    # the above is replacing the following non-JAX-differentiable code:
    # prob = stats.norm.expect(func=logistic_function, loc=mean, scale=np.sqrt(var))

    return prob


def allocation_function_mean_field_approx(mean: float, var: float) -> float:
    scale = jnp.sqrt(1 + (jnp.pi * B**2 * var) / 8)
    effective_mean = B * mean - jnp.log(C_logistic)
    prob = L_min + (L_max - L_min) * special.expit(effective_mean / scale)
    return prob.reshape()


# NOTE: If you change the number of features in the simulator, the value of "dim" here will need
# to be updated to match the new number of features.
TOTAL_FEATURE_DIM = 15


def oralytics_act_prob_function(
    beta: jnp.ndarray,
    advantage: jnp.ndarray,
    total_feature_dim: int,
    num_users_entered_before_last_update: int,
) -> float:
    """
    This function calculates the probability of taking action 1 given a user's "advantage" features
    and the model parameters "beta". This is intended to match up with what occurs in Oralytics,
    substituting in a sample mean for an expectation calculated by numerical integration.
    """

    n_params = len(advantage)

    # We don't actually use the total_feature_dim argument so that this function can be
    # JIT-compiled, but still could use it to check whether TOTAL_FEATURE_DIM is correct.
    # TODO: Figure out a way to actually check that they are equal...

    mu = beta[:TOTAL_FEATURE_DIM].reshape(-1, 1)
    utvar_inv_terms = (
        jax.lax.max(num_users_entered_before_last_update, 1) * beta[TOTAL_FEATURE_DIM:]
    )
    idx = jnp.triu_indices(TOTAL_FEATURE_DIM)
    utvar_inv = (
        jnp.zeros((TOTAL_FEATURE_DIM, TOTAL_FEATURE_DIM), dtype=jnp.float32)
        .at[idx]
        .set(utvar_inv_terms)
    )
    var_inv = utvar_inv + utvar_inv.T - jnp.diag(jnp.diag(utvar_inv))

    mu_adv = mu[-n_params:]
    adv_beta_mean = advantage.T.dot(mu_adv)

    # OPTIMIZATION: Compute advantage.T @ var_adv @ advantage using solve
    # Pad advantage with zeros for non-advantage dimensions
    advantage_padded = jnp.zeros(TOTAL_FEATURE_DIM)
    advantage_padded = advantage_padded.at[-n_params:].set(advantage)

    # Solve var_inv @ x = advantage_padded to get x = var @ advantage_padded
    x = jnp.linalg.solve(var_inv, advantage_padded)

    # advantage_padded.T @ var @ advantage_padded = advantage_padded.T @ x
    adv_beta_var = jnp.dot(advantage_padded, x)

    act_prob = allocation_function_monte_carlo(mean=adv_beta_mean, var=adv_beta_var)

    return act_prob
