import jax
import jax.numpy as jnp
import jax.scipy.special as special

RANDOM_VARS = jax.random.normal(jax.random.PRNGKey(0), (10000,))

C_logistic = 1  # Centers at .5 for arg 0


def logistic_function(x: float, L_min: float, L_max: float, steepness: float) -> float:

    numerator = L_max - L_min
    denominator_inverse = special.expit(steepness * x - jnp.log(C_logistic))
    return L_min + numerator * denominator_inverse


def allocation_function_monte_carlo(
    mean: float, var: float, L_min: float, L_max: float, steepness: float
) -> float:

    std = jnp.sqrt(var)
    samples = mean + (RANDOM_VARS * std)
    prob = jnp.mean(logistic_function(samples, L_min, L_max, steepness))

    # the above is replacing the following non-JAX-differentiable code:
    # prob = stats.norm.expect(func=logistic_function, loc=mean, scale=np.sqrt(var))

    return prob


def allocation_function_mean_field_approx(
    mean: float, var: float, L_min: float, L_max: float, steepness: float
) -> float:
    scale = jnp.sqrt(1 + (jnp.pi * steepness**2 * var) / 8)
    effective_mean = steepness * mean - jnp.log(C_logistic)
    prob = L_min + (L_max - L_min) * special.expit(effective_mean / scale)
    return prob.reshape()


def smooth_thompson_sampling_act_prob_function_no_action_centering(
    beta: jnp.ndarray,
    advantage: jnp.ndarray,
    num_users_entered_before_last_update: int,
    lower_clip: float,
    upper_clip: float,
    steepness: float,
) -> float:
    """
    This function calculates the probability of taking action 1 given a user's "advantage" features
    and the model parameters "beta". This is intended to match up with what occurs in Oralytics,
    substituting in a sample mean for an expectation calculated by numerical integration.
    """

    n_params = len(advantage) # 1
    total_feature_dim = n_params * 2  # only true because no action centering

    mu = beta[:total_feature_dim].reshape(-1, 1)
    utvar_inv_terms = (
        jax.lax.max(num_users_entered_before_last_update, 1) * beta[total_feature_dim:]
    )
    idx = jnp.triu_indices(total_feature_dim)
    utvar_inv = (
        jnp.zeros((total_feature_dim, total_feature_dim), dtype=jnp.float32)
        .at[idx]
        .set(utvar_inv_terms)
    )
    var_inv = utvar_inv + utvar_inv.T - jnp.diag(jnp.diag(utvar_inv))
    var = jnp.linalg.inv(var_inv)

    mu_adv = mu[-n_params:]
    var_adv = var[-n_params:, -n_params:] # [100000, 0; 0, 100000]

    adv_beta_mean = advantage.T.dot(mu_adv)
    adv_beta_var = advantage.T @ var_adv @ advantage

    act_prob = allocation_function_monte_carlo(
        mean=adv_beta_mean,
        var=adv_beta_var,
        L_min=lower_clip,
        L_max=upper_clip,
        steepness=steepness,
    )

    return act_prob

# def smooth_thompson_sampling_act_prob_function_no_action_centering_partial(
#     beta: jnp.ndarray,
#     advantage: jnp.ndarray,
#     num_users_entered_before_last_update: int,
#     lower_clip: float,
#     upper_clip: float,
#     steepness: float,
#     Z_id: int,
# ) -> float:
#     """
#     This function calculates the probability of taking action 1 given a user's "advantage" features
#     and the model parameters "beta". This is intended to match up with what occurs in Oralytics,
#     substituting in a sample mean for an expectation calculated by numerical integration.
#     """

#     # prob for each user in jax.vmap function
#     if Z_id == 0:
#         return jnp.array(0.5)
    
#     n_params = len(advantage) # 1
#     total_feature_dim = n_params * 2  # only true because no action centering

#     mu = beta[:total_feature_dim].reshape(-1, 1)
#     utvar_inv_terms = (
#         jax.lax.max(num_users_entered_before_last_update, 1) * beta[total_feature_dim:]
#     )
#     idx = jnp.triu_indices(total_feature_dim)
#     utvar_inv = (
#         jnp.zeros((total_feature_dim, total_feature_dim), dtype=jnp.float32)
#         .at[idx]
#         .set(utvar_inv_terms)
#     )
#     var_inv = utvar_inv + utvar_inv.T - jnp.diag(jnp.diag(utvar_inv))
#     var = jnp.linalg.inv(var_inv)

#     mu_adv = mu[-n_params:]
#     var_adv = var[-n_params:, -n_params:] # [100000, 0; 0, 100000]

#     adv_beta_mean = advantage.T.dot(mu_adv)
#     adv_beta_var = advantage.T @ var_adv @ advantage

#     act_prob = allocation_function_monte_carlo(
#         mean=adv_beta_mean,
#         var=adv_beta_var,
#         L_min=lower_clip,
#         L_max=upper_clip,
#         steepness=steepness,
#     )
#     return act_prob