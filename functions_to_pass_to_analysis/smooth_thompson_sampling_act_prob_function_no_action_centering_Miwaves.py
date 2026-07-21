import jax
import jax.numpy as jnp
import jax.scipy.special as special
import pickle as pkl
from jax import lax
from jax import debug


### changes from Synthetic dataset: smooth_thompson_sampling_act_prob_function_no_action_centering_twoarmed.py
"""
- C_logistic is set to 5
- RANDOM_VARS is loaded from Miwaves/randomvars.pkl
- RANDOM_VARS is a (5000,) array of random variables from a normal distribution with mean 0 and var 1
- RANDOM_VARS is used to sample from the distribution of the advantage features
- RANDOM_VARS is used to sample from the distribution of the advantage features
"""


def load_random_vars() -> jnp.array:
    RANDOMVARS_PATH = './Miwaves/randomvars.pkl'
    # Load random variables - normal with mean 0 and var 1
    with open(RANDOMVARS_PATH, "rb") as f:
        random_vars = pkl.load(f)
    return jnp.array(random_vars)

# 
RANDOM_VARS = load_random_vars() # (5000,)
# RANDOM_VARS = jax.random.normal(jax.random.PRNGKey(0), (10000,))

C_logistic = 5  # Miwaves


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
    return prob


### still allow the function of two-armed trials.
def smooth_thompson_sampling_act_prob_function_no_action_centering_Miwaves(
    beta: jnp.ndarray,
    advantage: jnp.ndarray,
    num_users_entered_before_last_update: int,
    lower_clip: float,
    upper_clip: float,
    steepness: float,
    Z_id: int,
) -> float:
    """
    This function calculates the probability of taking action 1 given a user's "advantage" features
    and the model parameters "beta". This is intended to match up with what occurs in Oralytics,
    substituting in a sample mean for an expectation calculated by numerical integration.
    
    note: this function is also used in the inference phase
    
    """

    # prob for each user in jax.vmap function
    def zero_branch(_):
        return jnp.array(0.5)
    # if Z_id == 0: # error because each Element in Z_id is an jax.array
    #     return jnp.array(0.5)
    def active_branch(_):
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
        # print(f"act_prob: {act_prob}")
        # print(f"adv_beta_mean: {adv_beta_mean}")
        # print(f"adv_beta_var: {adv_beta_var}")
        # breakpoint()
        return act_prob
    
    return lax.cond(jnp.asarray(Z_id) == 0, zero_branch, active_branch, operand=None)