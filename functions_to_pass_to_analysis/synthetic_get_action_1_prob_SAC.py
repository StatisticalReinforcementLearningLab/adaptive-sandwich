import jax
from jax import numpy as jnp
from jax import lax


def synthetic_get_action_1_prob_SAC(
    beta_est, # (n, 4+2) 
    lower_clip, 
    steepness, 
    upper_clip, 
    treat_states, # (n, 2) 
    Z_id # (n, 1)
):
    
    def zero_branch(_):
        return jnp.array(0.5)
    
    # prob for each user in jax.vmap function
    def policy(treat_states, beta_pi):
        logits = jnp.dot(treat_states, beta_pi)
        probs = lower_clip + (upper_clip - lower_clip) * jax.nn.sigmoid(steepness * logits) # [self.lower_clip, self.upper_clip]
        return probs

    def active_branch(_):
        treat_est = beta_est[-len(treat_states) :] # (2, )
        raw_prob = policy(treat_states, treat_est) # (n,2)*(2,) -> (n,)
        return raw_prob[()]
    
    return lax.cond(jnp.asarray(Z_id) == 0, zero_branch, active_branch, operand=None) # (n, )

