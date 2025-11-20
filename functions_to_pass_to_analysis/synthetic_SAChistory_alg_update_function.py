import jax.numpy as jnp
from jax import lax
import jax
# import numpy as np
from jax import debug

def synthetic_SAChistory_alg_update_function(
    beta: jnp.array, # beta_t: t>=2
    betaQ_target: jnp.array, # intialize one, which is fixed
    beta_pi_target: jnp.array, # intialize one, which is fixed
    n_users: int,  
    state: jnp.array, # up to the time step t
    next_state: jnp.array, # up to the time step t
    action: jnp.array, # up to the time step t
    next_action: jnp.array, # added
    rewards: jnp.array,
    lower_clip: float,
    upper_clip: float,
    steepness: float,
    ridge_penalty: float,
    gamma: float,
    Z_id: jnp.array,
) -> float:
    """
    Estimating function for SAC. this function is used only in the inference phase

    for historical update, beta_pi_target and betaQ_target are all the intial values without changing


    """
    def zero_branch(_):
        # zero beta estiamte and 0 gradient
        return jnp.zeros_like(beta)

    
    def policy(treat_states, beta_pi):
        logits = jnp.dot(treat_states, beta_pi)
        probs = lower_clip + (upper_clip - lower_clip) * jax.nn.sigmoid(steepness * logits) # [self.lower_clip, self.upper_clip]
        return probs
    
    def active_branch(_):
        dim = 4
        lambda_entropy = 1.0
        beta_Q = beta[:dim]
        beta_pi = beta[dim:]
 
        ##### estimation function for Q
        p_next = policy(next_state, beta_pi_target)  # we need to use the target policy to evaluate p_next in the last step
        Q_states_next = jnp.hstack([next_state, next_state * next_action.reshape(-1,1)]) # (t, 2) * (t, 1)
        Q_values_next = jnp.dot(Q_states_next, betaQ_target) 
        logp_next = jnp.log(jnp.where(next_action==1, p_next, jnp.clip(1.0 - p_next, 1e-8, 1.0))) 
        TD_target = rewards.flatten() + gamma * (Q_values_next - lambda_entropy * logp_next) 
        current_Q_states = jnp.hstack([state, action * state]) # (1, 4)
        Current_Q_values = jnp.dot(current_Q_states, beta_Q) # 
        residuals = jax.lax.stop_gradient(TD_target) - Current_Q_values
        # debug.print('beta_pi_target: {}', beta_pi_target)
        # debug.print('next_state: {}', next_state)
        # debug.print('next_action: {}', next_action)
        # debug.print('betaQ_tar: {}', betaQ_target)
        # debug.print('rewards: {}', rewards)
        # debug.print('Q_values_next: {}', Q_values_next)
        # debug.print('logp_next: {}', logp_next)
        # debug.print('betaQ: {}', beta_Q)
        # debug.print('TD_target (wrong): {}', TD_target)
        # debug.print('Current_Q_values: {}', Current_Q_values)
        # debug.print('current_Q_states: {}', current_Q_states)
        # debug.print('residuals: {}', residuals)

        vector_Q = -2*current_Q_states * residuals.reshape(-1,1)  # [t, 4] * [t, 1] -> [t, 4]
        vector_Q = jnp.mean(vector_Q, axis=0).reshape(-1, 1)  # [4, 1] average over t
        vector_Q =  vector_Q +  2 * ridge_penalty * beta_Q.reshape(-1, 1)  # [4, 1]  
        debug.print("vector_Q for each unit = {}", vector_Q)
        
        ##### estimation function for pi
        p0 = jax.nn.sigmoid(steepness * jnp.dot(state, beta_pi)).reshape(-1, 1)
        p = policy(state, beta_pi).reshape(-1, 1)  # (num_decision_times, 1)
        temp = jnp.dot(state, betaQ_target[int(dim/2):]).reshape(-1, 1) - lambda_entropy * jnp.log(p/(1-p))
        vector_pi = (1-2*lower_clip) * p0 * (1 - p0) * steepness * temp.reshape(-1, 1) * state # (t, 1) * (t, 2) -> (t, 2)
        vector_pi = jnp.mean(vector_pi, axis=0).reshape(-1, 1)  # (t, 2) -> (1, 2) -> (2, 1)
        debug.print("vector_pi for each unit = {}", vector_pi)
        return jnp.concatenate([vector_Q.flatten(), vector_pi.flatten()]) # (4,) + (2,) = (6,)

    z = jnp.asarray(Z_id)
    pred = jnp.all(z == 0)
    return lax.cond(pred, zero_branch, active_branch, operand=None)



