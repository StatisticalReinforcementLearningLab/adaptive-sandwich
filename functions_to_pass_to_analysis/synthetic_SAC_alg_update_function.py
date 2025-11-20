import jax.numpy as jnp
from jax import lax
import jax
# import numpy as np
from jax import debug

def synthetic_SAC_alg_update_function(
    beta: jnp.array, # beta_t: t>=2
    betaQ_target: jnp.array, # added
    beta_pi_target: jnp.array, # added
    n_users: int,  # Note this is the number of users that have entered the study *so far*
    state: jnp.array,
    next_state: jnp.array, # added
    action: jnp.array,
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
        # print('----------------------------------------------------------')
        # debug.print("beta = {}", beta)
        # debug.print("states = {}", state)
        # debug.print("betaQ_target = {}", betaQ_target)
        ##### estimation function for Q
        # print('state shape:', state.shape, state)
        p_next = policy(next_state, beta_pi_target)  # we need to use the target policy to evaluate p_next in the last step
        # next_action
        Q_states_next = jnp.hstack([next_state, next_state * next_action])
        Q_values_next = jnp.dot(Q_states_next, betaQ_target) 
        logp_next = jnp.log(jnp.where(next_action==1, p_next, jnp.clip(1.0 - p_next, 1e-8, 1.0))) 
        TD_target = rewards + gamma * (Q_values_next - lambda_entropy * logp_next) 
        # debug.print("p_next: {}", p_next)
        # debug.print("rewards: {}", rewards)
        # debug.print("gamma: {}", gamma)
        # debug.print("lambda_entropy: {}", lambda_entropy)
        # debug.print("next_action: {}", next_action)
        # debug.print("Q_values_next: {}", Q_values_next)
        # debug.print("logp_next: {}", logp_next)
        current_Q_states = jnp.hstack([state, action * state]) # (1, 4)
        Current_Q_values = jnp.dot(current_Q_states, beta_Q) # 
       
        # debug.print("Current_Q_values shape: {}", Current_Q_values)
        # debug.print("beta: {}", beta)
        # debug.print('TD_target : {}', TD_target)
        residuals = jax.lax.stop_gradient(TD_target) - Current_Q_values
        # debug.print('residuals : {}', residuals)
        # debug.print("current_Q_states: {}", current_Q_states)
        vector_Q = -2*jnp.dot(current_Q_states.T, residuals.reshape(-1,1)) # [4, 1] * [1, 1] -> [4, 1]  
        vector_Q =  vector_Q +  2 * ridge_penalty * beta_Q.reshape(-1, 1)  # [4, 1]  
        # debug.print('beta_Q : {}', beta_Q)
        debug.print("vector_Q for each unit = {}", vector_Q)
        
        ##### estimation function for pi
        p0 = jax.nn.sigmoid(steepness * jnp.dot(state, beta_pi)).reshape(-1, 1)
        p = policy(state, beta_pi).reshape(-1, 1)  # (num_decision_times, 1)
        temp = jnp.dot(state, betaQ_target[int(dim/2):]) - lambda_entropy * jnp.log(p/(1-p))
        
        vector_pi = (1-2*lower_clip) * p0 * (1 - p0) * steepness * temp.reshape(-1, 1) * state
        debug.print("vector_pi for each unit = {}", -vector_pi)
        return jnp.concatenate([vector_Q.flatten(), vector_pi.flatten()]) # (4,) + (2,) = (6,)

    z = jnp.asarray(Z_id)
    pred = jnp.all(z == 0)
    return lax.cond(pred, zero_branch, active_branch, operand=None)



