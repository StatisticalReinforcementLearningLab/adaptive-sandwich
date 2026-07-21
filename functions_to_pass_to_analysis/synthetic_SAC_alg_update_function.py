import jax.numpy as jnp
from jax import lax
import jax
from jax import debug


def synthetic_SAC_alg_update_function(
    beta: jnp.array, # the current updated beta_t: t>=2 (4+2, 1)
    beta_previous: jnp.array, # all updated beta_{:t}: t>=2 (t-1, 4+2)
    n_users: int,  # Note this is the number of users that have entered the study *so far*
    state: jnp.array, # (1, 2)
    next_state: jnp.array, # (1, 2)
    action: jnp.array, # (1, 1)
    next_action: jnp.array, # (1, 1)
    rewards: jnp.array, # (1, 1)
    lower_clip: float,
    upper_clip: float,
    steepness: float,
    ridge_penalty: float,
    constant_ridge: int,
    actor_ridge_penalty: float,
    gamma: float,
    Z_id: jnp.array,
    beta_initial: jnp.array, # not allow the gradient trace
    lambda_entropy: float,
) -> float:
    """
    Estimating function for SAC. this function is used only in the inference phase, and computation for each user in the parallel Jax computational graph.
    """
    def zero_branch(_):
        # zero beta estiamte and 0 gradient
        return jnp.zeros_like(beta)

    
    def policy(treat_states, beta_pi):
        logits = jnp.dot(treat_states, beta_pi)
        probs = lower_clip + (upper_clip - lower_clip) * jax.nn.sigmoid(steepness * logits) # [self.lower_clip, self.upper_clip]
        return probs
    
    def active_branch(_):
        dim = 4 # syn
        beta_Q = beta[:dim] # (4,)
        beta_pi = beta[dim:] # (2,)
        
        if beta_previous.shape[0] == 1:
            beta_target = beta_initial # the target should the initial beta to evaluate the estimating function
        else:
            beta_target = beta_previous[-2,:] # -1: the current beta, -2: the previous beta 

        betaQ_target = beta_target[:dim] # previous Q (4,)
        betapi_target = beta_target[dim:] # previous pi (2,)

        ##### estimation function for Q
        p_next = policy(next_state, betapi_target) # (1,)
        
        # method 1: sample-based evaluation: Q(S_{t+1}, A_{t+1})
        # Q_states_next = jnp.hstack([next_state, next_state * next_action]) # (1,4)
        # Q_values_next = jnp.dot(Q_states_next, betaQ_target) # (1,)
        # logp_next = jnp.log(jnp.where(next_action==1, p_next, 1.0 - p_next)) 
        
        # method 2: expectation-based evaluation: E[Q(S_{t+1}, A_{t+1})]
        Q_state_next1 = jnp.hstack([next_state, next_state * jnp.ones_like(next_action)]) # (1, 4)
        Q_state_next0 = jnp.hstack([next_state, next_state * jnp.zeros_like(next_action)]) # (1, 4)
        Q_value_next1 = jnp.dot(Q_state_next1, betaQ_target) # (1,)
        Q_value_next0 = jnp.dot(Q_state_next0, betaQ_target) # (1,)
        Q_values_next = p_next * Q_value_next1 + (1.0 - p_next) * Q_value_next0 # (1,)
        logp_next = p_next * jnp.log(p_next) + (1.0 - p_next) * jnp.log(1.0 - p_next) # (1,)
        
        TD_target = rewards + gamma * (Q_values_next - lambda_entropy * logp_next) 
        current_Q_states = jnp.hstack([state, action * state]) # (1, 4)
        Current_Q_values = jnp.dot(current_Q_states, beta_Q) # (1,)
        # residuals = jax.lax.stop_gradient(TD_target) - Current_Q_values # (1,1)
        residuals = TD_target - Current_Q_values # (1,1) allow the gradient to flow through the beta_previous
        vector_Q = -2*current_Q_states * residuals.reshape(-1,1)  # (1, 4) * (1, 1) -> (1, 4)
        vector_Q = vector_Q.reshape(-1, 1) # (4, 1)
        # vector_Q = jnp.mean(vector_Q, axis=0).reshape(-1, 1)  # [4, 1] average over t
        if constant_ridge:
            vector_Q =  vector_Q +  2 * ridge_penalty * beta_Q.reshape(-1, 1)  # (4, 1)  
        else:
            vector_Q =  vector_Q +  2 * ridge_penalty / n_users * beta_Q.reshape(-1, 1)  # (4, 1)  
        # debug.print("vector_Q for each unit = {}", vector_Q)
        # print(';n_users in the estimating function', n_users)
        ##### estimation function for pi (refer Eq.8 in Algorithm_SAC.tex)
        p0 = jax.nn.sigmoid(steepness * jnp.dot(state, beta_pi)).reshape(-1, 1) # (1,)
        p = policy(state, beta_pi).reshape(-1, 1)  # (1, 1) 
        temp = jnp.dot(state, betaQ_target[int(dim/2):]).reshape(-1, 1) - lambda_entropy  * jnp.log(p/(1-p)) 
        # temp = jnp.dot(state, betaQ_target[int(dim/2):]).reshape(-1, 1) - lambda_entropy / n_users * jnp.log(p/(1-p)) 
        vector_pi = (1-2*lower_clip) * steepness * p0 * (1 - p0) * temp.reshape(-1, 1) * state # (1, 1) * (1, 2) -> (1, 2)
        vector_pi = vector_pi.reshape(-1, 1)  # (1, 2) -> (2, 1)
        # vector_pi = jnp.mean(vector_pi, axis=0).reshape(-1, 1)  # (1, 2) -> (1, 2) -> (2, 1) 
        # debug.print("vector_pi for each unit = {}", vector_pi)
        return jnp.concatenate([vector_Q.flatten(), vector_pi.flatten()]) # (4,) + (2,) = (6,)

    z = jnp.asarray(Z_id) # (1,1)
    pred = jnp.all(z == 0)
    return lax.cond(pred, zero_branch, active_branch, operand=None) # (6,)
