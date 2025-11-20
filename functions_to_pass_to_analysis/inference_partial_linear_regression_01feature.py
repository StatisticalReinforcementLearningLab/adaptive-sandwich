import jax.numpy as jnp



# inference function: loss function to propogate the gradient over theta_est
def inference_partial_linear_regression_01feature(
    theta_est, # [p,]
    intercept,
    # past_reward, # [nT, 1]
    # action, # [nT, 1]
    reward, # [nT, 1]
    user_id, # [nT, 1]
    calendar_t,
    pretreat_feature1, # [nT, 1]
    pretreat_feature2, # [nT, 1]
    Z_id, # [nT, 1]
):

    ######## one-phase estimation
    n = jnp.unique(user_id).size
    T = jnp.unique(calendar_t).size
    C1 = pretreat_feature1.reshape(n, T)[:,0:1] # [n, 1]
    C2 = pretreat_feature2.reshape(n, T)[:,0:1] # [n, 2]
    intercept = intercept.reshape(n, T)[:,0:1]
    Z = Z_id.reshape(n, T)[:,0:1] # [n, 1]
    ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    return 0.5 * jnp.sum((ave_reward - theta_est[0] - theta_est[1] * Z - theta_est[2] * C1 - theta_est[3] * C2) ** 2) / n # one-dimentinal linear regression without intercept


    ######## two-phase estimation
    # n = jnp.unique(user_id).size
    # T = jnp.unique(calendar_t).size
    # C_design = jnp.hstack((intercept.reshape(n, T)[:,0:1], pretreat_feature1.reshape(n, T)[:,0:1], pretreat_feature2.reshape(n, T)[:,0:1])) # [n, 3]
    # Z = Z_id.reshape(n, T)[:,0:1] # [n, 1]
    # ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    # P = C_design @ jnp.linalg.pinv(C_design)
    
    # resi = ave_reward - P @ ave_reward
    # resi_ = Z - P @ Z
    # return 0.5 * jnp.sum((resi - theta_est[0] * resi_) ** 2) / n # one-dimentinal linear regression without intercept


