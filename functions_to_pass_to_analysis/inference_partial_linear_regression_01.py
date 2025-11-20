import jax.numpy as jnp



# inference function: loss function to propogate the gradient over theta_est
def inference_partial_linear_regression_01(
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

    ######## ignore the pretreat_features for simplicity
    n = jnp.unique(user_id).size
    T = jnp.unique(calendar_t).size
    Z = Z_id.reshape(n, T)[:,0:1] # [n, 1]
    ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    # theta_0 = ave_reward.mean()
    return 0.5 * jnp.sum((ave_reward - theta_est[0] - theta_est[1] * Z) ** 2) / n 





