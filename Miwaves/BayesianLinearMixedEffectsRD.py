import jax.numpy as jnp
import jax
import numpy as np
import os
import scipy.stats as stats
import scipy.linalg as linalg
import pickle as pkl
import logging
import copy

from functools import partial
from typing import Callable
from RLAlgorithm import RLAlgorithm
from priors import PARAM_SIZE
#from priors_RD import PARAM_SIZE


def invert_sigma_theta(Sigma_0: jnp.array, Sigma_u: jnp.array, nusers: int):
    """
    Invert the covariance matrix efficiently
    """
    C = jnp.linalg.inv(Sigma_u)
    D = -jnp.linalg.inv(Sigma_u + nusers * Sigma_0) @ Sigma_0 @ C

    inverse = jnp.kron(jnp.ones((nusers, nusers)), D) + jnp.kron(
        jnp.identity(nusers), C
    )

    return inverse


# In [12]:
def check_matrix(
    flat_lower_t: jnp.array,
    noise_precision: float,
    design_matrix: jnp.array,
    reward_matrix: jnp.array,
    mu_prior: jnp.array,
    sigma_prior: jnp.array,
    size: int,
    nusers: int,
    ts: int,
    debug: bool = False,
):
    """
    Objective function for optimization, but also checks
    if the resulting posterior is going to be PSD and within
    reasonable limits
    """

    # Construct the lower triangular matrix
    L = jnp.zeros((size, size), dtype=float)
    L = L.at[jnp.tril_indices(size)].set(flat_lower_t)

    # Construct the PSD matrix of random effects variance
    Sigma_u = L @ L.T
    del L

    # Construct X and y
    # X = jnp.linalg.inv(jnp.array(sigma_prior) + jnp.kron(jnp.identity(nusers), Sigma_u))
    # X = invert_sigma_theta(sigma_prior, Sigma_u, nusers)
    X = jnp.linalg.inv(jnp.kron(jnp.ones((nusers, nusers)), sigma_prior) + jnp.kron(jnp.identity(nusers), Sigma_u))
    A = jax.scipy.linalg.block_diag(
        *[
            design_matrix[ts * (i) : ts * (i + 1)].T
            @ design_matrix[ts * (i) : ts * (i + 1)]
            for i in range(nusers)
        ]
    )
    y = noise_precision

    eigvals = jnp.min(jnp.linalg.eigh(X + y * A)[0])
    
    # import pdb; pdb.set_trace()

    if eigvals <= 0 or jnp.isnan(eigvals):
        return 100000, False

    B = jnp.array(
        [
            design_matrix[ts * (i) : ts * (i + 1)].T
            @ reward_matrix[ts * (i) : ts * (i + 1)]
            for i in range(nusers)
        ]
    ).flatten()

    newpost_var = jnp.linalg.inv(X + y * A)

    eigvals = jnp.min(jnp.linalg.eigh(newpost_var)[0])

    if eigvals <= 0 or jnp.isnan(eigvals):
        return 100000, False

    temp_mean = X @ mu_prior + y * B
    newpost_mean = newpost_var @ temp_mean

    # Do the sanity checks

    # If any diagonal entry in the covariance matrix is less than 0
    # if jnp.any(jnp.diag(newpost_var) < 0):
    #     return 100000, False
    if jnp.min(jnp.diag(newpost_var)) < 0:
        return 100000, False

    # If absolute value of any posterior mean entry is greater than 10 (need to think
    # about this sanity check)
    # if jnp.any(jnp.abs(newpost_mean) > 10):
    #     return 100000, False
    if jnp.max(jnp.abs(newpost_mean).flatten()) > 10:
        return 100000, False

    # Evaluate the optimization function
    s1, part1 = jnp.linalg.slogdet(X)
    s2, part2 = jnp.linalg.slogdet(X + y * A)
    part2 = (-1) * part2
    part3 = nusers * ts * jnp.log(y)
    part4 = -1 * y * jnp.sum(jnp.square(reward_matrix))
    part5 = -1 * mu_prior.T @ X @ mu_prior
    # part6 = (
    #     (X @ mu_prior + y * B).T @ jnp.linalg.inv(X + y * A) @ (X @ mu_prior + y * B)
    # )
    part6 = temp_mean.T @ newpost_mean

    # Doing negative because we are minimizing
    result = -1 * (part1 + part2 + part3 + part4 + part5 + part6)
    del part1
    del part2
    del part3
    del part4
    del part5
    del part6
    return result, True


# Define the objective function
@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def obj_func(
    flat_lower_t: jnp.array,
    noise_precision: float,
    design_matrix: jnp.array,
    reward_matrix: jnp.array,
    mu_prior: jnp.array,
    sigma_prior: jnp.array,
    size: int,
    nusers: int,
    ts: int,
    debug: bool = False,
):
    """Objective function for optimization"""

    # Construct the lower triangular matrix
    L = jnp.zeros((size, size), dtype=float)
    L = L.at[jnp.tril_indices(size)].set(flat_lower_t)

    # Construct the PSD matrix of random effects variance
    Sigma_u = L @ L.T
    del L

    # Construct X and y
    # X = jnp.linalg.inv(jnp.array(sigma_prior) + jnp.kron(jnp.identity(nusers), Sigma_u))
    X = invert_sigma_theta(sigma_prior, Sigma_u, nusers)

    A = jax.scipy.linalg.block_diag(
        *[
            design_matrix[ts * (i) : ts * (i + 1)].T
            @ design_matrix[ts * (i) : ts * (i + 1)]
            for i in range(nusers)
        ]
    )

    B = jnp.array(
        [
            design_matrix[ts * (i) : ts * (i + 1)].T
            @ reward_matrix[ts * (i) : ts * (i + 1)]
            for i in range(nusers)
        ]
    ).flatten()

    y = noise_precision

    # Evaluate the optimization function
    s1, part1 = jnp.linalg.slogdet(X)
    s2, part2 = jnp.linalg.slogdet(X + y * A)
    part2 = (-1) * part2
    part3 = nusers * ts * jnp.log(y)
    part4 = -1 * y * jnp.sum(jnp.square(reward_matrix))
    part5 = -1 * mu_prior.T @ X @ mu_prior
    part6 = (
        (X @ mu_prior + y * B).T @ jnp.linalg.inv(X + y * A) @ (X @ mu_prior + y * B)
    )

    if debug:
        print("Part 1: ", -part1)
        print("Part 2: ", -part2)
        print("Part 3: ", -part3)
        print("Part 4: ", -part4)
        print("Part 5: ", -part5)
        print("Part 6: ", -part6)

    # Doing negative because we are minimizing
    result = -1 * (part1 + part2 + part3 + part4 + part5 + part6)

    del part1
    del part2
    del part3
    del part4
    del part5
    del part6

    return result


class BayesianLinearMixedEffectsRegressionRD(RLAlgorithm):
    """Bayesian linear regression RL algorithm with reward design"""

    def __init__(
        self,
        variant: int,
        prior_mean: np.array,
        prior_cov: np.array,
        init_cov_u: np.array,
        init_noise_var: float,
        num_users: int,
        alloc_func: Callable,
        max_iter: int = 500,
        learning_rate: float = 0.001,
        tolerance: float = 1e-6,
        starting_time_of_day: int = 0,
        rng: np.random.RandomState = None,
        s1_lookback: int = 6,
        act_cost_threshold: float = 0.0,
        debug: bool = False,
        logger_path: str = None,
    ):
        """Initialize the algorithm"""
        self.variant = variant
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.init_noise_var = init_noise_var
        self.noise_var = init_noise_var
        self.sigma_u = init_cov_u
        cholesky = np.linalg.cholesky(self.sigma_u)
        self.ltu_flat = cholesky[np.tril_indices(self.sigma_u.shape[0])].flatten()
        self.init_ltu_flat = copy.deepcopy(self.ltu_flat)
        self.allocation_function = alloc_func
        self.current_decision_point = 0
        self.time_of_day = starting_time_of_day
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.rng = rng
        self.bernoulli = stats.bernoulli
        self.user_data = {}
        self.num_users = num_users
        for i in range(self.num_users):
            self.user_data[i] = {
                "features": [{}],
                "state": [[0, self.time_of_day, 1]],  # default initial starting state
                "design_state": [],
                "action": [],
                "act_prob": [],
                "reward": [],
                "real_reward": [],
            }
        self.mu_0 = np.kron(np.ones(self.num_users), self.prior_mean)
        self.Sigma_0 = np.kron(
            np.ones((self.num_users, self.num_users)), self.prior_cov
        )
        self.Sigma_theta_t = self.Sigma_0 + np.kron(
            np.identity(self.num_users), self.sigma_u
        )
        self.posterior_mean = self.mu_0.copy()
        self.posterior_cov = self.Sigma_theta_t.copy()
        self.posterior_mean_history = []
        self.posterior_cov_history = []
        self.sigma_u_history = []
        self.noise_var_history = []
        self.debug = debug

        self.s1_lookback = s1_lookback
        self.act_cost_threshold = act_cost_threshold
        df = float(s1_lookback - 1) / s1_lookback
        nf = (1 - df) / (1 - df ** s1_lookback)
        self.S1_weights = np.array([nf * (df ** i) for i in range(s1_lookback)])

        # Logging stuff
        logfile = logger_path + "/log.txt"
        self.logger = logging.getLogger("BayesMixedEffectsRD")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def create_state(self, user: int):
        """Create the state vector for the given features and decision point index"""
        avg_reward = np.mean(self.user_data[user]["real_reward"][-3:])

        if avg_reward >= 2:
            S1 = 1
        else:
            S1 = 0

        # recent rewards
        #recent_rewards = np.flip(self.user_data[user]["real_reward"][-self.s1_lookback:])

        # create S1
        #S1 = np.dot(recent_rewards, self.S1_weights[: len(recent_rewards)]) / 3.0

        # create S2 based on time of day
        S2 = self.time_of_day

        # create S3 based on last cannabis use
        # if user did not use cannabis in the past decision point, S3 = 1
        if self.user_data[user]["features"][-1]["cannabis_use"] == 0:
            S3 = 1
        # if user used cannabis in the past decision point, S3 = 0
        # if the user did not report cannabis use in the past decision point, S3 = 0
        else:
            S3 = 0
        return [S1, S2, S3]

    def compute_reward(self, user: int):
        """Create the reward given the features for a given time index"""
        reward = 0
        features = self.user_data[user]["features"][-1]
        if features["survey_completion"] == 1:
            if features["activity_question"] == 1:
                reward = 3
            else:
                reward = 2
        elif features["app_usage"] == 1:
            reward = 1
        return reward

    def action_cost(self, user: int):
        """Return the action cost for a given user"""
        
        # Check the last action
        last_action = self.user_data[user]["action"][-1]

        # If the last action was 1, return the cost
        if last_action == 1:
            all_rewards = self.user_data[user]["real_reward"]
            std_all_rewards = np.std(all_rewards, ddof=1)

            if np.isnan(std_all_rewards):
                std_all_rewards = 1.12

            return self.act_cost_threshold * std_all_rewards

        # If the last action was 0, return 0
        else:
            return 0


    def update_design_row(self, user: int):
        """Update the design row for a given user"""

        # Get the state, action, and action probability
        state = self.user_data[user]["state"][-1]
        action = float(self.user_data[user]["action"][-1])
        act_prob = float(self.user_data[user]["act_prob"][-1])

        # Get the individual state elements
        s1 = state[0]
        s2 = state[1]
        s3 = state[2]

        # Create the baseline and advantage
        baseline = [1, s1, s2, s3, s1 * s2, s1 * s3, s2 * s3, s1 * s2 * s3]
        act_advantage = [(act_prob * i) for i in baseline]
        a_pi_advantage = [((action - act_prob) * i) for i in baseline]

        # Create the design row
        design_row = [
            *baseline[: PARAM_SIZE[self.variant][0]],
            *act_advantage[: PARAM_SIZE[self.variant][1]],
            *a_pi_advantage[: PARAM_SIZE[self.variant][2]],
        ]
        if self.debug:
            print(design_row)
        return design_row

    def clip_prob(self, prob, min: float = 0.2, max: float = 0.8):
        """Clip the probability to be between min and max"""
        return np.clip(prob, min, max)

    def get_action(self, user: int, decision_idx: int):
        """
        Get the action for a particular user for a particular decision point
        """
        # Get the state
        state = self.user_data[user]["state"][-1]

        # Compute the advantage
        advantage_default = [
            1,
            state[0],
            state[1],
            state[2],
            state[0] * state[1],
            state[0] * state[2],
            state[1] * state[2],
            state[0] * state[1] * state[2],
        ]
        advantage_with_intercept = np.array(
            advantage_default[: PARAM_SIZE[self.variant][2]]
        )
        num_params = np.sum(PARAM_SIZE[self.variant])
        posterior_mean_user = self.posterior_mean[
            user * num_params : (user + 1) * num_params
        ]
        posterior_cov_user = self.posterior_cov[
            user * num_params : (user + 1) * num_params,
            user * num_params : (user + 1) * num_params,
        ]

        # Compute the posterior mean of the adv term
        beta_mean = np.array(posterior_mean_user[-PARAM_SIZE[self.variant][2] :])

        # Compute the posterior covariance of the adv term
        beta_cov = np.array(
            posterior_cov_user[
                -PARAM_SIZE[self.variant][2] :, -PARAM_SIZE[self.variant][2] :
            ]
        )

        # Compute the posterior mean of the adv*beta distribution
        adv_beta_mean = advantage_with_intercept.T.dot(beta_mean)

        # Compute the posterior variance of the adv*beta distribution
        adv_beta_var = advantage_with_intercept.T @ beta_cov @ advantage_with_intercept

        # Call the allocation function
        prob = self.allocation_function(mean=adv_beta_mean, var=adv_beta_var)
        if self.debug:
            self.logger.info(f"[{decision_idx}] {user} {prob}")
            print(user, decision_idx, prob)
        if np.isnan(prob):
            self.logger.error(
                f"[{self.current_decision_point}] Probability NaN encountered for user: {user}"
            )
            self.logger.error(f"BETA_MEAN: {beta_mean} VAR: {beta_cov}")
            self.logger.error(f"ADV_BETA: {adv_beta_mean} VAR: {adv_beta_var}")
            self.logger.error(f"POST_MEAN: {posterior_mean_user}")
            self.logger.error(f"POST_VAR: {posterior_cov_user}")
            prob = 0.5

        # Clip the probability
        act_prob = self.clip_prob(prob)

        # Sample the action
        action = self.bernoulli.rvs(act_prob, random_state=self.rng)

        # Update the user data
        self.user_data[user]["action"].append(action)
        self.user_data[user]["act_prob"].append(act_prob)

        # Update the design state
        self.user_data[user]["design_state"].append(self.update_design_row(user))
        return action

    def update_data(self, user: int, decision_idx: int, data: dict):
        """
        Update the data for a particular user for a particular decision point
        """
        self.user_data[user]["features"].append(data)

    def end_decision_point(self, decision_idx: int):
        """
        End the decision point. Flip the time of day. This computes the reward
        for each user for the current decision point, increases the current
        decision point by 1, and computes the new state for each user for
        the new decision point
        """
        self.current_decision_point += 1
        self.time_of_day = 1 - self.time_of_day
        for i in range(self.num_users):
            self.user_data[i]["real_reward"].append(self.compute_reward(i))
            self.user_data[i]["reward"].append(self.compute_reward(i) - self.action_cost(i))
            self.user_data[i]["state"].append(self.create_state(i))

    def create_design_and_reward_matrix(self):
        """
        Create the design matrix and reward matrix up until the current decision point
        using design rows and reward history for each user
        """

        # Design matrix made by stacking design state of each user
        design_matrix = np.vstack(
            [self.user_data[i]["design_state"] for i in range(self.num_users)]
        )

        # Reward matrix made by stacking reward history of each user
        reward_matrix = np.hstack(
            [self.user_data[i]["reward"] for i in range(self.num_users)]
        )
        return design_matrix, reward_matrix

    def update_hyperparameters(self):
        """
        Update the noise variance and random effects variance using data up until
        the current decision point
        """

        # Update noise variance and random effects variance
        self.optimize_variance_parameters(debug=True)
        self.noise_var_history.append(self.noise_var)
        self.sigma_u_history.append(self.sigma_u)

    def update_posteriors(self):
        """
        Update the posteriors using data up until the current decision point
        """

        # Create design matrix and reward matrix
        design_matrix, reward_matrix = self.create_design_and_reward_matrix()
        sigma_theta_t_inverse = np.linalg.inv(self.Sigma_theta_t)
        ts = self.current_decision_point
        A = linalg.block_diag(
            *[
                design_matrix[ts * (i) : ts * (i + 1)].T
                @ design_matrix[ts * (i) : ts * (i + 1)]
                for i in range(self.num_users)
            ]
        )
        B = np.array(
            [
                design_matrix[ts * (i) : ts * (i + 1)].T
                @ reward_matrix[ts * (i) : ts * (i + 1)]
                for i in range(self.num_users)
            ]
        ).flatten()

        # Compute posterior mean and posterior covariance
        self.posterior_cov = np.linalg.inv(
            sigma_theta_t_inverse + (1 / self.noise_var) * A
        )
        self.posterior_mean = self.posterior_cov @ (
            sigma_theta_t_inverse @ self.mu_0.reshape(-1, 1)
            + (1 / self.noise_var * B).reshape(-1, 1)
        )

        # import pdb; pdb.set_trace()

        # Update the history of posterior mean and posterior covariance
        self.posterior_mean_history.append(self.posterior_mean)
        self.posterior_cov_history.append(self.posterior_cov)

    def dump_model_and_data(self, path: str):
        """
        Dump the data to a pkl file
        """
        user_data_path = os.path.join(path, "user_data.pkl")
        posterior_mean_path = os.path.join(path, "posterior_mean.pkl")
        posterior_cov_path = os.path.join(path, "posterior_cov.pkl")
        noise_var_path = os.path.join(path, "noise_var.pkl")
        sigma_u_path = os.path.join(path, "sigma_u.pkl")
        with open(user_data_path, "wb") as f:
            pkl.dump(self.user_data, f)
        #with open(posterior_mean_path, "wb") as f:
        #    pkl.dump(self.posterior_mean_history, f)
        # with open(posterior_cov_path, "wb") as f:
        # pkl.dump(self.posterior_cov_history, f)
        #with open(noise_var_path, "wb") as f:
        #    pkl.dump(self.noise_var_history, f)
        # with open(sigma_u_path, "wb") as f:
        # pkl.dump(self.sigma_u_history, f)

    def optimize_variance_parameters(self, debug: bool = False):
        """
        Optimize the noise variance and random effects variance using data up until
        the current decision point using empirical Bayes
        """
        if debug:
            print(self.current_decision_point)

        # Create design matrix and reward matrix
        design_matrix, reward_matrix = self.create_design_and_reward_matrix()
        init_ltu_flat = self.ltu_flat
        init_noise_var = 1.0 / self.noise_var
        min_flat = self.ltu_flat
        min_noise_var = init_noise_var
        skip = 0
        last_update = -1
        reset_flag = False

        # init_value = self.noise_var
        # init_value = 20.0
        lr = lr2 = self.learning_rate
        old_obj, valid = check_matrix(
            init_ltu_flat,
            init_noise_var,
            design_matrix,
            reward_matrix,
            self.mu_0,
            self.prior_cov,
            self.sigma_u.shape[0],
            self.num_users,
            self.current_decision_point,
        )
        print("Init Start obj: {}, valid: {} ".format(old_obj, valid))
        
        # import pdb; pdb.set_trace()

        if not valid:
            init_ltu_flat = self.init_ltu_flat
            init_noise_var = self.init_noise_var
            old_obj, _ = check_matrix(
                init_ltu_flat,
                init_noise_var,
                design_matrix,
                reward_matrix,
                self.mu_0,
                self.prior_cov,
                self.sigma_u.shape[0],
                self.num_users,
                self.current_decision_point,
            )

        min_obj = copy.deepcopy(old_obj)
        print("Start obj: {}".format(old_obj))

        # Do the optimization
        for idx in range(self.max_iter):
            # Compute the gradient of the objective function
            jacob = jax.grad(obj_func, argnums=0)(
                init_ltu_flat,
                init_noise_var,
                design_matrix,
                reward_matrix,
                self.mu_0,
                self.prior_cov,
                self.sigma_u.shape[0],
                self.num_users,
                self.current_decision_point,
            )
            grad = jax.grad(obj_func, argnums=1)(
                init_ltu_flat,
                init_noise_var,
                design_matrix,
                reward_matrix,
                self.mu_0,
                self.prior_cov,
                self.sigma_u.shape[0],
                self.num_users,
                self.current_decision_point,
            )
            new_ltu_flat = init_ltu_flat - (lr * jacob)

            # Update the value of noise variance
            if init_noise_var - lr2 * grad > 0.0001:
                new_noise_var = init_noise_var - (lr2 * grad)
            else:
                new_noise_var = init_noise_var
                lr2 = lr2 / 2
            # except Exception as e:
            #     import pdb; pdb.set_trace()
            obj_val, valid = check_matrix(
                new_ltu_flat,
                new_noise_var,
                design_matrix,
                reward_matrix,
                self.mu_0,
                self.prior_cov,
                self.sigma_u.shape[0],
                self.num_users,
                self.current_decision_point,
            )

            # Reduce the learning rate if objective is either null (i.e. invalid Sigma_u
            # or objective value explodes, or objective value goes negative, or the resulting
            # posteriors will be invalid i.e. either mean is too big, or variance is negative
            # for the diagonal entries)
            if jnp.isnan(obj_val) or obj_val > 10 * min_obj or obj_val < 0 or not valid:
                lr = lr / 2
                skip += 1
            else:
                if obj_val < min_obj:
                    min_flat = new_ltu_flat
                    min_noise_var = new_noise_var
                    min_obj = obj_val
                    last_update = idx
                init_ltu_flat = new_ltu_flat
                init_noise_var = new_noise_var
                skip = 0
            if debug:
                print(
                    "Iteration: {}, Noise variance: {}, Objective value: {}, Valid: {}, Nan: {}".format(
                        idx, 1.0 / init_noise_var, obj_val, valid, jnp.isnan(obj_val)
                    )
                )

            # Check if the change in objective value is small
            # if np.abs(obj_val - old_obj) < self.tolerance or skip == 10 or (idx == self.max_iter - 1) or (idx - last_update) > 150:
            if np.abs(obj_val - old_obj) < self.tolerance or (idx == self.max_iter - 1):
                if debug:
                    print(
                        "Converged at iteration: {} with value {}".format(
                            idx, 1.0 / min_noise_var
                        )
                    )
                    print("Sigma_U:", min_flat)
                break

            # Restart if we haven't gone below the previous objective value
            if (idx - last_update) > 250 or skip == 10:
                if reset_flag:
                    # Reset already done once, so just terminate
                    if debug:
                        print(
                            "Converged at iteration: {} with value {}".format(
                                idx, 1.0 / min_noise_var
                            )
                        )
                        print("Sigma_U:", min_flat)
                    break
                else:
                    # Reset back to initial params
                    init_ltu_flat = self.init_ltu_flat
                    init_noise_var = self.init_noise_var
                    lr = lr2 = self.learning_rate
                    last_update = idx
                    skip = 0
                    reset_flag = True
                    if debug:
                        print("Restarted at idx: ", idx)
            elif skip == 0:
                old_obj = obj_val

        # Update the noise variance
        self.noise_var = 1.0 / min_noise_var
        self.noise_var_history.append(self.noise_var)

        # Update the sigma_u
        self.ltu_flat = min_flat
        L = np.zeros(self.sigma_u.shape, dtype=float)
        L[np.tril_indices(self.sigma_u.shape[0])] = self.ltu_flat
        self.sigma_u = L @ L.T
        self.Sigma_theta_t = self.Sigma_0 + np.kron(
            np.identity(self.num_users), self.sigma_u
        )
        self.sigma_u_history.append(np.array(self.sigma_u))
