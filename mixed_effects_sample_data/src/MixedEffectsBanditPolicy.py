import numpy as np
from scipy.stats import bernoulli
from policy import Policy
import pandas as pd
import pickle as pkl


class MixedEffectsBanditPolicy(Policy):
    def __init__(
        self,
        m: int,
        T: int,
        mu_beta: np.ndarray,
        Sigma_beta: np.ndarray,
        Sigma_gamma: np.ndarray,
        sigma_e2: float,
        rho_func: callable,
        Xi_beta: float,
        Xi_gamma: float,
        xi_beta: float,
        xi_gamma: float,
        output_path: str,
        seed: int = None,
    ):
        """
        Initialize the Mixed Effects Bandit Policy.

        Parameters:
            m (int): Number of users.
            T (int): Number of decision points.
            mu_beta (np.ndarray): Prior mean vector for the fixed effects.
            Sigma_beta (np.ndarray): Prior covariance matrix for the fixed effects.
            Sigma_gamma (np.ndarray): Prior covariance matrix for the random effects.
            sigma_e2 (float): Noise variance.
            rho_func (function): Smoothing function for posterior sampling.
            Xi_beta (float): Preset scalar for Z_beta mapping.
            Xi_gamma (float): Preset scalar for Z_gamma mapping.
            xi_beta (float): Preset scalar for Z_beta action mapping.
            xi_gamma (float): Preset scalar for Z_gamma action mapping.
            output_path (str): Path to store the results.
            seed (int): Seed for random number generator.
        """
        self.m = m
        self.T = T
        self.mu_beta = (
            np.array(mu_beta) if not np.isscalar(mu_beta) else np.array([mu_beta])
        )
        self.Sigma_beta = (
            np.array(Sigma_beta)
            if not np.isscalar(Sigma_beta)
            else np.array([[Sigma_beta]])
        )
        self.Sigma_gamma = (
            np.array(Sigma_gamma)
            if not np.isscalar(Sigma_gamma)
            else np.array([[Sigma_gamma]])
        )
        self.sigma_e2 = sigma_e2
        self.rho_func = rho_func
        self.Xi_beta = Xi_beta
        self.Xi_gamma = Xi_gamma
        self.xi_beta = xi_beta
        self.xi_gamma = xi_gamma

        # Initialize a random generator with the provided seed
        self.rng = np.random.default_rng(seed)

        # Initial joint posterior
        self.mu_beta_gamma = np.concatenate([self.mu_beta, np.zeros((m, 1))]).reshape(
            -1, 1
        )
        self.Sigma_beta_gamma = np.block(
            [
                [
                    self.Sigma_beta,
                    np.zeros(
                        (self.Sigma_beta.shape[0], self.Sigma_gamma.shape[0] * self.m)
                    ),
                ],
                [
                    np.zeros(
                        (self.Sigma_gamma.shape[0] * self.m, self.Sigma_beta.shape[1])
                    ),
                    np.kron(np.eye(self.m), self.Sigma_gamma),
                ],
            ]
        )

        # Initialize the matrices to record states and statistics
        self.states = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.actions = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.action_prob = [[-1 for _ in range(self.m)] for _ in range(self.T)]
        self.outcomes = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.A_beta = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.A_gamma = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.A_beta_gamma = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.B_beta = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.B_gamma = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.Z_beta = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.Z_gamma = [[[] for _ in range(self.m)] for _ in range(self.T)]
        self.E = [[] for _ in range(self.T)]
        self.lambdat = [[] for _ in range(self.T)]

        # Tracker for last time
        self.last_time = -1

        # Dictionaries and pandas dataframes to store results for after study analysis

        # study_df columns:
        # calendar_time: int dtype
        # user_id: int dtype
        # in_study_indicator: int dtype (only 0 and 1, but 1 in this experiment)
        # action: int dtype
        # policy_number: int dtype
        # action_probability: float dtype
        # outcome: float dtype
        # state: float dtype

        self.study_df = pd.DataFrame(
            columns=[
                "calendar_time",
                "user_id",
                "in_study_indicator",
                "action",
                "policy_number",
                "action_probability",
                "outcome",
                "state",
            ]
        )

        # dictionary for estimating equation function at each update time
        # keyed by policy number followed by the user id
        self.estimating_equation_function_dict = {}
        self.estimating_equation_function_df = None

        # dictionary for action selection function at each update time
        # keyed by policy number followed by the user id
        self.action_selection_function_dict = {}
        self.action_selection_function_df = None

        # Output path to store the results
        self.output_path = output_path

    def generate_design_matrices(self, state: float, time_step: int, action: int):
        """
        Generate the design matrices Z_beta and Z_gamma based on state and action.

        Z_beta = Xi_beta * state + xi_beta * action
        Z_gamma = Xi_gamma * state + xi_gamma * action
        """
        Z_beta = self.Xi_beta[time_step] * state + self.xi_beta[time_step] * action
        Z_gamma = self.Xi_gamma[time_step] * state + self.xi_gamma[time_step] * action

        Z_beta = np.array(Z_beta).reshape(self.mu_beta.shape[0], 1)
        Z_gamma = np.array(Z_gamma).reshape(self.Sigma_gamma.shape[0], 1)

        return Z_beta, Z_gamma

    def update_AB_matrices(self, user_id: int, time_step: int):
        """
        Update the A and B matrices needed for posterior updates.

        Parameters:
            user_id (int): ID of the user.
            time_step (int): The current time step t.
        """
        # Get the Z matrices for the user at the given time step
        Z_beta, Z_gamma = (
            self.Z_beta[time_step][user_id],
            self.Z_gamma[time_step][user_id],
        )

        # Change to numpy array
        Z_beta = np.array(Z_beta)
        Z_gamma = np.array(Z_gamma)

        # Compute Z_beta^2, Z_gamma^2, and Z_beta * Z_gamma
        Z_beta2 = Z_beta @ Z_beta.T
        Z_gamma2 = Z_gamma @ Z_gamma.T
        Z_beta_gamma = Z_beta @ Z_gamma.T

        Z_beta2 = np.array(Z_beta2).reshape(
            self.mu_beta.shape[0], self.mu_beta.shape[0]
        )
        Z_gamma2 = np.array(Z_gamma2).reshape(
            self.Sigma_gamma.shape[0], self.Sigma_gamma.shape[0]
        )
        Z_beta_gamma = np.array(Z_beta_gamma).reshape(
            self.mu_beta.shape[0], self.Sigma_gamma.shape[0]
        )

        # Update the A and B matrices based on the time step
        if time_step == 0:
            self.A_beta[time_step][user_id] = Z_beta2
            self.A_gamma[time_step][user_id] = Z_gamma2
            self.A_beta_gamma[time_step][user_id] = Z_beta_gamma
            self.B_beta[time_step][user_id] = (
                Z_beta * self.outcomes[time_step][user_id]
            ).reshape(-1, 1)
            self.B_gamma[time_step][user_id] = (
                Z_gamma * self.outcomes[time_step][user_id]
            ).reshape(-1, 1)
        else:
            self.A_beta[time_step][user_id] = (
                self.A_beta[time_step - 1][user_id] + Z_beta2
            )
            self.A_gamma[time_step][user_id] = (
                self.A_gamma[time_step - 1][user_id] + Z_gamma2
            )
            self.A_beta_gamma[time_step][user_id] = (
                self.A_beta_gamma[time_step - 1][user_id] + Z_beta_gamma
            )
            self.B_beta[time_step][user_id] = self.B_beta[time_step - 1][user_id] + (
                Z_beta * self.outcomes[time_step][user_id]
            ).reshape(-1, 1)
            self.B_gamma[time_step][user_id] = self.B_gamma[time_step - 1][user_id] + (
                Z_gamma * self.outcomes[time_step][user_id]
            ).reshape(-1, 1)

    def update_posterior(self):
        """
        Updates the posterior distribution.
        """

        # Initialize the matrices
        A_beta_sum = np.zeros((self.Sigma_beta.shape[0], self.Sigma_beta.shape[0]))
        B_beta_sum = np.zeros((self.Sigma_beta.shape[0], 1))

        # Compute the matrices
        for user_id in range(self.m):
            A_beta_sum += self.A_beta[self.last_time][user_id]
            B_beta_sum += self.B_beta[self.last_time][user_id]

        B_gamma = np.array(self.B_gamma[self.last_time]).reshape(-1, 1)
        A_beta_gamma = np.array(self.A_beta_gamma[self.last_time]).reshape(
            -1, self.m * self.Sigma_gamma.shape[0]
        )
        Sigma_gamma_inv = np.linalg.inv(self.Sigma_gamma)

        D_t = np.block(
            [
                [
                    (
                        Sigma_gamma_inv
                        + (1 / self.sigma_e2) * self.A_gamma[self.last_time][i]
                        if i == j
                        else np.zeros_like(Sigma_gamma_inv)
                    )
                    for j in range(self.m)
                ]
                for i in range(self.m)
            ]
        )

        # Compute I_t as per equation (3.8)
        I_t = np.block(
            [
                [
                    np.linalg.inv(self.Sigma_beta) + (1 / self.sigma_e2) * A_beta_sum,
                    (1 / self.sigma_e2) * A_beta_gamma,
                ],
                [(1 / self.sigma_e2) * A_beta_gamma.T, D_t],
            ]
        )

        # Compute J_t as per equation (3.9)
        J_t = np.block(
            [
                [
                    np.linalg.inv(self.Sigma_beta) @ self.mu_beta
                    + (1 / self.sigma_e2) * B_beta_sum
                ],
                [(1 / self.sigma_e2) * B_gamma],
            ]
        )

        # Update the posterior mean and covariance
        I_t_inv = np.linalg.inv(I_t)
        self.mu_beta_gamma = I_t_inv @ J_t
        self.Sigma_beta_gamma = I_t_inv

    def update_policy(self):
        self.last_time += 1
        self.update_posterior()
        self.calculate_statistics()

        # Save the results for after study analysis
        if self.last_time == self.T - 1:
            self.save_results()

    def record(
        self, user_id: int, time_step: int, state: float, action: int, outcome: float
    ):
        """
        Record the outcome and update matrices using the observed outcome.
        """

        # Record the state, action, and outcome
        self.states[time_step][user_id].append(state)
        self.actions[time_step][user_id].append(action)
        self.outcomes[time_step][user_id].append(outcome)

        # Add a row to the study dataframe
        self.study_df = pd.concat(
            [
                self.study_df,
                pd.DataFrame(
                    {
                        "calendar_time": [time_step + 1],
                        "user_id": [user_id],
                        "in_study_indicator": [1],
                        "action": [action],
                        "policy_number": [self.last_time + 1],
                        "action_probability": [self.action_prob[time_step][user_id]],
                        "outcome": [outcome],
                        "state": [state],
                    }
                ),
            ],
            ignore_index=True,
        )

        # Compute the design matrices
        Z_beta, Z_gamma = self.generate_design_matrices(state, time_step, action)
        self.Z_beta[time_step][user_id].append(Z_beta)
        self.Z_gamma[time_step][user_id].append(Z_gamma)

        # Compute the A and B matrices
        self.update_AB_matrices(user_id, time_step)

    def choose_action(self, user_id: int, time_step: int, state: float):
        """
        Select an action for the given user based on the mixed-effects bandit policy.

        Parameters:
            user_id (int): ID of the user.
            time_step (int): The current time step t.
            state (float or np.ndarray): The current state S_{i,t} for user i at time t.

        Returns:
            action (int): The chosen action (0 or 1).
        """
        # Generate design matrices for action = 1 and action = 0
        Z_beta_1, Z_gamma_1 = self.generate_design_matrices(state, time_step, action=1)
        Z_beta_0, Z_gamma_0 = self.generate_design_matrices(state, time_step, action=0)

        # Advantage function
        fs = np.vstack([Z_beta_1 - Z_beta_0, Z_gamma_1 - Z_gamma_0])

        # Advantage function mean
        # It is basically beta (first part of mu_beta_gamma) + gamma_i (second part of mu_beta_gamma
        # for the user i)
        user_mean = np.vstack(
            [
                self.mu_beta_gamma[0 : Z_beta_0.shape[0]],
                self.mu_beta_gamma[
                    Z_beta_0.shape[0]
                    + user_id * Z_gamma_0.shape[0] : Z_beta_0.shape[0]
                    + (user_id + 1) * Z_gamma_0.shape[0]
                ],
            ]
        )

        # Advantage function variance
        # It is the variance of the beta and gamma_i for the user i
        user_cov = np.block(
            [
                [
                    self.Sigma_beta_gamma[0 : Z_beta_0.shape[0], 0 : Z_beta_0.shape[0]],
                    self.Sigma_beta_gamma[
                        0 : Z_beta_0.shape[0],
                        Z_beta_0.shape[0]
                        + user_id * Z_gamma_0.shape[0] : Z_beta_0.shape[0]
                        + (user_id + 1) * Z_gamma_0.shape[0],
                    ],
                ],
                [
                    self.Sigma_beta_gamma[
                        Z_beta_0.shape[0]
                        + user_id * Z_gamma_0.shape[0] : Z_beta_0.shape[0]
                        + (user_id + 1) * Z_gamma_0.shape[0],
                        0 : Z_beta_0.shape[0],
                    ],
                    self.Sigma_beta_gamma[
                        Z_beta_0.shape[0]
                        + user_id * Z_gamma_0.shape[0] : Z_beta_0.shape[0]
                        + (user_id + 1) * Z_gamma_0.shape[0],
                        Z_beta_0.shape[0]
                        + user_id * Z_gamma_0.shape[0] : Z_beta_0.shape[0]
                        + (user_id + 1) * Z_gamma_0.shape[0],
                    ],
                ],
            ]
        )

        # Compute the advantage mean and variance
        adv_mean = fs.T @ user_mean
        adv_var = fs.T @ user_cov @ fs

        # Check with the other posterior calculation
        mu_it, Sigma_it = self.calculate_posteriors_using_stats(user_id)

        if self.last_time >= 0:
            assert np.allclose(mu_it, user_mean)
            assert np.allclose(Sigma_it, user_cov)

        # Use smoothing function rho to determine action probability
        pi = self.rho_func(mean=adv_mean, var=adv_var)

        # Record the action probability
        self.action_prob[time_step][user_id] = pi

        # Select action based on Bernoulli sample with probability pi, using the random generator
        action = bernoulli.rvs(pi, random_state=self.rng)

        # Record the parameters for the action selection function
        # for after study analysis
        if time_step != 0:
            betas = np.concatenate(
                [
                    self.lambdat[self.last_time].flatten(),
                    self.E[self.last_time][
                        np.triu_indices(self.E[self.last_time].shape[0])
                    ].flatten(),
                ]
            )
        else:
            betas = np.concatenate(
                [
                    self.mu_beta.flatten(),
                    (
                        (1 / self.m)
                        * self.Sigma_beta[np.triu_indices(self.Sigma_beta.shape[0])]
                    ).flatten(),
                ]
            )
        if time_step + 1 not in self.action_selection_function_dict.keys():
            self.action_selection_function_dict[time_step + 1] = {}
        if time_step == 0:
            self.action_selection_function_dict[time_step + 1][user_id] = (
                betas,
                self.m,
                np.array([[0]]),
                np.array([[0]]),
                np.array([[0]]),
                fs,
                np.linalg.inv(self.Sigma_gamma),
                self.sigma_e2,
            )
        else:
            self.action_selection_function_dict[time_step + 1][user_id] = (
                betas,
                self.m,
                self.A_beta_gamma[self.last_time][user_id],
                self.A_gamma[self.last_time][user_id],
                self.B_gamma[self.last_time][user_id],
                fs,
                np.linalg.inv(self.Sigma_gamma),
                self.sigma_e2,
            )

        return action

    def calculate_statistics(self):
        """
        Compute the statistics E_t and lambda_t for the current time step.
        """

        # Calculate the inverse of the random effects covariance matrix
        sigma_gamma_inv = np.linalg.inv(self.Sigma_gamma)

        # Calculate the intermediate sums
        temp1 = np.zeros((sigma_gamma_inv.shape[0], sigma_gamma_inv.shape[1]))
        temp2 = np.zeros((sigma_gamma_inv.shape[0], sigma_gamma_inv.shape[1]))

        for i in range(self.m):
            A_it_gamma = np.linalg.inv(
                self.A_gamma[self.last_time][i] + (self.sigma_e2) * sigma_gamma_inv
            )
            temp1 += (
                self.A_beta_gamma[self.last_time][i]
                @ A_it_gamma
                @ self.A_beta_gamma[self.last_time][i].T
            )

            temp2 += (
                self.A_beta_gamma[self.last_time][i]
                @ A_it_gamma
                @ self.B_gamma[self.last_time][i]
            )

        # Compute the E_t and lambda_t values
        E_t = (
            1
            / (self.m)
            * (
                np.linalg.inv(self.Sigma_beta)
                + (np.sum(self.A_beta[self.last_time], axis=0) / self.sigma_e2)
                - (temp1 / self.sigma_e2)
            )
        )

        lambda_t = (
            (1 / self.m)
            * np.linalg.inv(E_t)
            @ (
                np.linalg.inv(self.Sigma_beta) @ self.mu_beta
                + 1 / self.sigma_e2 * np.sum(self.B_beta[self.last_time], axis=0)
                - 1 / self.sigma_e2 * temp2
            )
        )

        self.E[self.last_time] = E_t
        self.lambdat[self.last_time] = lambda_t

        # The last time step's update is never used
        if self.last_time != self.T - 1:
            # Now save user-specific parameters for the estimating equation function
            # The policy numbers are 1-indexed
            self.estimating_equation_function_dict[self.last_time + 1] = {}
            betas = np.concatenate(
                [lambda_t.flatten(), E_t[np.triu_indices(E_t.shape[0])].flatten()]
            )
            for i in range(self.m):
                self.estimating_equation_function_dict[self.last_time + 1][i] = (
                    betas,
                    self.m,
                    self.states[self.last_time][i],
                    self.A_beta[self.last_time][i],
                    self.A_beta_gamma[self.last_time][i],
                    self.A_gamma[self.last_time][i],
                    self.B_beta[self.last_time][i],
                    self.B_gamma[self.last_time][i],
                    self.actions[self.last_time][i],
                    self.action_prob[self.last_time][i],
                    self.last_time + 1,
                    self.outcomes[self.last_time][i],
                    self.mu_beta,
                    np.linalg.inv(self.Sigma_beta),
                    np.linalg.inv(self.Sigma_gamma),
                    self.sigma_e2,
                )

    def calculate_posteriors_using_stats(self, user_id: int):
        """
        Update the posterior distribution using the statistics E_t and lambda_t.
        """

        if self.last_time == -1:
            mean = np.vstack([self.mu_beta, np.zeros(self.mu_beta.shape)])
            cov = np.block(
                [
                    [self.Sigma_beta, np.zeros(self.Sigma_beta.shape)],
                    [np.zeros(self.Sigma_beta.shape), self.Sigma_gamma],
                ]
            )
            return mean, cov

        # Compute the inverse of the mE_t matrix
        mE_t_inv = (1 / self.m) * np.linalg.inv(self.E[self.last_time])

        # Compute tilde_Ait_gamma
        Ait_gamma_inv = np.linalg.inv(
            self.A_gamma[self.last_time][user_id]
            + (self.sigma_e2) * np.linalg.inv(self.Sigma_gamma)
        )

        # Compute the posterior mean and covariance
        mu_it = np.vstack(
            [
                self.lambdat[self.last_time],
                Ait_gamma_inv
                @ (
                    self.B_gamma[self.last_time][user_id]
                    - self.A_beta_gamma[self.last_time][user_id].T
                    @ self.lambdat[self.last_time]
                ),
            ]
        )

        Sigma_it = np.block(
            [
                [
                    mE_t_inv,
                    -mE_t_inv
                    @ self.A_beta_gamma[self.last_time][user_id]
                    @ Ait_gamma_inv,
                ],
                [
                    -Ait_gamma_inv
                    @ self.A_beta_gamma[self.last_time][user_id].T
                    @ mE_t_inv,
                    self.sigma_e2 * Ait_gamma_inv
                    + Ait_gamma_inv
                    @ self.A_beta_gamma[self.last_time][user_id].T
                    @ mE_t_inv
                    @ self.A_beta_gamma[self.last_time][user_id]
                    @ Ait_gamma_inv,
                ],
            ]
        )

        return mu_it, Sigma_it

    def save_results(self):
        """
        Save the results to the output path.
        """
        self.study_df = self.study_df.astype(
            {
                "calendar_time": int,
                "user_id": int,
                "in_study_indicator": int,
                "action": int,
                "policy_number": int,
                "action_probability": float,
                "outcome": float,
                "state": float,
            }
        )
        with open(self.output_path + "study_df.pkl", "wb") as f:
            pkl.dump(self.study_df, f)

        with open(
            self.output_path + "estimating_equation_function_dict.pkl", "wb"
        ) as f:
            pkl.dump(self.estimating_equation_function_dict, f)

        with open(self.output_path + "action_selection_function_dict.pkl", "wb") as f:
            pkl.dump(self.action_selection_function_dict, f)
