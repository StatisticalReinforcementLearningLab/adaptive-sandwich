import numpy as np
import pandas as pd
from policy import Policy


class Simulator:
    def __init__(
        self,
        num_users: int = 100,
        num_time_steps: int = 10,
        sigma_e: float = 0.1,
        state_mean: float = 1,
        state_std: float = 0.2,
        seed: int = None,
        Delta_t: np.ndarray = None,
    ):
        self.num_users = num_users
        self.num_time_steps = num_time_steps
        self.sigma_e = sigma_e
        self.state_mean = state_mean
        self.state_std = state_std
        self.actions = [0, 1]  # Binary action set {0, 1}

        # Initialize a random generator with the provided seed
        self.rng = np.random.default_rng(seed)

        # Handle Delta_t: if provided, ensure constraint; otherwise, generate within constraint
        if Delta_t is not None:
            # Check if all values in Delta_t satisfy the constraint
            if not np.all(np.abs(Delta_t) <= 1):
                raise ValueError("All values in Delta_t must satisfy |Delta_t| <= 1.")
            self.Delta_t = Delta_t
        else:
            # Generate Delta_t with values within the range [-1, 1]
            self.Delta_t = self.rng.uniform(-1, 1, num_time_steps)

        self.results = []  # To store results

    def generate_state(self):
        """Generate state S_i,t ~ N(state_mean, state_std) independently for each time step."""
        return self.rng.normal(self.state_mean, self.state_std)

    def generate_outcome(self, state: float, action: int, time_step: int):
        """Generate the outcome Y_i,t(A) for a given action A at time t."""
        # f(state) = 0 as per assumption, so outcome depends only on Delta_t, action, and noise
        noise = self.rng.normal(0, self.sigma_e)
        outcome = self.Delta_t[time_step] * action + noise
        return outcome

    def run_simulation(self, policy: Policy):
        """
        Run the simulation across users and time steps.

        Parameters:
            policy (function): A function that takes (user_id, time_step, state) as input and returns an action.
        """
        for time_step in range(self.num_time_steps):
            for user_id in range(self.num_users):
                # Generate state for each user and time step
                state = self.generate_state()

                # Decide action based on the policy function provided
                action = policy.choose_action(user_id, time_step, state)

                # Calculate outcome
                outcome = self.generate_outcome(state, action, time_step)

                # Record the state action with the observed outcome
                policy.record(user_id, time_step, state, action, outcome)

                # Record the results
                self.results.append((user_id, time_step, state, action, outcome))

            # Update the policy at the end of each time step
            policy.update_policy()

        # Convert results to DataFrame for easy analysis
        return pd.DataFrame(
            self.results, columns=["User", "Time", "State", "Action", "Outcome"]
        )
