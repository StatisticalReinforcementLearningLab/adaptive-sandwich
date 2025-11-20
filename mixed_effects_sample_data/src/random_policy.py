import numpy as np
from policy import Policy

class RandomPolicy(Policy):
    def __init__(self, seed=None):
        """Initialize the random policy."""
        
        # Initialize a random generator with the provided seed
        self.rng = np.random.default_rng(seed)

    def choose_action(self, user_id, time_step, state):
        """Choose an action randomly."""
        return self.rng.choice([0, 1])

    def record(self, user_id, time_step, state, action, outcome):
        """No record needed for a random policy."""
        pass

    def update_policy(self):
        """No update needed for a random policy."""
        pass