# Abstract class for policy

from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def choose_action(self, user_id, time_step, state):
        """
        Choose an action based on the user, time step, and state.

        Parameters:
            user_id (int): The ID of the user.
            time_step (int): The current time step.
            state (float): The current state of the user.

        Returns:
            int: The chosen action.
        """
        pass

    def record(self, user_id, time_step, state, action, outcome):
        """
        Record the state action outcome tuple.

        Parameters:
            user_id (int): The ID of the user.
            time_step (int): The current time step.
            state (float): The current state of the user.
            action (int): The chosen action.
            outcome (float): The observed outcome.
        """
        pass

    def update_policy(self):
        """
        Update the policy
        """
        pass
