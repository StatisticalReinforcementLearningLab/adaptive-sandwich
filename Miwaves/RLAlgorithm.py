
# Import libraries
from abc import ABC, abstractmethod

# Create RL algorithm abstract class
class RLAlgorithm(ABC):
    """Abstract class for RL algorithm"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, user, decision_idx):
        pass

    @abstractmethod
    def update_data(self, user, decision_idx, data):
        pass