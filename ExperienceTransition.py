import numpy as np


class ExperienceTransition:
    """
    Represents a tuple containing the current preprocessed sequence (fi),
    the action that's taken, the reward given from the action and the resulting sequence.

    If the action leads to the game over state the resulting sequence is `None`.
    """

    def __init__(self, preprocessedSequences: np.ndarray, action: int, reward: int, nextSequence: np.ndarray):
        self.preprocessedSequences = preprocessedSequences  # type: np.ndarray
        self.action = action  # type: int
        self.reward = reward  # type: int
        self.nextSequence = nextSequence  # type: np.ndarray
