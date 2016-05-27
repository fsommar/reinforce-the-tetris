from typing import List

import numpy as np
import copy
import os
import random
import pickle
from keras.models import Sequential
from keras.optimizers import RMSprop

from ExperienceTransition import ExperienceTransition
import utils

REPLAY_MEMORY_SIZE = 100000  # 1'000'000 in original report
REPLAY_START_SIZE = REPLAY_MEMORY_SIZE / 20
REPLAY_FILE = "potentially_large_supervised_replay_memory.dat"
UPDATE_FREQUENCY = NETWORK_UPDATE_FREQUENCY = REPLAY_MEMORY_SIZE / 1000
EPSILON_START = 0.15
EPSILON_END = 0.1
EPSILON_ANNEAL_FACTOR = 10000


def buildNetwork() -> Sequential:
    from keras.layers.core import Dense, Flatten
    from keras.layers.convolutional import Convolution2D

    network = Sequential()
    # 3 convolutional layers, each followed by a ReLU activation
    # no. filters, filter size x, y, stride(x, y), shape(channels, x, y)
    network.add(Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu', input_shape=(4, 20, 20)))
    network.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    network.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    # 2 fully connected layers, the last being the classifier
    network.add(Flatten())
    network.add(Dense(512, activation='relu'))
    # Output layer. 4 possible actions: {pass, left, right, rotate}
    network.add(Dense(4))
    return network


def lossFunction(y_true, y_pred):
    import theano.tensor as T
    return T.sqr(y_true - y_pred)


def trainOnBatch(learningNetwork: Sequential, targetNetwork: Sequential,
                 transitionBatch: List[ExperienceTransition], gamma: float = 0.99):
    ys = np.zeros(shape=(len(transitionBatch), 4))
    xs = []
    for i, experienceTransition in enumerate(transitionBatch):
        xs.append(experienceTransition.preprocessedSequences)
        for j in range(4):
            if j == experienceTransition.action:
                ys[i][j] = experienceTransition.reward
                # If the game does not terminate, add value from next sequence
                if not experienceTransition.doesTerminate():
                    fi_t_plus_one = np.roll(experienceTransition.preprocessedSequences, 1, axis=0)
                    fi_t_plus_one[0] = experienceTransition.nextSequence
                    target = max(predictOnSequence(targetNetwork, fi_t_plus_one))
                    ys[i][j] += gamma * target
            else:
                # For the other 3 actions, we don't want the network to learn anything new,
                # but we still need to provide a target for the (supervised) training
                ys[i][j] = max(predictOnSequence(learningNetwork, experienceTransition.preprocessedSequences))

    xs = np.stack(xs)
    learningNetwork.train_on_batch(xs, ys)


def predictOnSequence(network: Sequential, sequence: np.ndarray) -> List[float]:
    return network.predict_on_batch(utils.sequenceAsBatch(sequence))[0]


def predictBestAction(network: Sequential, sequence: np.ndarray) -> int:
    return np.argmax(predictOnSequence(network, sequence))


def compileNetwork(network: Sequential, learningRate: float = 0.00025, rho: float = 0.95):
    network.compile(optimizer=RMSprop(lr=learningRate, rho=rho), loss=lossFunction)
    # These two calls take some seconds to perform,
    # because the first predict and train calls take longer time.
    network.predict_on_batch(np.zeros(shape=(1, 4, 20, 20)))
    network.train_on_batch(np.zeros(shape=(1, 4, 20, 20)), np.zeros(shape=(1, 4)))


def loadReplaysIfExist() -> List[ExperienceTransition]:
    if os.path.isfile(REPLAY_FILE):
        print("Loaded replays")
        with open(REPLAY_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return [None] * REPLAY_MEMORY_SIZE


class DQNData:
    def __init__(self, learningNetwork: Sequential, gamma: float = 0.99):
        self.learningNetwork = learningNetwork  # type: Sequential
        self.targetNetwork = copy.deepcopy(learningNetwork)  # type: Sequential
        compileNetwork(self.learningNetwork)
        compileNetwork(self.targetNetwork)

        self.gamma = gamma  # type: float
        self.epsilon = EPSILON_START  # type: float
        self.trainCount = 0  # type: int

        self.replayMemory = loadReplaysIfExist()  # type: List[ExperienceTransition]
        self.preprocessedSequences = np.zeros((4, 20, 20), dtype=np.bool)  # type: np.ndarray
        self.replays = 0  # type: int
        # Enter new sequences after the existing ones, if replayMemory is not full
        try:
            self.replays = self.replayMemory.index(None)
        except ValueError:
            pass
        print(self.replays)
        self.prevScore = 0  # type: int
        self.prevState = np.zeros((20, 20))  # type: np.ndarray
        self.currentState = np.zeros((20, 20))  # type: np.ndarray

    def update(self, action: int, score: int = 0, gameOver: bool = False) -> None:
        # Roll the channels (the first dimension in the shape) to make place for the latest state.
        self.preprocessedSequences = np.roll(self.preprocessedSequences, 1, axis=0)
        # Let the previous state (i.e. current sequence) be the first item in the tensor.
        self.preprocessedSequences[0] = self.prevState
        # Advance one step (corresponding to executing an action).
        prevState = np.copy(self.currentState)

        pps = np.copy(self.preprocessedSequences)
        if gameOver:
            # The action taken lead to a game over.
            self.replayMemory[self.replays] = ExperienceTransition(pps,
                                                                   action=action,
                                                                   reward=0,
                                                                   # A nextSequence of None indicates a game over.
                                                                   nextSequence=None)
        else:
            # It's important not to pass any references as the arrays WILL be changed later.
            self.replayMemory[self.replays] = ExperienceTransition(pps,
                                                                   action=action,
                                                                   reward=score - self.prevScore,
                                                                   # Un-intuitively the NEW prevState is the actual
                                                                   # sequence resulting from the action.
                                                                   nextSequence=prevState)

        # Make sure not to go out of bounds in the replay memory.
        self.replays = (self.replays + 1) % REPLAY_MEMORY_SIZE
        self.prevScore = score

        # Update the target network every UPDATE_FREQUENCY frame
        self.trainCount = (self.trainCount + 1) % UPDATE_FREQUENCY
        if self.trainCount == 0:
            self.targetNetwork.set_weights(self.learningNetwork.get_weights())
            print("Target network updated!")
            print("epsilon is now {}".format(self.epsilon))

        if self.epsilon > EPSILON_END:
            self.epsilon -= 1 / EPSILON_ANNEAL_FACTOR


    def trainOnMiniBatch(self, miniBatch: List[ExperienceTransition]) -> None:
        trainOnBatch(self.learningNetwork, self.targetNetwork, miniBatch, self.gamma)

    def predictAction(self) -> int:
        """
        Uses an epsilon-greedy policy to select the next action to perform;
        with an epsilon probability to perform a random action, otherwise use
        the network to predict the best action to take.
        :return: the action to perform in range [0, 3].
        """
        if random.random() < self.epsilon:
            return random.choice(range(4))
        else:
            return predictBestAction(self.learningNetwork, self.preprocessedSequences)

    def getFilledMemory(self) -> List[ExperienceTransition]:
        if self.replayMemory[self.replays] is not None:
            return self.replayMemory
        return self.replayMemory[:self.replays]

    def writeReplayFile(self) -> None:
        with open(REPLAY_FILE, "wb") as f:
            pickle.dump(self.replayMemory, f)
        print("Wrote replay memory to {}".format(REPLAY_FILE))

    def show(self):
        import matplotlib.pyplot as plt
        # When matplotlib shows the window with the array, it crashes internally (at least on OS X).
        # The windows are still opened but the game does not continue running.
        _, axes = plt.subplots(2, 2, sharey='row', sharex='col')
        axes[(0, 0)].imshow(self.preprocessedSequences[0].T, cmap='Greys', interpolation='nearest')
        axes[0, 0].set_title("$t_0$")
        axes[0, 1].imshow(self.preprocessedSequences[1].T, cmap='Greys', interpolation='nearest')
        axes[0, 1].set_title("$t_{-1}$")
        axes[1, 0].imshow(self.preprocessedSequences[2].T, cmap='Greys', interpolation='nearest')
        axes[1, 0].set_title("$t_{-2}$")
        axes[1, 1].imshow(self.preprocessedSequences[3].T, cmap='Greys', interpolation='nearest')
        axes[1, 1].set_title("$t_{-3}$")
        plt.show()
