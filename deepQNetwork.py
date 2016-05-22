from typing import List
import numpy as np

from keras.models import Sequential
from keras.optimizers import RMSprop

from ExperienceTransition import ExperienceTransition
import utils


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
                 transitionBatch: List[ExperienceTransition], gamma: float=0.99):
    ys = np.zeros(shape=(len(transitionBatch), 4))
    xs = []
    for i, experienceTransition in enumerate(transitionBatch):
        xs.append(experienceTransition.preprocessedSequences)
        for j in range(4):
            if j == experienceTransition.action:
                ys[i][j] = experienceTransition.reward
                # If the game does not terminate, add value from next sequence
                if not experienceTransition.doesTerminate():
                    # TODO: Change to use the upcoming function in ExperienceTransition
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


def compileNetwork(network: Sequential, learningRate: float=0.00025, rho: float=0.95):
    network.compile(optimizer=RMSprop(lr=learningRate, rho=rho), loss=lossFunction)
    # These two calls take some seconds to perform,
    # because the first predict and train calls take longer time.
    network.predict_on_batch(np.zeros(shape=(1, 4, 20, 20)))
    network.train_on_batch(np.zeros(shape=(1, 4, 20, 20)), np.zeros(shape=(1, 4)))
