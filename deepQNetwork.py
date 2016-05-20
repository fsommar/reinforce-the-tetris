from typing import List
from ExperienceTransition import ExperienceTransition

import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop


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
    # Output layer. 4 possible actions: {left, right, rotate, pass}
    network.add(Dense(4))
    return network


def lossFunction(y_true, y_pred):
    import theano.tensor as T
    return T.sqr(y_true - y_pred)


def trainOnBatch(learningNetwork: Sequential, targetNetwork: Sequential,
                 transitionBatch: List[ExperienceTransition], gamma: float):
    ys = np.zeros(shape=(len(transitionBatch), 4))
    xs = []
    for i, experienceTransition in enumerate(transitionBatch):
        xs.append(experienceTransition.preprocessedSequences)
        for j in range(4):
            if j == experienceTransition.action:
                ys[i][j] = experienceTransition.reward
                if experienceTransition.nextSequence is not None: # Check that the game does not terminate
                    target = max(targetNetwork.predict_on_batch(np.stack((experienceTransition.nextSequence,)))[0])
                    ys[i][j] += gamma * target
            else:
                # For the other 3 actions, we don't want the network to learn anything new,
                # but we still need to provide a target for the (supervised) training
                ys[i][j] = max(learningNetwork.predict_on_batch(
                    np.stack((experienceTransition.preprocessedSequences,)))[0])
    xs = np.stack(xs)
    learningNetwork.train_on_batch(xs, ys)


def compileNetwork(network: Sequential, learningRate: float, rho: float):
    network.compile(optimizer=RMSprop(lr=learningRate, rho=rho), loss=lossFunction)