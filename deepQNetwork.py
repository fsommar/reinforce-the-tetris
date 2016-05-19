import numpy as np
from keras.models import Sequential
import theano.tensor as T


def buildNetwork():
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    network = Sequential()
    # no. filters, filter size x, y, stride(x, y), shape(channels, x, y)
    network.add(Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu', input_shape=(4, 20, 20)))
    network.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    network.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))

    network.add(Flatten())
    network.add(Dense(512, activation='relu'))
    # Output layer. 4 possible actions: {left, right, rotate, pass}
    network.add(Dense(4))
    return network


def lossFunction(y_true, y_pred):
    return T.sqr(y_true - y_pred)


# TODO: Update function below with class experienceTransition.

def trainOnBatch(network: Sequential, targetNetwork: Sequential, transitionBatch, gamma: float):
    ys = np.zeros((len(transitionBatch), 4))
    for i, transition in enumerate(transitionBatch):
        ys[i] = transition.reward
        if not transition.terminates:
            target = max(targetNetwork.predict_on_batch(transition.nextFrame)[0])
            ys[i] += gamma * target

    # TODO: Want to have "identity training" for all actions except transition.action
    network.train_on_batch()