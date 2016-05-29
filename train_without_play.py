#
# Attempt to speed up training by only training,
# i.e. no simulations and no new states added to memory
#

import random
import os.path

from keras.models import Sequential

import utils
import deepQNetwork as dqn

ARCHITECTURE_FILE = "dqn_architecture"
INITIAL_WEIGHTS_FILE = "initial_weights"
SAVE_WEIGHTS_FILE = "saved_weights"
BATCH_SIZE = 32
MAX_ITER = 200000
SAVE_FREQUENCY = 5000


def loadNetworkIfExists() -> Sequential:
    if (os.path.isfile(ARCHITECTURE_FILE + ".json") and
            os.path.isfile(INITIAL_WEIGHTS_FILE + ".h5")):
        print("Loaded existing architecture and weights")
        return utils.loadArchitectureAndWeights(ARCHITECTURE_FILE, INITIAL_WEIGHTS_FILE)
    network = dqn.buildNetwork()
    utils.saveArchitectureAndWeights(network, ARCHITECTURE_FILE, INITIAL_WEIGHTS_FILE)
    return network


def main():
    dqnData = dqn.DQNData(learningNetwork=loadNetworkIfExists())
    usedMemory = dqnData.getFilledMemory()
    for i in range(MAX_ITER):
        miniBatch = random.sample(usedMemory, BATCH_SIZE)
        dqnData.trainOnMiniBatch(miniBatch)
        print(i)
        if (i % SAVE_FREQUENCY) == 0:
            dqnData.targetNetwork.set_weights(dqnData.learningNetwork.get_weights())
            save_name = "{}_{}".format(SAVE_WEIGHTS_FILE, i)
            utils.saveWeights(dqnData.learningNetwork, save_name)
            print("Saved weights at step {}".format(i))

if __name__ == "__main__":
    main()
