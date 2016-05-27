#
# File used to test functionality of deepQNetwork
#

import os
from keras.models import Sequential
import copy
import numpy as np
import deepQNetwork as dqn
import utils

ARCHITECTURE_FILE = "test_architecture"
INITIAL_WEIGHTS_FILE = "saved_weights.h5"

network = dqn.buildNetwork()
network.load_weights(INITIAL_WEIGHTS_FILE)
weights = network.get_weights()
for i in range(len(weights)):
    print("i = {}, len = {}".format(i, len(weights[i])))
    print(weights[i].shape)
    print(weights[i])
