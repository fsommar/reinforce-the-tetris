#
# File used to test functionality of deepQNetwork
#

from keras.models import Sequential
from keras.layers import Dense
import copy
import random
import numpy as np
from keras.optimizers import RMSprop
from ExperienceTransition import ExperienceTransition
import deepQNetwork as dqn


architectureFileName = "test_architecture"
weightsFileName = "test_weights.h5"

network = dqn.buildNetwork()
dqn.compileNetwork(network, learningRate=0.00025, rho=0.95)

batch = []
for _ in range(3): # Simulate batch och ExperienceReplays with randomized images
    data = np.random.rand(4, 20, 20)
    data2 = np.random.rand(4, 20, 20)
    action = random.randint(0,3)
    score = random.random()
    batch.append(ExperienceTransition(data, action, score, data2))

# Make sure this runs without exception, which it does :)
dqn.trainOnBatch(network, network, batch, 0.1)