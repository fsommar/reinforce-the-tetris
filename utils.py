import sys
from keras.models import Sequential
import numpy as np


def sequenceAsBatch(sequence: np.ndarray) -> np.ndarray:
    # This function converts the array have one additional index
    # Shape (4, 20, 20) -> (1, 4, 20, 20)
    batch = np.stack((sequence,))
    reduced_batch = batch[:, :, 0:10, :]
    return reduced_batch


def saveArchitecture(network, fileName: str):
    try:
        # Uses 'with' to ensure that file is closed properly
        with open(fileName + '.json', 'w') as f:
            f.write(network.to_json())
        return True  # Save successful
    except:
        print(sys.exc_info())  # Prints exceptions
        return False  # Save failed


def saveWeights(network: Sequential, fileName: str, overwrite=False):
    network.save_weights(fileName + '.h5', overwrite)


def saveArchitectureAndWeights(network: Sequential, architectureFileName: str, weightsFileName: str):
    saveArchitecture(network, architectureFileName)
    saveWeights(network, weightsFileName)


def loadArchitecture(fileName: str) -> Sequential:
    from keras.models import model_from_json
    with open(fileName + '.json') as f:
        network = model_from_json(f.read())
    return network


def loadWeights(network: Sequential, fileName: str):
    network.load_weights(fileName + '.h5')


def loadArchitectureAndWeights(architectureFileName: str, weightsFileName: str) -> Sequential:
    network = loadArchitecture(architectureFileName)
    loadWeights(network, weightsFileName)
    return network
