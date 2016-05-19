import sys
from keras.models import Sequential


def saveArchitecture(network, fileName: str):
    try:
        # Uses 'with' to ensure that file is closed properly
        with open(fileName + '_architecture.json', 'w') as f:
            f.write(network.to_json())
        return True  # Save successful
    except:
        print(sys.exc_info())  # Prints exceptions
        return False  # Save failed


def saveWeights(network: Sequential, fileName: str):
    network.save_weights(fileName + '.h5')


def saveArchitectureAndWeights(network: Sequential, architectureFileName: str, weightsFileName: str):
    saveArchitecture(network, architectureFileName)
    saveWeights(network, weightsFileName)


def loadArchitecture(fileName: str) -> Sequential:
    from keras.models import model_from_json
    with open(fileName + '_architecture.json') as f:
        network = model_from_json(f.read())
    return network


def loadWeights(network: Sequential, fileName: str) -> Sequential:
    network.load_weights(fileName + 'h5')
    return network


def loadAcrhitectureAndWeights(architectureFileName: str, weightsFileName: str) -> Sequential:
    network = loadArchitecture(architectureFileName)
    return loadWeights(network, weightsFileName)
