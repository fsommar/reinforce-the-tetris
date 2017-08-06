import pickle

with open("replay_memory.dat", "wb") as out:
    with open("test.dat", "rb") as f:
        list = pickle.load(f)
    for x in list:
        if x.nextSequence is None:
            x.reward = -1
    pickle.dump(list, out)
