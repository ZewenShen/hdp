import pickle

def dump(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)