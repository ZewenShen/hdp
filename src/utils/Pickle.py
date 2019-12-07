import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import pickle

def dump(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)