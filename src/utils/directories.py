import os
import pickle

def project_root_dir():
    return os.getcwd()

def data_dir():
    return os.path.abspath(project_root_dir() + "/data")

def raw_data_dir():
    return os.path.abspath(data_dir() + "/raw")

def processed_data_dir():
    return os.path.abspath(data_dir() + "/processed")

def open_pickle_file(path):
    pickle.load(open(path, "rb"))

def save_pickle_file(variable, path):
    pickle.dump(variable, open(path, "wb"))