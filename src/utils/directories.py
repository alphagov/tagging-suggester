import os as dir_os
import pickle as dir_pickle

def project_root_dir():
    return dir_os.getcwd()

def data_dir():
    return dir_os.path.abspath(project_root_dir() + "/data")

def raw_data_dir():
    return dir_os.path.abspath(data_dir() + "/raw")

def processed_data_dir():
    return dir_os.path.abspath(data_dir() + "/processed")

def open_pickle_file(path):
    return dir_pickle.load(open(path, "rb"))

def save_pickle_file(variable, path):
    dir_pickle.dump(variable, open(path, "wb"))
