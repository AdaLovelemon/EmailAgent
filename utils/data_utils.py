import os
import yaml
from argparse import Namespace
import scipy.io
import numpy as np
import pickle

def list_all_items(directory_path):
    items = os.listdir(directory_path)
    # join the directory path with each item
    full_paths = [os.path.join(directory_path, item) for item in items]
    return full_paths

def list_items_with_postfix(file_dir, postfix):
    """
    Args:
        file_dir: The directory from which you want to extract data
        postfix: The postfix with which the data you want to extract ends.
     
    """
    paths = []
    for file_name in os.listdir(file_dir):
        if file_name.endswith(postfix):
            paths.append(os.path.join(file_dir, file_name))

    return paths

def load_mat_file(file_path):
    """
    Extracted Results:
    -----------------
        - `__header__`: header of the file.
        - `__version__`: version of the matlab file.
        - `__globals__`: global variables.

        - Other variables in the file are true data.
    """
    data = scipy.io.loadmat(file_path)
    return data


def load_dat_file(file_path, load_pickle=False):
    data = None
    encodings = ['utf-8', 'latin1', 'ascii', 'utf-16', 'utf-32', 'cp1252', 'gbk', 'shift_jis']
    if not load_pickle:
        for encoding in encodings:
            try:
                data = np.loadtxt(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
    else:
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path} with available encodings.")
    return data


def load_npy_file(file_path):
    data = np.load(file_path)
    return data

def load_config_as_dict(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def dict_to_namespace(dictionary):
    namespace = Namespace()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config_as_namespace(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return dict_to_namespace(config)

def namespace_to_dict(config):
    config_dict = {k: vars(v) if isinstance(v, Namespace) else v for k, v in vars(config).items()}
    return config_dict


def save_config_as_yaml(config, file_path):
    if isinstance(config, Namespace):
        config = namespace_to_dict(config)

    with open(file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def load_pwd(dir):
    with open(dir, 'r') as file:
        pwd = file.read()
    return pwd