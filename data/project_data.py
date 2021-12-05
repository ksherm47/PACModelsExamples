import pandas as pd
import numpy as np
import pickle
import zipfile
import os

data_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(data_dir, 'data1cleaned.csv')
total_data = pd.read_csv(data_file_path, sep=',').values

# Clean out first 9 columns
total_data = np.delete(total_data, list(range(9)), 1)

data_dim = total_data.shape[1] - 1  # subtract label

# Extract any compressed data detected
__data_zip_path = os.path.join(data_dir, 'saved_data.zip')
if os.path.exists(__data_zip_path):
    print(f'Extracting previously saved data from {__data_zip_path}...')
    with zipfile.ZipFile(__data_zip_path, 'r') as zr:
        zr.extractall(data_dir)
    os.remove(__data_zip_path)


def zip_data(data_objects: list, clean: bool = False, remove_previous: bool = False):
    if remove_previous and os.path.exists(__data_zip_path):
        os.remove(__data_zip_path)

    curr_dir = os.getcwd()
    os.chdir(data_dir)
    with zipfile.ZipFile(__data_zip_path, 'w') as zw:
        for obj in data_objects:
            zw.write(obj)
            if clean:
                os.remove(obj)
    os.chdir(curr_dir)


def get_data_sample(num_samples):
    m = num_samples if num_samples < total_data.shape[0] else total_data.shape[0]

    np.random.shuffle(total_data)
    train_data = total_data[:m, :]
    test_data = total_data[m:, :]
    train_labels = train_data[:, 25]
    test_labels = test_data[:, 25]
    train_data = np.delete(train_data, 25, 1)
    test_data = np.delete(test_data, 25, 1)

    return train_data, train_labels, test_data, test_labels


def get_full_data():
    full_data, full_data_labels, _, _ = get_data_sample(total_data.shape[0])
    return full_data, full_data_labels


def save_data_obj(data_object, filename, protocol=5):
    full_path = os.path.join(data_dir, filename)
    with open(full_path, 'wb') as fw:
        pickle.dump(data_object, fw, protocol=protocol)


def get_data_obj(filename):
    full_path = os.path.join(data_dir, filename)
    with open(full_path, 'rb') as fr:
        data_object = pickle.load(fr)
    return data_object


def delete_data_obj(filename):
    full_path = os.path.join(data_dir, filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def data_obj_exists(filename):
    full_path = os.path.join(data_dir, filename)
    return os.path.exists(full_path)
