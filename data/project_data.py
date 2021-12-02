import pandas as pd
import numpy as np
import os

data_file_path = os.path.join('.', 'data1cleaned.csv')
total_data = pd.read_csv(data_file_path, sep=',').values

# Clean out first 9 columns
total_data = np.delete(total_data, list(range(9)), 1)


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

