import pandas as pd
import numpy as np
import os

data_file_path = os.path.join('.', 'data1cleaned.csv')
total_data = pd.read_csv(data_file_path, sep=',').values

# Clean out first 9 columns
total_data = np.delete(total_data, list(range(9)), 1)

# Split into train and test
total_data = np.random.shuffle(total_data)
__split_index = int(0.8 * total_data.shape[0])
train_data = total_data[1:__split_index, :]
test_data = total_data[__split_index:total_data.shape[0], :]
