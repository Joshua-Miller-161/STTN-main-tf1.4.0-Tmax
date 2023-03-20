# @Time     : Jan. 10, 2019 15:26
# @Author   : Josh Miller/Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(data_seq, n_frame, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param data_seq: np.ndarray, source data / time-series.
    :param n_frame: int, the number of frames within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [np.shape(data_seq)[0]-n_frame, n_frame, np.shape(data_seq)[1], C_0].
    '''

    print(" =====", np.shape(data_seq)[0], n_frame, np.shape(data_seq)[1], C_0)
    sequences = np.zeros((np.shape(data_seq)[0] - n_frame, n_frame, np.shape(data_seq)[1], C_0))

    for i in range(np.shape(sequences)[0]):
        temp_seq = data_seq[i:i+n_frame, ...]

        print('i =', i, ', i+n_frame =', i + n_frame, ', n_frame =', n_frame, ', temp_seq =', np.shape(temp_seq), np.shape(temp_seq.reshape(np.shape(temp_seq)[0], np.shape(temp_seq)[1], 1)), ', data_seq =', np.shape(data_seq))

        sequences[i, ...] = temp_seq.reshape(np.shape(temp_seq)[0], np.shape(temp_seq)[1], 1)
    return sequences


def data_gen(file_path, data_config, n_frame=21):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :return: dict, dataset that contains training, validation and test with stats.
    '''

    train_ratio, val_ratio, test_ratio = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path).values

    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    print(" ----------------- np.shape(data_seq) = ", np.shape(data_seq)) # (365, 350)

    print(' ()()()()()()()()() min_tmax =', min(data_seq.ravel().ravel().ravel()))
    print(' ()()()()()()()()() max_tmax =', max(data_seq.ravel().ravel().ravel()))

    ''' Get all the sequences of lenght n_frame '''
    sequences = seq_gen(data_seq, n_frame)


    ''' Split into training, validation, and test datasets '''
    seq_train, seq_val, seq_test = train_val_test_split(sequences, train_ratio, val_ratio, test_ratio)

    print(" ------------------------------------ ", np.shape(seq_train), np.shape(seq_val), np.shape(seq_test))

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]


def train_val_test_split(data, train_ratio, val_ratio, test_ratio, random_state=69):
    '''
    Split data into training, validation, and testing data subsets.
    :param data: np.array, the data to be split - more than 1 dimensional.
    :param train_ratio: float, the percentage of the data to be subset for training.
    :param val_ratio: float, the percentage of the data to be subset for validation.
    :param test_ratio: float, the percentage of the data to be subset for testing.
    :param random_state: int, seed for the shuffler.
    '''
    assert abs(train_ratio + val_ratio + test_ratio - 1 < 1e-8), 'Error: train_ratio + val_ratio + test_ratio must equal 1.'

    np.random.seed(random_state)

    idx = np.arange(0, np.shape(data)[0])
    np.random.shuffle(idx)

    num_train = int(round(train_ratio * len(idx)))
    num_val   = int(round(val_ratio * len(idx)))
    num_test  = len(idx) - num_train - num_val
    
    dims_list = list(np.shape(data)[1:])
    print(" +++++", np.shape(data), np.shape(data)[1:], dims_list)

    dims_list_train = [num_train]
    dims_list_val   = [num_val]
    dims_list_test  = [num_test]

    for x in dims_list:
        dims_list_train.append(x)
        dims_list_val.append(x)
        dims_list_test.append(x)

    print(" +++++", dims_list_train, dims_list_val, dims_list_test)

    train_arr = np.zeros(tuple(dims_list_train), float)
    val_arr   = np.zeros(tuple(dims_list_val), float)
    test_arr  = np.zeros(tuple(dims_list_test), float)
    
    for i in range(num_train):
        train_arr[i, ...] = data[idx[i], ...]

    for i in range(num_train, num_train + num_val):
        val_arr[i - num_train, ...] = data[idx[i], ...]

    for i in range(num_train + num_val, len(idx)):
        test_arr[i - (num_train + num_val), ...] = data[idx[i], ...]

    return train_arr, val_arr, test_arr


def Scale(data, reference, method='standard'):
    '''
    This function scales the data, either using the StandardScaler or MaxAbsScaler
    :param data: nparray, the data to be scaled 
    :param reference: nparray, what to use as a reference for the scaler, must be same size as data
    :param method: str, either 'standard' or 'maxabs', chooses the scaling method
    '''
    data = np.array(data)
    reference = np.array(reference)

    orig_shape = np.shape(data)
    print("-0-0-0-0-0-0-0-0-0-0-0-0- np.shape(data) =", np.shape(data), ", np.shape(reference) =", np.shape(reference), ', orig_shape =', orig_shape)
    if method == 'standard': # Makes mean = 0, stdev = 1
        mu  = np.mean(reference.ravel().ravel().ravel())
        std = np.std(reference.ravel().ravel().ravel())

        def Scaler(data):
            return (data - mu) / std

    elif method == 'maxabs': # Divdes by max value, better for sparse data
        max_ = max(abs(reference.ravel().ravel().ravel()))

        def Scaler(data):
            return data / max_
    else:
        raise ValueError('Invalid method specified. Allowed values are "standard" and "maxabs".')
    
    scaled_data = Scaler(data.ravel().ravel().ravel())

    return scaled_data.reshape(orig_shape)