from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class UCI_digit():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'uci-digit.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000,)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == 'UCI-digit':
        dataset = UCI_digit('./data/')
        dims = [64, 76, 216]
        view = 3
        data_size = 2000
        class_num = 10

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
