from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class Mfeat():
    def __init__(self, path):

        data = scipy.io.loadmat(path + 'Mfeat.mat')
        scaler = MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(2000,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['data'][0][2].T.astype(np.float32))
        self.V4 = scaler.fit_transform(data['data'][0][3].T.astype(np.float32))
        self.V5 = scaler.fit_transform(data['data'][0][4].T.astype(np.float32))
        self.V6 = scaler.fit_transform(data['data'][0][5].T.astype(np.float32))
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "Mfeat":
        dataset = Mfeat('./data/')
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num