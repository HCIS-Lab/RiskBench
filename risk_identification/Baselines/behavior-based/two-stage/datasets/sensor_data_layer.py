import os.path as osp

import torch
import torch.utils.data as data
import numpy as np

__all__ = [
    'SensorEncoderDataLayer',
]

class SensorEncoderDataLayer(data.Dataset):
    def __init__(self, data_root, sessions):
        self.data_root = data_root
        self.sessions = sessions

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session+'.npy'))
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            for idx in range(sensor.shape[0]):
                self.inputs.append([sensor[idx], target[[idx]]])

    def __getitem__(self, index):
        sensor_input, target = self.inputs[index]
        sensor_input = torch.from_numpy(sensor_input.astype(np.float32))
        return sensor_input, target

    def __len__(self):
        return len(self.inputs)
