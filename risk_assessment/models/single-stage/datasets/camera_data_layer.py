import os.path as osp

import torch
import torch.utils.data as data
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np

__all__ = [
    'ResNet3DDataLayer',
    'ResNetDataLayer',
]

class ResNet3DDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, camera_transforms, online=True, duration=16):
        self.data_root = data_root
        self.sessions = sessions
        self.camera_transforms = camera_transforms
        self.online = online
        self.duration = duration

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            for start in range(90, target.shape[0]-self.duration):
                if self.online:
                    self.inputs.append([session, start, target[[start+self.duration-1]]])
                else:
                    self.inputs.append([session, start, target[[start+self.duration//2]]])

    def __getitem__(self, index):
        session, start, target = self.inputs[index]

        camera_input_stack = []
        for shift in range(self.duration):
            camera_name = str(start+shift+1).zfill(5)+'.jpg'
            camera_path = osp.join(self.data_root, 'camera', session, camera_name)
            camera_input = self.camera_transforms(Image.open(camera_path).convert('RGB'))
            camera_input_stack.append(camera_input)
        camera_inputs = torch.stack(camera_input_stack).transpose(0,1)

        return camera_inputs, target

    def __len__(self):
        return len(self.inputs)

class ResNetDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, camera_transforms):
        self.data_root = data_root
        self.sessions = sessions
        self.camera_transforms = camera_transforms

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            for idx in range(target.shape[0]):
                self.inputs.append([session, idx, target[idx]])

    def __getitem__(self, index):
        session, idx, target = self.inputs[index]

        camera_name = str(idx+1).zfill(5)+'.jpg'
        camera_path = osp.join(self.data_root, 'camera', session, camera_name)
        camera_input = Image.open(camera_path).convert('RGB')
        camera_input = self.camera_transforms(camera_input)

        return camera_input, target

    def __len__(self):
        return len(self.inputs)
