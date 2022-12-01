import sys, glob, os

try:
    #
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    #
    # sys.path.append('../carla/agents/navigation')
    sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')
    sys.path.append('rss/') # rss
    sys.path.append('LBC/') # LBC

except IndexError:
    pass


import torch
from map_model import MapModel
import common
import pathlib
import uuid
import copy
from collections import deque
import cv2
import numpy as np

from PIL import Image
import math

from utils.heatmap import ToHeatmap


net = MapModel.load_from_checkpoint('./model_weight/epoch=23.ckpt')
# net.cuda()
# net.eval()

# net = torch.load('./model_weight/model_epoch_net49.pt')
net.cuda()


# print(net)

net.eval()

N_CLASSES = len(common.COLOR) #6 
# print(N_CLASSES)



topdown = Image.open("./test_data/instance_segmentation/00000095.png")
topdown = topdown.crop((128, 0, 128 + 256, 256))
topdown = np.array(topdown)
topdown = topdown[:, :, 0]
topdown = common.CONVERTER[topdown]
topdown = torch.LongTensor(topdown)
topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()



u = np.float32( [-52.35981369018555, -3.0144176483154297 ])  

theta = 181.2656390621597
theta = math.radians(theta)
if np.isnan(theta):
    theta = 0.0
# theta = theta #+ np.pi / 2
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
    ])

v = np.array([-77.99332427978516, 12.793258666992188 ])

target = R.T.dot(v - u)
target *= 5.5 #PIXELS_PER_WORLD

target += [128, 256] # 128 256
target = np.clip(target, 0, 256) # 0, 256
target = torch.FloatTensor(target)

topdown = topdown.reshape([1, 6, 256, 256])
target = target.reshape([1, 2])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

topdown = topdown.to(device)
target = target.to(device)

to_heatmap = ToHeatmap()

# target_heatmap =  to_heatmap(target, topdown)[:, None]
# out = net(torch.cat((topdown, target_heatmap), 1))

points_pred = net.forward(topdown, target)
print(points_pred)

control = net.controller(points_pred).cpu().squeeze()
print(control)


