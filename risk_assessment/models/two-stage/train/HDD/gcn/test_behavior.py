import config as cfg
import utils as utl
from models import GCN as Model
from datasets import GCNDataLayer as DataLayer
import os
import sys
import time
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm

sys.path.insert(0, '../../../')


# python test_behavior.py --time_steps 5 --cause interactive

def to_device(x, device):
    return x.to(device)  # .transpose(0,1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='all',
                        type=str, required=True)
    parser.add_argument('--gpu', default='0, 1', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)  # 5e-04
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--time_steps', default=5, type=int, required=True)
    parser.add_argument('--partial_conv', default=True, type=bool)  # True
    parser.add_argument('--data_augmentation', default=True, type=bool)
    parser.add_argument('--dist', default=False, type=bool)
    parser.add_argument('--fusion', default='attn',
                        choices=['avg', 'gcn', 'attn'], type=str)

    args = cfg.parse_args(parser)
    args.phases = ['test']

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args.inputs, args.time_steps, pretrained=False,
                  partialConv=args.partial_conv, fusion=args.fusion)

    # url = 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
    # model.loadmodel(url)

    state_dict = torch.load(
        "snapshots/all/2022-9-1_183046_w_dataAug_attn/inputs-camera-epoch-20.pth")
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]
    model.load_state_dict(state_dict_copy)

    model = nn.DataParallel(model).to(device)
    print("Model Parameters:", count_parameters(model))

    softmax = nn.Softmax(dim=1).to(device)
    camera_transforms = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                      [0.5, 0.5, 0.5],),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


data_sets = {
    phase: DataLayer(
        data_root=args.data_root,
        _cause=args.cause,
        sessions=getattr(args, phase+'_session_set'),
        camera_transforms=camera_transforms,
        time_steps=args.time_steps,
        data_augmentation=args.data_augmentation,
        dist=args.dist
    )
    for phase in args.phases
}

data_loaders = {
    phase: data.DataLoader(
        data_sets[phase],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    for phase in args.phases
}

vel_metrics = []
target_metrics = []

start = time.time()

for phase in args.phases:
    model.train(False)
    with torch.set_grad_enabled(False):
        with tqdm(data_loaders[phase], unit="batch") as tepoch:
            for batch_idx, (camera_inputs, trackers, mask, dist_mask, vel_target) \
                    in enumerate(tepoch):

                tepoch.set_description(f"Testing... ")
                batch_size = camera_inputs.shape[0]

                camera_inputs = to_device(
                    camera_inputs, device)  # (bs, t, c , w, h)
                mask = mask.to(device)
                dist_mask = dist_mask.to(device)
                trackers = to_device(trackers, device)  # (bs, t, n, 4)
                vel_target = to_device(vel_target, device).view(-1)  # (bs)

                vel = model(camera_inputs, trackers,
                            device, dist_mask, mask)

                # print(np.argmax(vel.detach().to('cpu').numpy(), axis=1))
                # print(np.array(vel_target.detach().to('cpu').numpy()))
                # print(vel)

                vel = softmax(vel).to('cpu').numpy()
                vel_target = vel_target.to('cpu').numpy()
                vel_metrics.extend(vel)
                target_metrics.extend(vel_target)

end = time.time()
result_path = 'results'
result_name = 'temp.json'
mAP, result = utl.compute_result(args.class_index, vel_metrics, target_metrics,
                                 result_path, result_name, save=True, verbose=False)
ACC_stop = result['ACC_stop']
ACC_go = result['ACC_go']

print("=====================================")
print(result)
print("=====================================")
print(f'ACC_stop: {ACC_stop:.5f}')
print(f'ACC_go: {ACC_go:.5f}')
print(f'running time: {end-start:.2f} sec')
