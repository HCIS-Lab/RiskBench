import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


from .GAT import GAT
from .roi_align.roi_align import CropAndResize
from .pdresnet50 import pdresnet50


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GAT_LSTM(nn.Module):
    def __init__(self, inputs, time_steps=90, pretrained=True):
        super(GAT_LSTM, self).__init__()

        self.hidden_size = 128
        self.time_steps = time_steps
        self.num_box = 60
        self.backbone = pdresnet50(pretrained=pretrained)

        if inputs in ['camera', 'sensor', 'both']:
            self.with_camera = 'sensor' not in inputs
            self.with_sensor = 'camera' not in inputs
        else:
            raise(RuntimeError(
                'Unknown inputs of {}, ''supported inputs consist of "camera", "sensor", "both"', format(inputs)))

        # 2d conv after backbones and flatten the features
        self.camera_features = nn.Sequential(
            nn.Conv2d(2048, 20, kernel_size=1),
            # nn.Conv2d(1536, 20, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        # specify feature size
        if self.with_camera:
            self.fusion_size = 1280
        else:
            raise(RuntimeError('Inputs of sensor is invalid'))

        # GAT module, parameter is initialized in GAT.py
        self.gat = GAT()
        self.out_feature_size_per_node = 256

        # temporal modeling
        self.drop = nn.Dropout(p=0.1)
        self.lstm = nn.LSTMCell(
            self.out_feature_size_per_node, self.hidden_size)

        # classifier
        self.vel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

    def loadmodel(self, filepath):
        state_dict = model_zoo.load_url(filepath)
        if self.partialConv:
            # process pretrained model dict for partialConv inceptionv2
            for key in state_dict.keys():
                if 'mixed_5b.branch3.1' in key:
                    new_key = "mixed_5b.branch3_conv2d"+key[18:]
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
                    # print(new_key)

        self.backbone.load_state_dict(state_dict)
        print('Load model states from: ', filepath)

    def cropFeature(self, camera_input, tracker, box_ind, crop_height=8, crop_width=8):
        ###############################################
        # camera_input :    BxCxWxH
        # tracker :         BxNx4
        # box_ind :         (BxN)
        # RoIAlign + flatten features
        ###############################################

        batch_size = tracker.shape[0]
        num_box = tracker.shape[1]

        camera_input = camera_input.contiguous()

        # reshape_tracker :  BxNx4 ->(BxN)x4
        tracker = tracker.view(-1, 4)
        tracker.requires_grad = False

        # box_ind :  [batch_size*num_box]
        box_ind.contiguous().requires_grad = False

        # bounding box coordinates: (y1, x1, y2, x2)
        roi_align = CropAndResize(crop_height, crop_width)
        crops = roi_align(camera_input, tracker, box_ind)

        # 2d conv + flatten : BxNx1280
        crops = self.camera_features(crops)
        crops = crops.view(batch_size, num_box, -1)

        return crops

    def step(self, camera_input, hx, cx):

        if self.with_camera:
            fusion_input = camera_input
        else:
            raise(RuntimeError('Sensor Data is not Input'))

        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))

        return hx, cx

    def forward(self, camera_inputs, trackers, device, hx=None, cx=None):
        ###########################################
        #  camera_input:    BxTxCxWxH
        #  tracker:         BxTxNx4
        ###########################################
        # Record input size
        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        c = camera_inputs.shape[2]
        w = camera_inputs.shape[3]
        h = camera_inputs.shape[4]

        # assign index for RoIAlign
        # box_ind : (BxN)
        box_ind = np.array([np.arange(batch_size)] *
                           self.num_box).transpose(1, 0).reshape(-1)
        box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

        # initialize LSTM
        # hx = torch.zeros(
        #     (batch_size*self.num_box, self.hidden_size)).to(device)
        # cx = torch.zeros(
        #     (batch_size*self.num_box, self.hidden_size)).to(device)
        
        logit_vel_stack = []


        # CNN
        # camera_inputs :(bs*t,c,w,h)
        with torch.no_grad():
            camera_inputs = self.backbone.features(
                camera_inputs.view(-1, c, w, h))
            # Reshape the input to LSTM
            c = camera_inputs.shape[1]
            w = camera_inputs.shape[2]
            h = camera_inputs.shape[3]

            camera_inputs = camera_inputs.view(
                batch_size, t, c, w, h)  # BxTxCxWxH
        

        # Running LSTM
        att_score_lst = []
        for l in range(0, self.time_steps):
            tracker = trackers[:, l].contiguous()
            camera_input = camera_inputs[:, l]

            # ROIAlign : BxNx1280
            feature_input = self.cropFeature(camera_input, tracker, box_ind)
            feature_input = feature_input.view(-1, self.fusion_size).to(device)

            # input for Graph Attention Network
            src_idx = [0]*self.num_box
            tar_idx = [i for i in range(self.num_box)]
            edge_index = np.array([src_idx, tar_idx])
            edge_index = torch.from_numpy(
                edge_index.astype(np.int64)).to(device)

            # Graph attention network -> output_feature: BxNx1280
            data = (feature_input, edge_index)
            lstm_feature_input, att_score = self.gat(data)

            att_score_lst.append(att_score)

            # LSTM
            hx, cx = self.step(lstm_feature_input, hx, cx)

        ret_hx, ret_cx = hx.clone(), cx.clone()        

        # BxNx128
        hx = hx.view(batch_size, self.num_box, -1)
        # Bx128
        hx = torch.sum(hx, dim=1)

        # Bx2
        vel = self.vel_classifier(self.drop(hx))
        logit_vel_stack.append(vel)
        logit_vel_stack = torch.stack(logit_vel_stack).view(-1, 2)

        return logit_vel_stack, att_score_lst, ret_hx, ret_cx
