import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
# from .inceptionresnetv2 import InceptionResNetV2
from .inceptionresnetv2_partialConv import InceptionResNetV2_Partial
from .roi_align.roi_align import CropAndResize
from .pdresnet50 import pdresnet50

import torch.nn.functional as F
import numpy as np

__all__ = [
    'GCN',
]

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GCN(nn.Module):
    def __init__(self, inputs, time_steps=90, pretrained=True, partialConv=False, fusion='avg'):
        super(GCN, self).__init__()

        self.hidden_size = 512
        self.partialConv = partialConv
        self.fusion = fusion
        if inputs in ['camera', 'sensor', 'both']:
            self.with_camera = 'sensor' not in inputs
            self.with_sensor = 'camera' not in inputs
        else:
            raise(RuntimeError(
                'Unknown inputs of {}, '
                'supported inputs consist of "camera", "sensor", "both"', format(inputs)))
        self.time_steps = time_steps
        self.pretrained = pretrained
        self.num_box = 60  # TODO

        # build backbones
        if self.partialConv:
            self.backbone = pdresnet50(pretrained=pretrained)
            # self.backbone = InceptionResNetV2_Partial(num_classes=1001)
        else:
            self.backbone = InceptionResNetV2(num_classes=1001)

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

        # temporal modeling
        self.drop = nn.Dropout(p=0.1)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        # gcn module
        if self.fusion == 'gcn':
            raise (RuntimeError('GCN fusion is not implemented'))
        elif self.fusion == 'attn':
            self.emb_size = self.hidden_size
            self.fc_emb_1 = nn.Linear(
                self.hidden_size, self.emb_size, bias=False)
            self.fc_emb_2 = nn.Linear(self.emb_size * 2, 1)

        # classifier
        self.vel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

        # initialize weights
        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('BasicConv2d') != -1:
        #         pass
        #     elif classname.find('Conv') != -1:
        #         m.weight.data.normal_(0.0, 0.001)
        #     elif classname.find('BatchNorm') != -1:
        #         m.weight.data.normal_(1.0, 0.001)
        #         m.bias.data.fill_(0.001)

    # load pre-trained model
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

    def message_passing(self, input_feature, trackers, dist_mask=None):
        #########################################
        # input_feature:   (BxN) * 2000
        # trackers:         BxTxNx4
        # dist_mask:        BxN
        #############################################

        if self.fusion == 'avg':
            num_box = trackers.shape[2]
            mask = trackers[:, -1, :, 2]+trackers[:, -1, :, 3]
            mask = mask.view(-1, num_box)  # (BxN)x1
            mask = mask != 0
            # BxNxH
            input_feature = input_feature.view(-1, num_box, self.hidden_size)
            input_feature[~mask.byte()] = 0
            ego_feature = input_feature[:, 0, :]
            updated_feature = torch.sum(
                input_feature, 1) / torch.sum(mask, 1, keepdim=True).float()

            fusion_feature = torch.cat((ego_feature, updated_feature), 1)

        elif self.fusion == 'gcn':
            raise (RuntimeError('GCN fusion is not implemented'))

        elif self.fusion == 'attn':
            num_box = trackers.shape[2]
            mask = trackers[:, -1, :, 2]+trackers[:, -1, :, 3]
            mask = mask != 0  # (B, N, 1)

            # emb_feature: (BxN, H) -> (BxN, self.emb_size)
            emb_feature = self.fc_emb_1(input_feature)
            # emb_feature:  (B, N, self.emb_size)
            emb_feature = emb_feature.view(-1, num_box, self.emb_size)

            # ego_feature: (B, N , self.emb_size)
            ego_feature = emb_feature[:, 0,
                                      :].view(-1, 1, self.emb_size).repeat(1, num_box, 1)

            # emb_feature:(B, N, 2*self.emb_size)
            emb_feature = torch.cat((ego_feature, emb_feature), 2)
            # emb_feature:(BxN, 2*self.emb_size)
            emb_feature = emb_feature.view(-1, 2 * self.emb_size)

            # emb_feature: (B, N, 1)
            emb_feature = self.fc_emb_2(emb_feature).view(-1, num_box, 1)
            emb_feature[~(mask.byte().to(torch.bool))] = torch.tensor(
                [-float("Inf")]).cuda(0)
            '''
            if not dist_mask:
                dist_mask = dist_mask.view(-1, num_box, 1)  # (B, N, 1)
                dist_mask = dist_mask != 0
                emb_feature[~dist_mask.byte()] = torch.tensor([-float("Inf")])
            '''

            # emb_feature: (B, N , 1)
            attn_weights = F.softmax(emb_feature, dim=1)
            # attn_weights: (BxN, 1)
            attn_weights = attn_weights.view(-1, 1)

            # ori_ego_feature : (B, H)
            ori_ego_feature = input_feature.view(-1,
                                                 num_box, self.hidden_size)[:, 0, :]
            # input_feature : (BxN, H)
            input_feature = input_feature.view(-1, self.hidden_size)

            # fusion_feature : (B, N, H)
            fusion_feature = (
                input_feature * attn_weights).view(-1, num_box, self.hidden_size)
            # input_feature : (B, H)
            fusion_feature = torch.sum(fusion_feature, 1)
            # updated_fusion : (B, 2*H)
            fusion_feature = torch.cat((ori_ego_feature, fusion_feature), 1)

            return fusion_feature, attn_weights

        else:
            raise (RuntimeError('{} fusion is not implemented'.format(self.fusion)))

        return fusion_feature, None  # ,attn_weights

    def step(self, camera_input, hx, cx):

        if self.with_camera:
            fusion_input = camera_input
        else:
            raise(RuntimeError('Sensor Data is not Input'))

        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))

        return hx, cx

    def forward(self, camera_inputs, trackers, device, dist_mask, mask=None):
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

        # Define mask if mask does not exists
        if len(mask.size()) == 0:
            mask = torch.ones((batch_size, t, c, w, h)).to(device)

        # assign index for RoIAlign
        # box_ind : (BxN)
        box_ind = np.array([np.arange(batch_size)] *
                           self.num_box).transpose(1, 0).reshape(-1)
        box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

        # initialize LSTM
        hx = torch.zeros(
            (batch_size*self.num_box, self.hidden_size)).to(device)
        cx = torch.zeros(
            (batch_size*self.num_box, self.hidden_size)).to(device)
        logit_vel_stack = []

        # CNN
        if self.partialConv:
            temp = camera_inputs.view(-1, c, w, h)
            temp2 = mask.view(-1, c, w, h)
            camera_inputs = self.backbone.features(temp,
                temp2)  # camera_inputs :(bs,t,c,w,h)
        else:
            camera_inputs = self.backbone.features(
                camera_inputs.view(-1, c, w, h))  # camera_inputs :(bs,t,c,w,h)

        # Reshape the input to LSTM
        c = camera_inputs.shape[1]
        w = camera_inputs.shape[2]
        h = camera_inputs.shape[3]

        camera_inputs = camera_inputs.view(batch_size, t, c, w, h)  # BxTxCxWxH


        # Running LSTM
        for l in range(0, self.time_steps):
            tracker = trackers[:, l].contiguous()
            camera_input = camera_inputs[:, l]

            # ROIAlign : BxNx1280
            feature_input = self.cropFeature(camera_input, tracker, box_ind)
            feature_input = feature_input.view(-1, self.fusion_size)

            # LSTM
            hx, cx = self.step(feature_input, hx, cx)

        updated_feature, _ = self.message_passing(
            hx, trackers, dist_mask)  # BxH

        vel = self.vel_classifier(self.drop(updated_feature))
        logit_vel_stack.append(vel)
        logit_vel_stack = torch.stack(logit_vel_stack).view(-1, 2)

        return logit_vel_stack
