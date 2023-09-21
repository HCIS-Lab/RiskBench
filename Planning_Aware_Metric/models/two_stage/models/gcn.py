import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.ops import roi_align
from .backbone import Riskbench_backbone
from .pdresnet50 import pdresnet50
import torch.nn.functional as F
import numpy as np

__all__ = [
    'GCN',
]


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class GCN(nn.Module):
    def __init__(self, time_steps=5, pretrained=True, partialConv=True, use_intention=False, NUM_BOX=25):
        super(GCN, self).__init__()

        self.time_steps = time_steps
        self.pretrained = pretrained
        self.partialConv = partialConv
        self.use_intention = use_intention
        self.num_box = NUM_BOX  # TODO
        self.hidden_size = 512

        # build backbones
        if self.partialConv:
            self.backbone = pdresnet50(pretrained=self.pretrained)

        self.object_backbone = Riskbench_backbone(
            roi_align_kernel=8, n=self.num_box)

        # 2d conv after backbones and flatten the features
        self.camera_features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        if self.use_intention:
            self.state_features = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(inplace=True),
            )
            self.intention_features = nn.Sequential(
                nn.Linear(9, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        # specify feature size
        self.fusion_size = 512 + (256 if self.use_intention else 0)

        # temporal modeling
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        # gcn module
        self.emb_size = self.hidden_size
        self.fc_emb_1 = nn.Linear(
            self.hidden_size, self.emb_size, bias=False)
        self.fc_emb_2 = nn.Linear(self.emb_size * 2, 1)

        # classifier
        self.vel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def state_model(self, state_input):

        batch_size = state_input.shape[0]
        num_box = state_input.shape[1]
        state_input = state_input.reshape(batch_size*num_box, -1)

        state_feature = self.state_features(state_input)
        state_feature = state_feature.reshape(batch_size, num_box, -1)

        return state_feature

    def intention_model(self, intention_input):

        batch_size = intention_input.shape[0]
        num_box = intention_input.shape[1]

        intention_input = intention_input.reshape(batch_size*num_box, -1)

        # intention_feature = self.IntentionModel(ped, ctxt, ped_bbox, obj_bbox, obj_cls)
        intention_feature = self.intention_features(intention_input)
        intention_feature = intention_feature.reshape(batch_size, num_box, -1)

        return intention_feature

    def message_passing(self, input_feature, trackers, device=0):
        #########################################
        # input_feature:   (BxN) * 2000
        # trackers:         BxTxNx4
        # dist_mask:        BxN
        #############################################

        num_box = trackers.shape[2]+1
        B = len(trackers)

        mask = torch.ones((B, num_box))
        mask[:, 1:] = trackers[:, -1, :, 2]+trackers[:, -1, :, 3]
        mask = mask != 0  # (B, N, 1)
        
        # emb_feature: (BxN, H) -> (BxN, self.emb_size)
        emb_feature = self.fc_emb_1(input_feature)
        # emb_feature:  (B, N, self.emb_size)
        emb_feature = emb_feature.reshape(-1, num_box, self.emb_size)

        # ego_feature: (B, N , self.emb_size)
        ego_feature = emb_feature[:, 0,
                                  :].reshape(-1, 1, self.emb_size).repeat(1, num_box, 1)

        # emb_feature:(B, N, 2*self.emb_size)
        emb_feature = torch.cat((ego_feature, emb_feature), 2)
        # emb_feature:(BxN, 2*self.emb_size)
        emb_feature = emb_feature.reshape(-1, 2 * self.emb_size)


        # emb_feature: (B, N, 1)
        emb_feature = self.fc_emb_2(emb_feature).reshape(-1, num_box, 1)
        emb_feature[~(mask.byte().to(torch.bool))] = torch.tensor(
            [-float("Inf")]).cuda(device)

        # emb_feature: (B, N , 1)
        attn_weights = F.softmax(emb_feature, dim=1)
        # attn_weights: (BxN, 1)
        attn_weights = attn_weights.reshape(-1, 1)

        # ori_ego_feature : (B, H)
        ori_ego_feature = input_feature.reshape(-1,
                                             num_box, self.hidden_size)[:, 0, :]
        # input_feature : (BxN, H)
        input_feature = input_feature.reshape(-1, self.hidden_size)

        # fusion_feature : (B, N, H)
        fusion_feature = (
            input_feature * attn_weights).reshape(-1, num_box, self.hidden_size)
        # input_feature : (B, H)
        fusion_feature = torch.sum(fusion_feature, 1)
        # updated_fusion : (B, 2*H)
        fusion_feature = torch.cat((ori_ego_feature, fusion_feature), 1)

        return fusion_feature, attn_weights

    def step(self, camera_input, hx, cx):

        fusion_input = camera_input
        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))

        return hx, cx

    def forward(self, camera_inputs, trackers, device, mask=None,
                state_inputs=None, intention_inputs=None):

        ###########################################
        #  camera_input:    BxTxCxWxH
        #  tracker:         BxTxNx4
        ###########################################

        # Record input size
        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        c = camera_inputs.shape[2]
        h = camera_inputs.shape[3]
        w = camera_inputs.shape[4]

        # Define mask if mask does not exists
        if len(mask.size()) == 0:
            mask = torch.ones((batch_size, t, c, h, w)).to(device)

        # initialize LSTM
        hx = torch.zeros(
            (batch_size*(self.num_box+1), self.hidden_size)).to(device)
        cx = torch.zeros(
            (batch_size*(self.num_box+1), self.hidden_size)).to(device)

        """ ego feature"""
        # BxTxCxHxW -> (BT)xCxHxW
        camera_inputs = camera_inputs.reshape(-1, c, h, w)

        # (BT)x2048x8x20
        if self.partialConv:
            ego_features = self.backbone.features(camera_inputs, mask.reshape(-1, c, h, w))
        else:
            ego_features = self.backbone.features(camera_inputs)

        # Reshape the ego_features to LSTM
        c = ego_features.shape[1]
        h = ego_features.shape[2]
        w = ego_features.shape[3]

        # (BT)x2048x8x20 -> BxTx2048x8x20
        ego_features = ego_features.reshape(batch_size, t, c, h, w)

        """ object feature"""
        # BxTxNx4 -> (BT)xNx4
        tracker = trackers.reshape(-1, self.num_box, 4)

        # (BT)xNx512
        _, obj_features = self.object_backbone(camera_inputs, tracker)

        # BxTxNx512
        obj_features = obj_features.reshape(batch_size, t, self.num_box, -1)

        # Running LSTM
        for l in range(0, self.time_steps):

            # BxTx2048x8x20 -> Bx2048x8x20
            ego_feature = ego_features[:, l].clone()

            # BxTxNx512 -> BxNx512
            obj_feature = obj_features[:, l].clone()

            # Bx2048x8x20 -> Bx512x1x1 ->  Bx1x512
            ego_feature = self.camera_features(
                ego_feature).reshape(batch_size, 1, -1)

            # 1x(1+N)x512
            feature_input = torch.cat((ego_feature, obj_feature), 1)

            # intention_feature : BxNx128
            if self.use_intention:

                state_input = state_inputs[:, l]
                intention_input = intention_inputs[:, l]
                state_feature = self.state_model(state_input)
                intention_feature = self.intention_model(intention_input)
                feature_input = torch.cat(
                    (feature_input, intention_feature, state_feature), -1)
            
            # (1+N)x512
            feature_input = feature_input.reshape(-1, self.fusion_size)

            # LSTM
            hx, cx = self.step(feature_input, hx, cx)

        updated_feature, _ = self.message_passing(hx, trackers, device)

        vel = self.vel_classifier(self.drop(updated_feature))
        vel = self.sigmoid(vel).reshape(-1)

        return vel
