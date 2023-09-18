import torch
import torch.nn as nn
from .backbone import Riskbench_backbone
from .pdresnet50 import pdresnet50
import torch.nn.functional as F

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
            roi_align_kernel=8, n=self.num_box, pretrained=pretrained)

        # 2d conv after backbone
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
            # self.intention_features = nn.Sequential(
            #     nn.Linear(10, 64),
            #     nn.BatchNorm1d(64),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(64, 256),
            #     nn.BatchNorm1d(256),
            #     nn.ReLU(inplace=True),
            # )

        # temporal modeling
        self.fusion_size = 512 + (128 if self.use_intention else 0)
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        # gcn module
        self.emb_size = self.hidden_size
        self.fc_emb_1 = nn.Linear(
            self.hidden_size, self.emb_size, bias=False)
        self.fc_emb_2 = nn.Linear(self.emb_size * 2, 1)

        # final classifier
        self.vel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def state_model(self, state_input):
        
        # Bx(1+N)x2 -> (B(1+N))x2 -> (B(1+N))x128 -> Bx(1+N)x128
        batch_size = state_input.shape[0]
        num_box = state_input.shape[1]

        state_input = state_input.reshape(batch_size*num_box, -1)

        state_feature = self.state_features(state_input)
        state_feature = state_feature.reshape(batch_size, num_box, -1)

        return state_feature

    def intention_model(self, intention_input):

        batch_size = intention_input.shape[0]

        # Bx10 -> Bx256 -> Bx1x256 -> Bx(1+N)x256
        intention_feature = self.intention_features(intention_input)
        intention_feature = intention_feature.reshape(batch_size, 1, -1)
        intention_feature = intention_feature.repeat(1, self.num_box+1, 1)

        return intention_feature

    def message_passing(self, input_feature, trackers, device=0):
        #############################################
        # input_feature:    (B(1+N))xH
        # trackers:         BxTxNx4
        #############################################

        num_box = trackers.shape[2]+1
        B = len(trackers)

        mask = torch.ones((B, num_box))
        mask[:, 1:] = trackers[:, -1, :, 2]+trackers[:, -1, :, 3]
        mask = mask != 0  # (B, N, 1)

        # (B(1+N))xH -> (B(1+N))xself.emb_size
        emb_feature = self.fc_emb_1(input_feature)
        # (B(1+N))xself.emb_size ->  Bx(1+N)xself.emb_size
        emb_feature = emb_feature.reshape(-1, num_box, self.emb_size)

        # Bx(1+N)xself.emb_size
        ego_feature = emb_feature[:, 0,
                                  :].reshape(-1, 1, self.emb_size).repeat(1, num_box, 1)

        # Bx(1+N)x(2*self.emb_size)
        emb_feature = torch.cat((ego_feature, emb_feature), 2)
        # (B(1+N))x(2*self.emb_size)
        emb_feature = emb_feature.reshape(-1, 2 * self.emb_size)

        # Bx(1+N)x1
        emb_feature = self.fc_emb_2(emb_feature).reshape(-1, num_box, 1)
        emb_feature[~(mask.byte().to(torch.bool))] = torch.tensor(
            [-float("Inf")]).to(device)

        # Bx(1+N)x1
        attn_weights = F.softmax(emb_feature, dim=1)
        # (B(1+N))x1
        attn_weights = attn_weights.reshape(-1, 1)

        # BxH
        ori_ego_feature = input_feature.reshape(-1,
                                                num_box, self.hidden_size)[:, 0, :]
        # (B(1+N))xH
        input_feature = input_feature.reshape(-1, self.hidden_size)

        # Bx(1+N)xH
        fusion_feature = (
            input_feature * attn_weights).reshape(-1, num_box, self.hidden_size)
        # BxH
        fusion_feature = torch.sum(fusion_feature, 1)
        # Bx(2*H)
        fusion_feature = torch.cat((ori_ego_feature, fusion_feature), 1)

        return fusion_feature, attn_weights

    def step(self, camera_input, hx, cx):

        fusion_input = camera_input
        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))

        return hx, cx

    def forward(self, camera_inputs, trackers, mask=None, intention_inputs=None, state_inputs=None, device='cuda'):

        ###########################################
        #  camera_input     :   BxTxCxWxH
        #  tracker          :   BxTxNx4
        #  intention_inputs :   Bx10
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
            (batch_size*(1+self.num_box), self.hidden_size)).to(device)
        cx = torch.zeros(
            (batch_size*(1+self.num_box), self.hidden_size)).to(device)

        """ ego feature"""
        # BxTxCxHxW -> (BT)xCxHxW
        camera_inputs = camera_inputs.reshape(-1, c, h, w)

        # (BT)x2048x8x20
        if self.partialConv:
            ego_features = self.backbone.features(
                camera_inputs, mask.reshape(-1, c, h, w))
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

            # Bx(1+N)x512
            feature_input = torch.cat((ego_feature, obj_feature), 1)

            if self.use_intention:

                # BxTx(1+N)x2 -> Bx(1+N)x2 
                state_input = state_inputs[:, l]
                # Bx(1+N)x2  -> Bx(1+N)x128
                state_feature = self.state_model(state_input)
                # Bx(1+N)x512 -> Bx(1+N)x(512+128)
                feature_input = torch.cat(
                    (feature_input, state_feature), -1)

                # # Bx(1+N)x256
                # intention_feature = self.intention_model(intention_inputs)
                # # Bx(1+N)x512 -> Bx(1+N)x(512+256)
                # feature_input = torch.cat(
                #     (feature_input, intention_feature), -1)

            # Bx(1+N)x512 -> (B(1+N))x512
            feature_input = feature_input.reshape(-1, self.fusion_size)

            # LSTM
            hx, cx = self.step(feature_input, hx, cx)

        updated_feature, _ = self.message_passing(hx, trackers, device)

        vel = self.vel_classifier(self.drop(updated_feature))
        vel = self.sigmoid(vel).reshape(-1)

        return vel
