import torch
import torch.nn as nn
import numpy as np
import torchvision

class collision_loss(nn.Module):
    def __init__(self, time, weights):
        super(collision_loss, self).__init__()
        self.cel =  nn.CrossEntropyLoss(weight=weights, reduction = 'none')
        self.time = time

    def forward(self, outputs, targets, device):
        # outputs: txBx2, targets: B
        loss = torch.tensor(0.0).to(device)
        for i, pred in enumerate(outputs):
            temp_loss = self.cel(pred, targets) # B
            exp_loss = torch.multiply(torch.exp(torch.tensor(-max(0, self.time-i-1) / 12.0)), temp_loss)
            exp_loss = torch.multiply(exp_loss, targets)
            loss = torch.add(loss, torch.mean(torch.add(temp_loss, exp_loss)))
        return loss

class risk_obj_loss(nn.Module):
    def __init__(self, weights):
        super(risk_obj_loss, self).__init__()
        self.weight = weights
        # self.CEloss =  nn.CrossEntropyLoss(weight=weights, reduction='sum')
    
    def forward(self, outputs, targets, device):
        # self.weight = self.weight.to(device)
        loss = torch.tensor(0.0).to(device)
        batch = targets.shape[0]

        #outputs: Bxtxn, targets: Bxn
        for out, tar in zip(outputs, targets):# for each batch
            for pred in out:# for each frame
                # pred: n, tar: n
                temp_loss = self.weight[1] * (tar*torch.log(pred+1e-5)) + self.weight[0] * ((1-tar)*torch.log(1-pred+1e-5))
                temp_loss = torch.neg(torch.mean(temp_loss))
                loss += temp_loss
        loss /= batch
        # print(loss)
        return loss

class Supervised(nn.Module):
    def __init__(self, device, n_obj=40, n_frame=60, features_size=1024, frame_features_size=256, hidden_layer_size=256, lstm_size=256):
        super(Supervised, self).__init__()
        self.device = device
        self.n_frame = n_frame
        self.n_obj = n_obj

        self.features_size = features_size
        self.frame_features_size = frame_features_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layer_size = lstm_size

        self.frame_pre_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.frame_layer = nn.Linear(self.frame_features_size, self.hidden_layer_size)
        self.object_layer = nn.Sequential(
            nn.Linear(self.features_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.hidden_layer_size),
            nn.ReLU(inplace=True),                                                 
        )

        self.fusion_size = self.hidden_layer_size#2 * self.hidden_layer_size
        self.bn1 = nn.BatchNorm1d(self.features_size)
        self.drop = nn.Dropout(p=0.5)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU(inplace=True)

        self.lstm = nn.LSTMCell(self.fusion_size, self.lstm_layer_size)
        self.collision_output = nn.Linear(self.lstm_layer_size, 2)
        self.risk_output = nn.Linear(self.hidden_layer_size, 1)

        self.att_w = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,1)), requires_grad = True)#.to('cuda')
        self.att_wa = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.lstm_layer_size,self.hidden_layer_size)), requires_grad = True)#.to('cuda')
        self.att_ua = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,self.hidden_layer_size)), requires_grad = True)#.to('cuda')
        self.att_ba = nn.Parameter(torch.zeros(self.hidden_layer_size), requires_grad = True)#.to('cuda')

    def step(self, fusion, hx, cx):
        hx, cx = self.lstm(self.drop(fusion), (hx, cx))
        return self.collision_output(hx), hx, cx

    def attention_layer(self, object, h_prev):
        brcst_w = torch.tile(torch.unsqueeze(self.att_w, 0), (self.n_obj, 1, 1)) # n x h x 1
        image_part = torch.matmul(object, torch.tile(torch.unsqueeze(self.att_ua, 0), (self.n_obj, 1, 1))) + self.att_ba # n x b x h
        e = torch.tanh(torch.matmul(h_prev, self.att_wa) + image_part) # n x b x h
        return brcst_w, e

    def normalization(self, input_features):
        # bxtxnx12544
        b, t, n, c = input_features.size()
        input_features = input_features.view(-1, c)
        input_features = self.bn1(input_features)
        input_features = input_features.view(b, t, n, c)
        return input_features

    def forward(self, input_features, input_frame):
        # features: b,t,40,C  (batch, frame, n(obj), C)
        batch_size = input_features.size()[0]
        input_features = input_features.view(batch_size, self.n_frame, self.n_obj, -1)
        
        hx = torch.zeros((batch_size, self.lstm_layer_size)).to(self.device)
        cx = torch.zeros((batch_size, self.lstm_layer_size)).to(self.device)
        zeros_object =  torch.sum(input_features.permute(1, 2, 0, 3), 3).eq(0) # t x n x b
        zeros_object = ~zeros_object 
        zeros_object = zeros_object.float().contiguous()
        input_features = self.normalization(input_features)

        out_c = []      #collision
        soft_pred_c = []
        out_r = []      #risk
        sig_pred_r = []

        for i in range(self.n_frame):
            full_frame = self.frame_pre_layer(input_frame[:, i])
            full_frame = self.global_avg_pooling(full_frame)
            full_frame = full_frame.view(batch_size, self.frame_features_size)
            full_frame = self.frame_layer(full_frame)#bxh

            object = input_features[:, i].permute(1, 0, 2).contiguous() # nxbxc
            object = object.view(-1, self.features_size).contiguous() # (nxb)xc
            object = self.object_layer(object) #(nxb)xh
            object = object.view(self.n_obj, batch_size, self.hidden_layer_size)
            object = object * torch.unsqueeze(zeros_object[i], 2)#nxbxh

            for n in range(self.n_obj):
                object[n].add(full_frame)
            object = self.relu(object)

            ### Collision Prediction Branch
            brcst_w, e = self.attention_layer(object, hx)
            alphas = torch.mul(nn.functional.softmax(torch.sum(torch.matmul(e, brcst_w), 2), 0), zeros_object[i])#nxb
            attention_list = torch.mul(torch.unsqueeze(alphas, 2), object)#nxbxh 
            attention = torch.sum(attention_list, 0) # bxh

            # fusion = torch.cat((img, attention), 1)
            pred_c, hx, cx = self.step(attention, hx, cx)#bx2

            out_c.append(pred_c)
            soft_c = nn.functional.softmax(pred_c, dim=1)#bx2
            soft_pred_c.append(soft_c)

            ### Risk Object Prediction Branch
            attention_list = attention_list.permute(1, 0, 2)
            risk_score = self.risk_output(attention_list)#bxnxh -> bxnx1
            risk_score = risk_score.squeeze(dim=2)#bxn

            out_r.append(risk_score)
            pred_r = torch.sigmoid(risk_score)#bxn
            sig_pred_r.append(pred_r)

        logits_c = torch.stack(out_c)
        soft_pred_c = torch.stack(soft_pred_c)
        soft_pred_c = soft_pred_c.permute(1, 0, 2)#txbx2 ->bxtx2

        score_r = torch.stack(out_r)#txbxn
        sig_pred_r = torch.stack(sig_pred_r)
        sig_pred_r = sig_pred_r.permute(1, 0, 2)#txbxn->bxtxn

        ## soft_pred_c: softmax prediction of score_c (bxtx2)
        ## score_c: each frame's collision score(logits) (txbx2)  for loss
        ## sig_pred_r: sigmoid prediction of score_r (bxtxn)
        ## score_r: each frame's objects' risk score (txbxn)  for loss
        return soft_pred_c, logits_c, sig_pred_r, score_r