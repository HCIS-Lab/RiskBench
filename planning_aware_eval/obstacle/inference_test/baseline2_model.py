import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import enum
import os
import numpy as np
import fnmatch

__all__ = [
    'SA',
]

class custom_loss(nn.Module):
    def __init__(self, time,weights):
        super(custom_loss, self).__init__()

        self.cel =  nn.CrossEntropyLoss(weight = weights,reduction = 'none')
        self.time = time

    def forward(self, outputs, targets,device):
        # targets: b (True or false)
        # outputs: txbx2
        # targets = targets.long() # convert to 0 or 1
        # outputs = outputs.permute(1,0,2) #bxtx2
        loss = torch.tensor(0.0).to(device)
        for i,pred in enumerate(outputs):
            #bx2
            temp_loss = self.cel(pred,targets) # b
            exp_loss = torch.multiply(torch.exp(torch.tensor(-max(0,self.time-i-1) / 15.0)), temp_loss)
            exp_loss = torch.multiply(exp_loss,targets)
            loss = torch.add(loss, torch.mean(torch.add(temp_loss,exp_loss)))
        return loss



class Baseline_SA(nn.Module):
    def __init__(self, object_num,device,ablation,global_avg_pooling = True,n_frame=100, features_size=1024, frame_features_size=256, hidden_layer_size=256, lstm_size=512):
        super(Baseline_SA, self).__init__()
        self.n_frame = n_frame
        self.object_num = object_num
        self.global_avg_pooling = None
        self.device = device
        self.ablation = ablation
        self.features_size = features_size
        self.frame_features_size = frame_features_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layer_size = lstm_size
        self.frame_layer = nn.Linear(self.frame_features_size, self.hidden_layer_size)
        
        if self.features_size>1024:
            if ablation==1:
                self.object_layer = nn.Sequential(
                    nn.Linear(self.features_size, self.hidden_layer_size),
                    nn.ReLU(inplace=True),                                                 
                )
            elif ablation==0:
                self.object_layer = nn.Sequential(
                    nn.Linear(self.features_size, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True), 
                    # nn.Dropout(p=0.4),
                    nn.Linear(1024, self.hidden_layer_size),
                    nn.ReLU(inplace=True),                                                 
                )
            elif ablation==2:
                self.object_layer = nn.Sequential(
                    nn.Linear(64*7*7, self.hidden_layer_size),
                    nn.ReLU(inplace=True),                                               
                )
                self.conv = nn.Sequential(
                    nn.Conv2d(256,64,1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
            elif ablation==3:
                self.object_layer = nn.Sequential(
                    nn.Linear(256, self.hidden_layer_size),
                    nn.ReLU(inplace=True),                                               
                )
        else:
             self.object_layer = nn.Sequential(
                nn.Linear(self.features_size, self.hidden_layer_size),
                nn.ReLU(inplace=True), 
                # nn.Dropout(p=0.3),
            )
        if global_avg_pooling:
            self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        if ablation<2:
            self.bn1 = nn.BatchNorm1d(self.features_size)
        
        self.fusion_size = 2*self.hidden_layer_size #if self.with_frame else self.hidden_layer_size
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.lstm_layer_size)
        self.output_layer = nn.Linear(self.lstm_layer_size, 2)
        self.att_w = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,1)),requires_grad = True)#.to('cuda')
        self.att_wa = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.lstm_layer_size,self.hidden_layer_size)),requires_grad = True)#.to('cuda')
        self.att_ua = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,self.hidden_layer_size)),requires_grad = True)#.to('cuda')
        self.att_ba = nn.Parameter(torch.zeros(self.hidden_layer_size),requires_grad = True)#.to('cuda')

    def step(self, fusion, hx, cx):

        hx, cx = self.lstm(self.drop(fusion), (hx, cx))

        return self.output_layer(hx), hx, cx,#self.risky_object(hx)

    def attention_layer(self, object, h_prev):
        brcst_w = torch.tile(torch.unsqueeze(self.att_w, 0), (self.object_num,1,1)) # n x h x 1
        image_part = torch.matmul(object, torch.tile(torch.unsqueeze(self.att_ua, 0), (self.object_num,1,1))) + self.att_ba # n x b x h
        e = nn.functional.relu(torch.matmul(h_prev,self.att_wa)+image_part) # n x b x h
        return brcst_w, e

    def normalization(self, input_features):
        # bxtx20x12544
        b,t,n,c = input_features.size()
        input_features = input_features.view(-1,c)
        input_features = self.bn1(input_features)
        input_features = input_features.view(b,t,n,c)
        return input_features

    # def initialize_device(self):
    #     self.att_w = self.att_w.to(self.device)
    #     self.att_wa = self.att_wa.to(self.device)
    #     self.att_ua = self.att_ua.to(self.device)
    #     self.att_ba = self.att_ba.to(self.device)

    def forward(self, input_features, input_frame):
        # self.initialize_device()
        # features: b,t,20,C  (batch, frame, n(obj), C)
        # input_frame -> global avg pooling
        batch_size = input_features.size()[0]  
        input_features = input_features.view(batch_size,self.n_frame,self.object_num,-1)
        hx = torch.zeros((batch_size, self.lstm_layer_size)).to(self.device)
        cx = torch.zeros((batch_size, self.lstm_layer_size)).to(self.device)
        out = []
        input_frame = self.global_avg_pooling(input_frame)
        zeros_object =  torch.sum(input_features.permute(1,2,0,3),3).eq(0) # t x n x b
        zeros_object = ~zeros_object 
        zeros_object = zeros_object.float().contiguous()
        input_features = self.normalization(input_features)
        for i in range(self.n_frame):
            img = input_frame[:,i].view(-1,self.frame_features_size)
            img = self.frame_layer(img)
            object = input_features[:,i].permute(1,0,2).contiguous() # nxbxc
            object = object.view(-1, self.features_size).contiguous() # (nxb)xc
            object = self.object_layer(object) #(nxb)xh
            object = object.view(self.object_num,batch_size,self.hidden_layer_size)
            # object = torch.matmul(object,torch.unsqueeze(zeros_object[i],2))
            object = object*torch.unsqueeze(zeros_object[i],2)
            
            brcst_w,e = self.attention_layer(object,hx)
            # alphas = nn.functional.softmax(torch.mul(torch.sum(torch.matmul(e,brcst_w),2),zeros_object[i]),0)
            alphas = torch.mul(nn.functional.softmax(torch.sum(torch.matmul(e,brcst_w),2),0),zeros_object[i]) # n x b
            attention_list = torch.mul(torch.unsqueeze(alphas,2),object) # n x b x h
            attention = torch.sum(attention_list,0) # b x h
            # concat frame & object
            fusion = torch.cat((img,attention),1)
            pred,hx,cx = self.step(fusion,hx,cx)
            out.append(pred)

            if i == 0:
                soft_pred = nn.functional.softmax(pred,dim=1)
                all_alphas = torch.unsqueeze(alphas,0)
            else:
                temp_soft_pred = nn.functional.softmax(pred,dim=1)
                soft_pred = torch.cat([soft_pred,temp_soft_pred],1)
                temp_alphas = torch.unsqueeze(alphas,0)
                all_alphas = torch.cat([all_alphas, temp_alphas],0)

        out_stack = torch.stack(out)
        soft_pred = soft_pred.view(batch_size,self.n_frame,-1)
        all_alphas = all_alphas.permute(2,0,1)

        return soft_pred, all_alphas, out_stack

    