import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset.common import get_dataset
from models.backbone import Riskbench_backbone
from models.DSA_RRL import Baseline_SA, custom_loss
from common import get_parser, get_one_hot

import wandb
log_every_steps = 10


def get_logs_path(args):
    PATH = time.localtime(time.time())[1:5]
    PATH = '_'.join(str(i) for i in PATH)
    PATH = os.path.join('logs',PATH)
    os.mkdir(PATH)
    config = vars(args)
    with open(os.path.join(PATH,"config.json"), "w") as outfile:
        json.dump(config, outfile)
    return PATH

class Collision_Metrics:
    def __init__(self,thresholds=[0.4,0.5,0.6]):
        self.thresholds = torch.tensor(thresholds)
        self.reset()

    def reset(self):
        self.tp = torch.zeros_like(self.thresholds,dtype=float)
        self.fp = torch.zeros_like(self.thresholds,dtype=float)
        self.tn = torch.zeros_like(self.thresholds,dtype=float)
        self.fn = torch.zeros_like(self.thresholds,dtype=float)

    def set_device(self,device):
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.tn = self.tn.to(device)
        self.fn = self.fn.to(device)
        self.thresholds = self.thresholds.to(device)

    def update(self,pred,label):
        pred = pred.softmax(dim=1)
        device = pred.device
        self.set_device(device)
        pred = (pred[:,None,1] > self.thresholds).bool()
        label = label[:,1].bool()
        self.tp += (pred & label[:,None]).sum(0)
        self.fp += (pred & ~label[:,None]).sum(0)
        self.tn += (~pred & ~label[:,None]).sum(0)
        self.fn += (~pred & label[:,None]).sum(0)
        
    def calculate(self):
        acc = (self.tp+self.tn)/(self.tp+self.fp+self.tn+self.fn)
        precision = (self.tp)/(self.tp+self.fp)
        recall = (self.tp)/(self.tp+self.fn)
        # F1 = 
        result = {}
        for i, threshold in enumerate(self.thresholds):
            s = threshold.cpu().numpy()
            s = str(np.round(s,1))
            result[f"Accuracy/{s}"] = acc[i]
            result[f"Recall/{s}"] = recall[i]
            result[f"Precision/{s}"] = precision[i]
        return result

class Collision_trainer:
    def __init__(self,args,dataset_setting,model,criterion,optimizer,critical_frame,cuda,intention=False,state=False):
        assert not intention & state
        self.metrics = Collision_Metrics()
        self.critical_frame = critical_frame
        self.cuda = cuda
        self.intention = intention
        self.state = state
        if cuda:
            criterion = criterion.cuda()
            model = model.cuda()
        self.criterion = criterion
        self.model = model
        
        train_dataset = get_dataset(args.root,dataset_setting,mode='train')
        val_dataset = get_dataset(args.root,dataset_setting,mode='val')
        self.trainset = DataLoader(train_dataset,batch_size=args.batch, shuffle=True, num_workers=5,pin_memory=True)
        self.valset = DataLoader(val_dataset,batch_size=args.batch, shuffle=True, num_workers=5,pin_memory=True)

        self.optimizer = optimizer
        self.log_path = get_logs_path(args)

        print(f"Trainset length: {len(train_dataset)}, Valset length: {len(val_dataset)}")

    def training(self,epoch):
        min_loss = 10000
        for i in range(epoch):
            print(f"Epoch {i+1}/{epoch}")
            train_bar = tqdm(self.trainset,leave=False,desc="Train")
            for j,batch in enumerate(train_bar):
                loss_dict, _ = self.step(batch,mode='train')
                loss = loss_dict['total_loss'].item()
                train_bar.set_postfix({'loss': float(loss)})
                # print(f"\rLoss: {float(loss.item())}",end='\r')
                if j % log_every_steps == log_every_steps-1:
                    for k,v in loss_dict.items():
                        wandb.log({f'train/loss/{k}': v.item()})
            self.log_metric('train')

            val_loss = 0.0
            val_bar = tqdm(self.valset,leave=False,desc="Val")
            for j,batch in enumerate(val_bar):
                loss_dict, all_alphas = self.step(batch,mode='val')
                loss = loss_dict['total_loss'].item()
                val_loss += loss
                val_bar.set_postfix({'loss': float(loss)})
                # print(f"\rLoss: {float(loss)}",end='\r')

                if j % log_every_steps == log_every_steps-1:
                    for k,v in loss_dict.items():
                        wandb.log({f'val/loss/{k}': v.item()})
            
            if val_loss/(j+1) < min_loss:
                print("\n=============Saving best model=============")
                min_loss = val_loss
                torch.save(self.model.state_dict(), self.log_path+'/best_model.pt')

            self.log_metric('val')
    def step(self,batch,mode):
        """
            run model, loss, metrics
        """
        all_alphas = None
        intention = None
        state = None
        img, box, label, obj_targets = batch['img'],batch['bbox'], batch['label'], batch['risky_id']
        if self.intention:
            intention = get_one_hot(batch['s_type'],batch['s_id'])
            intention = torch.from_numpy(np.array(intention).astype(np.float32))
        elif self.state:
            state = batch['state']
        if self.cuda:
            img = img.cuda()
            box = box.cuda()
            label = label.cuda()
            if self.intention:
                intention = intention.cuda()
            elif self.state:
                state = state.cuda()

        if mode == 'train':
            self.model = self.model.train()
            pred, _, obj_pred = self.model(img, box,intention,state)
            if self.model.supervised:
                if self.cuda:
                    obj_targets = obj_targets.cuda()
                loss_dict = self.criterion(pred,label,obj_pred,obj_targets)
            else:
                loss_dict = self.criterion(pred,label)
            loss = loss_dict['total_loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif mode == 'val':
            self.model = self.model.eval()
            with torch.no_grad():
                pred, _, obj_pred = self.model(img, box,intention,state)
                if self.cuda:
                    obj_targets = obj_targets.cuda()
                if self.model.supervised:
                    loss_dict = self.criterion(pred,label,obj_pred,obj_targets)
                else:
                    loss_dict = self.criterion(pred,label)

        self.metrics.update(pred[:,self.critical_frame],label)
        return loss_dict, all_alphas

    def log_metric(self,mode):
        metric_result = self.metrics.calculate()
        result = {}
        for k,v in metric_result.items():
            result[f"{mode}/metrics/{k}"] = v
        wandb.log(result)
        self.metrics.reset()

if __name__ == '__main__':
    wandb.init(project='RiskBench', entity='tomy45651')

    args = get_parser()
    n_frame = 40
    object_num = 20
    setting = {"object_num":object_num,"frame_num":n_frame,"load_img_first":args.load_first}
    intention = args.intention
    supervised = args.supervised
    state = args.state
    backbone = Riskbench_backbone(8,object_num,intention=intention)
    SA = Baseline_SA(backbone,n_frame,object_num,intention=intention,supervised=supervised,state=state)
    criterion = custom_loss(int(n_frame*0.8),[0.5,1],supervised=supervised)
    optimizer = torch.optim.AdamW(SA.parameters(),lr=args.lr,weight_decay=args.wd)
    trainer = Collision_trainer(args,setting,SA,criterion,optimizer,int(n_frame*0.8),True,intention=intention,state=state)
    trainer.training(args.epoch)