from cgi import test
import pathlib
from turtle import position
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import json
import argparse
import GT_loader

class CarlaDataset(Dataset):
    def __init__(self, root_local,root_nas,mode, baseline,time,tracking=None,gt_box=True):
        """
            root: features path
            mode: training, testing or validating
            baseline: int, 1,2 or 3.
        """
        self.gt_box = gt_box
        self.mode = mode
        self.baseline = baseline
        self.time = time
        self.root_local = root_local
        self.root_nas = root_nas
        self.tracking = tracking
        self.datas_path = []
        self.labels = []
        total = 0
        positive = 0
        wrong_count = []
        less = []
        not_in_front = []
        for s_type in os.listdir(root_local):
            if s_type == 'obstacle' and mode == "training":
                continue
            # if s_type =='non-interactive' and (mode == "validating"or mode == "testing"):
            #     continue
            label = True if s_type == 'collision' else False
            for s_id in os.listdir(os.path.join(root_local,s_type)):
                map_name = int(s_id.split('_')[0])
                if mode == "training" and (map_name == 10 or map_name==5):
                    continue
                if mode == "testing" and map_name != 10:
                    continue
                if mode == "validating" and map_name != 5:
                    continue
                for variant in os.listdir(os.path.join(root_local,s_type,s_id,'variant_scenario')):
                    if s_type!= "non-interactive":
                        a, _, _ = GT_loader.getGTframe(s_type,s_id,variant)
                        if a is None:
                            less.append(s_type+'_'+s_id+'_'+variant)
                            continue
                    nas_path = os.path.join(root_nas,s_type,s_id,'variant_scenario',variant)
                    local_path = os.path.join(root_local,s_type,s_id,'variant_scenario',variant)
                    if os.path.exists(os.path.join(local_path,'wrong_flag')):
                        wrong_count.append(s_type+'_'+s_id+'_'+variant)
                        continue
                    if not(self.mode == "validating" or self.mode == "testing"):
                        if os.path.exists(os.path.join(local_path,'not_in_front')):
                            not_in_front.append(os.path.join(s_type,s_id,'variant_scenario',variant))
                            continue
                    if label and mode == "training":
                        history = open(os.path.join(nas_path,'collision_history.json'))
                        collision_frame = json.load(history)[0]['frame']
                        history.close()
                        features = sorted(os.listdir(os.path.join(local_path,'features','rgb','front')))
                        first = int(features[0])
                        end = int(features[-1])
                        if collision_frame-int(time*0.8)<first or collision_frame+int(time*0.2)>end:
                            continue
                        positive += 1
                    elif label:
                        positive += 1
                    total += 1
                    self.datas_path.append(os.path.join(s_type,s_id,'variant_scenario',variant))
                    self.labels.append(label)
        print(not_in_front)
        print("-------")
        print(wrong_count)
        print("-------")
        print(less)
        print("-------")
        print(mode,':')
        print('\tTotal data: %d\n\tPositive data: %d\n\tRatio: %f'%(total,positive,float(positive)/float(total)))
            
    def __getitem__(self, index):
        # list frame_n, 81
        # nn.linear(xxx,81)
        """
             baseline1 read full_frame(p5 or res5), collision label
             baseline2 read full_frame, roi, bbox, collision label
             baseline3 read full_frame, roi, bbox, collision label, risky object label
        """
        risky_object = -1
        collision_frame = None
        collision_label = self.labels[index]
        nas_path = os.path.join(self.root_nas,self.datas_path[index])
        local_path = os.path.join(self.root_local,self.datas_path[index])
        if self.tracking is not None:
            tracker = open(os.path.join(local_path,'tracker.json'))
            tracker_data = json.load(tracker)
            tracker.close()
        if collision_label:
            history = open(os.path.join(nas_path,'collision_history.json'))
            collision_data = json.load(history)[0]
            collision_frame = collision_data['frame']
            if self.tracking is not None:
                collision_id = collision_data['actor_id'] & 0xffff
                if not(self.mode == "validating" or self.mode == "testing"):
                    risky_object = tracker_data[str(collision_id)]
            history.close()
        features_path = sorted(os.listdir(os.path.join(local_path,'features','rgb','front')))
        first_frame = int(features_path[0])
        if self.mode == "validating" or self.mode == "testing":
            temp = self.datas_path[index].split('/')
            if temp[0] == "non-interactive":
                GT_frame = None
            else:
                _, GT_frame, _ = GT_loader.getGTframe(temp[0],temp[1],temp[3])
        if collision_frame is not None:
            if self.mode == "validating" or self.mode == "testing":
                ###
                if GT_frame-first_frame>=self.time:
                    features_path = features_path[GT_frame-first_frame-self.time:GT_frame-first_frame]
                else:
                    features_path = features_path[:self.time]
                ###
                # features_path = features_path[:GT_frame-first_frame]
            else:
                features_path = features_path[collision_frame-first_frame-int(self.time*0.8):collision_frame-first_frame+int(self.time*0.2)]
            
        else:
            if self.mode == "validating" or self.mode == "testing":
                ###
                features_path = features_path[:self.time]
                ###
                # if temp[0] != "non-interactive":
                #     features_path = features_path[:GT_frame-first_frame]
        # else:
            # if self.mode == "validating" or self.mode == "testing":
            #     # rand_choice = np.random.randint(0,len(features_path)-self.time)
            #     # features_path = features_path[rand_choice:rand_choice+self.time]

            #     # batch = 1
                
            #     # if GT_frame is not None:
            #     #     if GT_frame-first_frame>=self.time:
            #     #         features_path = features_path[GT_frame-first_frame-self.time:GT_frame-first_frame]
            #     #     else:
            #     #         features_path = features_path[:self.time]
            #     # else:
            #     #     features_path = features_path[:self.time]
            #     if GT_frame is None:
            #         features_path = features_path[:self.time]

            # else:
            # features_path = features_path[:self.time]

            
        # return torch.tensor([1]), torch.tensor([1]), collision_label, self.datas_path[index]
        # collect frame_features
        frame_features = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'frame.pt')).unsqueeze(0)
        for temp_path in features_path[1:]:
            temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'frame.pt')).unsqueeze(0)
            frame_features = torch.cat((frame_features,temp),dim=0)
        
        if self.baseline >= 2:
            if self.gt_box:
                if self.tracking is None:
                    temp = torch.zeros((20,256,7,7))
                    roi = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'object.pt'))
                    object_num = 20 if roi.shape[0]>20 else roi.shape[0]
                    temp[:object_num] = roi[:object_num]
                    roi = temp.unsqueeze(0)
                    for temp_path in features_path[1:]:
                        temp = torch.zeros((20,256,7,7))
                        roi_temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'object.pt'))
                        object_num = 20 if roi_temp.shape[0]>20 else roi_temp.shape[0]
                        temp[:object_num] = roi_temp[:object_num]
                        roi = torch.cat((roi,temp.unsqueeze(0)),dim=0)
                else:
                    temp = torch.zeros((self.tracking,256,7,7))
                    roi = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'object.pt'))
                    box_file = open(os.path.join(nas_path,'bbox','front',features_path[0]+'.json'))
                    boxes = json.load(box_file)
                    box_file.close()
                    for i,box in enumerate(boxes):
                        if tracker_data[str(box['actor_id'])]>=self.tracking:
                            continue
                        temp[tracker_data[str(box['actor_id'])]] = roi[i]
                    roi = temp.unsqueeze(0)
                    for temp_path in features_path[1:]:
                        temp = torch.zeros((self.tracking,256,7,7))
                        roi_temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'object.pt'))
                        box_file = open(os.path.join(nas_path,'bbox','front',temp_path+'.json'))
                        boxes = json.load(box_file)
                        box_file.close()
                        for i,box in enumerate(boxes):
                            if tracker_data[str(box['actor_id'])]>=self.tracking:
                                continue
                            temp[tracker_data[str(box['actor_id'])]] = roi_temp[i]
                        roi = torch.cat((roi,temp.unsqueeze(0)),dim=0)

        if self.baseline == 1:
            return frame_features, collision_label, self.datas_path[index]
        elif self.baseline == 2:
            return frame_features, roi, collision_label, self.datas_path[index]
        elif self.baseline == 3:
            #risky_object += 1
            risk_label = np.zeros(self.tracking)
            if risky_object > -1:  #-1 means no collision
                risk_label[risky_object] = 1.0
            return frame_features, roi, collision_label, risk_label, self.datas_path[index]
        
    def __len__(self):
       return len(self.labels)

def get_parser():
    parser = argparse.ArgumentParser(description="Baseline 2")
    parser.add_argument(
        '-m',
        '--mode',
        required=True,
        help="training, testing or demo",
        type=str
    )
    parser.add_argument(
        '--model',
        help="which model to use",
        type=str
    )
    parser.add_argument(
        '--localpath',
        required=True,
        help="local(server) path",
        type=str
    )
    parser.add_argument(
        '--naspath',
        required=True,
        help="nas path",
        type=str
    )
    parser.add_argument(
        '--batch',
        default=10,
        help="batch size",
        type=int
    )
    parser.add_argument(
        '--lr',
        default=0.0001,
        help="learning rate",
        type=float
    )
    parser.add_argument(
        '--seed',
        default=1,
        help="random seed",
        type=int
    )
    parser.add_argument(
        '--epoch',
        default=16,
        help="number of epochs",
        type=int
    )
    parser.add_argument(
        '--time',
        default=60,
        help="video time",
        type=int
    )
    return parser

def get_dataset_loader(baseline,local_path,nas_path,tracking,batch_size,collate,clip_time,random_seed=12,validation=False):
    if not validation:
        trainset = CarlaDataset(local_path,nas_path,'training',baseline,clip_time,tracking=tracking)
    testset = CarlaDataset(local_path,nas_path,'testing',baseline,clip_time,tracking=tracking)
    validateset = CarlaDataset(local_path,nas_path,'validating',baseline,clip_time,tracking=tracking)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    trainloader = None
    testloader = None
    if not validation:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=5, 
                                                collate_fn=collate, drop_last =True, shuffle=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=5,
                                            collate_fn=collate, drop_last =True, shuffle=True,  pin_memory=True)

    validationloader = torch.utils.data.DataLoader(validateset, batch_size=batch_size, num_workers=5,
                                            collate_fn=collate, drop_last =True, shuffle=True, pin_memory=True)                                      
    return trainloader,testloader,validationloader

def calculate_matric(pred,labels):
    correct = float((pred==labels).sum().float())
    # TP_FP = labels[labels==1].shape[0]
    TP_FP = pred[pred==1].shape[0]
    TP = float((pred[labels==1]==labels[labels==1]).sum().float())
    FP = TP_FP - TP
    TN = correct - TP
    FN = (int(labels.shape[0])-correct)-FP
    return TP,FP,TN,FN


def get_specific_data(baseline,time,root_local,root_nas,tracking,s_type,s_id,variant):

    risky_object = -1
    collision_frame = None
    collision_label = True if s_type=="collision" else False
    nas_path = os.path.join(root_nas,s_type,s_id,'variant_scenario',variant)
    local_path = os.path.join(root_local,s_type,s_id,'variant_scenario',variant)
    if tracking is not None:
        tracker = open(os.path.join(local_path,'tracker.json'))
        tracker_data = json.load(tracker)
        tracker.close()
    if collision_label:
        history = open(os.path.join(nas_path,'collision_history.json'))
        collision_data = json.load(history)[0]
        collision_frame = collision_data['frame']
        if tracking is not None:
            collision_id = collision_data['actor_id'] & 0xffff
            risky_object = tracker_data[str(collision_id)]
        history.close()
    features_path = sorted(os.listdir(os.path.join(local_path,'features','rgb','front')))
    first_frame = int(features_path[0])
    if collision_frame is not None:
        features_path = features_path[collision_frame-first_frame-int(time*0.8):collision_frame-first_frame+int(time*0.2)]
    else:
        features_path = features_path[-time:]
    frame_features = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'frame.pt')).unsqueeze(0)
    for temp_path in features_path[1:]:
        temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'frame.pt')).unsqueeze(0)
        frame_features = torch.cat((frame_features,temp),dim=0)
    
    if baseline >= 2:
        if tracking is None:
            temp = torch.zeros((20,256,7,7))
            roi = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'object.pt'))
            object_num = 20 if roi.shape[0]>20 else roi.shape[0]
            temp[:object_num] = roi[:object_num]
            roi = temp.unsqueeze(0)
            for temp_path in features_path[1:]:
                temp = torch.zeros((20,256,7,7))
                roi_temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'object.pt'))
                object_num = 20 if roi_temp.shape[0]>20 else roi_temp.shape[0]
                temp[:object_num] = roi_temp[:object_num]
                roi = torch.cat((roi,temp.unsqueeze(0)),dim=0)
        else:
            temp = torch.zeros((tracking,256,7,7))
            roi = torch.load(os.path.join(local_path,'features','rgb','front',features_path[0],'object.pt'))
            box_file = open(os.path.join(nas_path,'bbox','front',features_path[0]+'.json'))
            boxes = json.load(box_file)
            box_file.close()
            for i,box in enumerate(boxes):
                if tracker_data[str(box['actor_id'])]>=tracking:
                    continue
                temp[tracker_data[str(box['actor_id'])]] = roi[i]
            roi = temp.unsqueeze(0)
            for temp_path in features_path[1:]:
                temp = torch.zeros((tracking,256,7,7))
                roi_temp = torch.load(os.path.join(local_path,'features','rgb','front',temp_path,'object.pt'))
                box_file = open(os.path.join(nas_path,'bbox','front',temp_path+'.json'))
                boxes = json.load(box_file)
                box_file.close()
                for i,box in enumerate(boxes):
                    if tracker_data[str(box['actor_id'])]>=tracking:
                        continue
                    temp[tracker_data[str(box['actor_id'])]] = roi_temp[i]
                roi = torch.cat((roi,temp.unsqueeze(0)),dim=0)

    if baseline == 1:
        return frame_features, collision_label
    elif baseline == 2:
        return frame_features, roi, collision_label
    elif baseline == 3:
        risky_object += 1
        return frame_features, roi, collision_label, risky_object
