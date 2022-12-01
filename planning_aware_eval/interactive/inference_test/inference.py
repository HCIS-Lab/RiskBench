import os
import argparse
import json
import torch
import numpy as np
import cv2

import inference_test.baseline2_model as baseline2_model
from inference_test.Model import Supervised
import inference_test.utils as utils


def read_input(data_path, start_frame):
    

    # png_list = sorted(os.listdir(img_path))
    # print(png_list)

    png_list = []
    for index in range(4,-1,-1):
        save_id = start_frame - index
        png_list.append(f'{save_id:08d}.png')
    # print(png_list)


    
    # json_list = sorted(os.listdir(bbox_path))

    json_list = []
    for index in range(4,-1,-1):
        save_id = start_frame - index
        json_list.append(f'{save_id:08d}.json')


    img_path = os.path.join(data_path, "rgb/front")
    # print(img_path)
    img_list = []
    for f in png_list:
        img = cv2.imread(os.path.join(img_path, f))
        img = cv2.resize(img, (640, 360))
        img_list.append(img)
    
    bbox_path = os.path.join(data_path, "bbox/front")
    bbox_list = []
    for j in json_list:
        json_file = open(os.path.join(bbox_path, j))
        data = json.load(json_file)
        json_file.close()
        bbox_list.append(data)

    utils.create_tracklet(data_path)
    utils.order_match(data_path)

    return img_list, bbox_list


def get_features(raw_img, bbox, max_obj, device, data_path, tracking=False):
    frame_features, roi_features = utils.run_model(raw_img, bbox, device)
    
    if tracking:
        #load tracker_data
        track_file = open(os.path.join(data_path, 'tracker.json'))
        track_data = json.load(track_file)
        track_file.close()
    
        temp = torch.zeros((max_obj, 256, 7, 7))
        roi = roi_features[0]
        for i, box in enumerate(bbox[0]):
            if track_data[str(box['actor_id'])] >= max_obj:
                continue
            temp[track_data[str(box['actor_id'])]] = roi[i]
    
        roi = temp.unsqueeze(0)
        for i, roi_temp in enumerate(roi_features[1:], 1):
            temp = torch.zeros((max_obj, 256, 7, 7))
    
            for j, box in enumerate(bbox[i]):
                if track_data[str(box['actor_id'])]>= max_obj:
                    continue
                temp[track_data[str(box['actor_id'])]] = roi_temp[j]
            roi = torch.cat((roi, temp.unsqueeze(0)),dim=0)
    else:
        temp = torch.zeros((20,256,7,7))
        roi = roi_features[0]
        object_num = 20 if roi.shape[0]>20 else roi.shape[0]
        temp[:object_num] = roi[:object_num]
        roi = temp.unsqueeze(0)
        for roi_temp in roi_features[1:]:
            temp = torch.zeros((20,256,7,7))
            object_num = 20 if roi_temp.shape[0]>20 else roi_temp.shape[0]
            temp[:object_num] = roi_temp[:object_num]
            roi = torch.cat((roi,temp.unsqueeze(0)),dim=0)

    frame_features, roi = frame_features.unsqueeze(0), roi.unsqueeze(0)
    return frame_features, roi


def inference(device, data_path, model_path, start_frame):
    n_obj = 40
    net = Supervised(device, n_obj=n_obj, n_frame=60, features_size=256*7*7).to(device)
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    net.eval()

    img, bbox = read_input(data_path, start_frame)

    with torch.no_grad():
        frame, roi = get_features(img, bbox, n_obj, device, data_path,  tracking=True)
        frame, roi = frame.to(device), roi.to(device)

        cur_len = roi.shape[1]
        net.n_frame = cur_len
        pred_c, _, pred_r, _ = net(roi, frame)
        
        pred_c = pred_c.cpu().numpy().squeeze(axis=0)#tx2
        score_r = pred_r.cpu().numpy().squeeze(axis=0)#txn

    track_file = open(os.path.join(data_path, 'tracker_inverse.json'))
    track_data = json.load(track_file)
    track_file.close()

    colli_thres = 0.5
    instance_thres = 0.8
    accum = 1
    result = []#list of (list:risk object) each frame
    #sweep each frame

    for i, score in enumerate(score_r):
        risk_list = []
        flag = True #for accumulate condition check
        if i < accum-1:
            flag = False
        else:
            for cnt in range(accum):#check accumulate condition
                if pred_c[i-cnt, 1] < colli_thres:
                    flag = False
                    break
                
        if flag:#predict risk exist
            for n in range(len(track_data)):#num of actual actors
                if n >= n_obj:
                    break
                if score[n] > instance_thres:
                    risk_list.append(track_data[str(n)])

        result.append(risk_list)

    # print(result)
    return result


def SA_inference(device, data_path, model_path, start_frame):
    accum = 1
    colli_thres = 0.5
    n_obj = 20
    instance_thres = 0.2

    net = baseline2_model.Baseline_SA(20,device,0,n_frame=5,features_size=256*7*7) 
    net= torch.nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    if device == "cpu":
        net = net.module.to(device)
    else:
        net = net.to(device)
    net.eval()

    img, bbox = read_input(data_path, start_frame)
    with torch.no_grad():
        frame, roi = get_features(img, bbox, None, device, data_path)
        frame, roi = frame.to(device), roi.to(device)

        cur_len = roi.shape[1]
        if device == "cpu":
            net.n_frame = cur_len
        else:
            net.module.n_frame = cur_len
        collision_score, risk_score, _= net(roi, frame)
        
        collision_score = collision_score.cpu().numpy().squeeze(axis=0)#txn
        risk_score = risk_score.cpu().numpy().squeeze(axis=0)#txn
        # print(score_r[-1])
        # risk_id = predict(score_r)
    result = []#list of (list:risk object) each frame
    #sweep each frame

    for i, score in enumerate(risk_score):
        risk_list = []
        flag = True #for accumulate condition check
        if i < accum-1:
            flag = False
        else:
            for cnt in range(accum):#check accumulate condition
                if collision_score[i-cnt, 1] < colli_thres:
                    flag = False
                    break
                
        if flag:#predict risk exist
            for n in range(len(bbox[i])):#num of actual actors
                if n >= n_obj:
                    break
                if score[n] > instance_thres:
                    risk_list.append(str(bbox[i][n]["actor_id"]))

        result.append(risk_list)
    return result
