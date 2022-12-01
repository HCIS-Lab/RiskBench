import numpy as np
import torch
from detectron2.structures import Boxes
from inference_test.maskrcnn import get_maskrcnn
from inference_test.MaskFormer.demo.demo import get_maskformer
import os
import os.path as osp
import json
import csv


def get_models(device):
    maskformer = get_maskformer(device)
    detectron = get_maskrcnn(device)
    return maskformer.to(device).eval(), detectron.to(device).eval()


def run_model(inputs_raw, bbox, device):
    """
        inputs: List[img], List[ List[Dict{"box","actor_id"}] ]
        return: frame_features, roi 
    """
    maskformer, model = get_models(device)

    inputs = []
    for frame in inputs_raw:
        height, width = frame.shape[:2]
        frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1)).to(device)
        inputs.append({"image": frame, "height": height, "width": width})

    with torch.no_grad():
        # p2~p5
        fpn_features = maskformer.get_fpn_features(inputs) # res5, [mask(p2),p2,p3,p4,p5]
        features_maskformer = fpn_features[4]
        ###new
        roi_input_list = []
        roi_input_size = []
        # data_list: batch of json
        for datas in bbox:
            temp_list = []
            roi_input_size.append(len(datas))
            for data in datas:
                temp_list.append(data['box'])
            temp_list = np.round(np.array(temp_list)*0.5)
            temp_list = torch.from_numpy(temp_list).to(device).view(-1,4)
            temp_list = Boxes(temp_list)
            roi_input_list.append(temp_list)
        roi_gt = model.roi_heads.box_pooler(fpn_features[1:],roi_input_list)
        counter = 0
        out_roi = []
        for size in roi_input_size:
            out_roi.append(roi_gt[counter:counter+size].detach().clone().cpu())
            counter += size

        return features_maskformer, out_roi


def order_match(data_path, n_obj=80, order_by_freq=False):
    match = {}  # dict[actor_id]: feature order
    match_inverse = {}
    count = {}
    order = 0
    with open(osp.join(data_path, 'tracklet.csv'), newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            actor_id = int(row[1]) # & 0xffff

            if actor_id not in match:
                match[actor_id] = order
                match_inverse[order] = actor_id
                order += 1
        
            if order == n_obj:
                break
        
        # ## order by freq
        if order_by_freq:
            for row in rows:
                actor_id = int(row[1])
                if actor_id not in count:
                    count[actor_id] = 1
                else:
                    count[actor_id] += 1
            # print(count)
            sorted_key = sorted(count, key=count.get, reverse=True)
            for actor_id in sorted_key:
                match[actor_id] = order
                order += 1

    # print(match)
    # print(order)
    with open(f"{data_path}/tracker.json", 'w') as f:
        json.dump(match, f)
    with open(f"{data_path}/tracker_inverse.json", 'w') as f:
        json.dump(match_inverse, f)


def create_tracklet(data_path):
    bbox_path = osp.join(data_path, "bbox/front")
    json_list = sorted(os.listdir(bbox_path))

    with open(osp.join(data_path, "tracklet.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for j in json_list:
            bbox_list = []
            id_list = []
            frame = int(osp.splitext(j)[0])

            json_file = open(osp.join(bbox_path, j))
            data = json.load(json_file)
            json_file.close()
            # data_list
            
            for dict in data:
                bbox_list.append(dict['box'])
                id_list.append(dict['actor_id'])
            for bbox, id in zip(bbox_list, id_list):
                # bbox: left_bot, right_top
                h = bbox[2]-bbox[0]
                w = bbox[3]-bbox[1]

                row = [frame, id, bbox[0], bbox[1], h, w]
                writer.writerow(row)