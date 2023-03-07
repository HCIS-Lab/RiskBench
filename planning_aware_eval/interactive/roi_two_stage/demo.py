import copy
import json
import os
import numpy as np
import torch
import argparse
from roi_two_stage.inference.behavior_tool import build_tracking
from roi_two_stage.inference.test_jacky import train
from torchvision import transforms

import gdown







def read_log(confidence_go, risk_id_list, score_list, sweeping=1, diff_threshold=0.2):

    risk_id_list = list(map(str, risk_id_list))
    file_name = "./roi_two_stage/inference/temp_weight/roi_history.json"
    is_risky_dict = dict()
    history = []

    if os.path.exists(file_name):

        json_file = open(file_name)
        history = json.load(json_file)
        json_file.close()

        info = {}
        info["frame no."] = history[-1]["frame no."]+1
        info["confidence_go"] = np.float(confidence_go)
        info["score"] = dict(zip(risk_id_list, score_list))

        history.append(info)

    else:
        info = {}
        info["frame no."] = 5
        info["confidence_go"] = np.float(confidence_go)
        info["score"] = dict(zip(risk_id_list, score_list))
        history = [info]

    with open(file_name, 'w') as f:
        json.dump(history, f, indent=4)

    def check_risky(info, actor_id):
        return info["score"][actor_id]-info["confidence_go"] >= diff_threshold

    for actor_id in risk_id_list:
        if len(history) < sweeping:
            is_risky_dict[actor_id] = False
        else:
            is_risky_dict[actor_id] = True
            for info in history[::-1][:sweeping]:
                if not check_risky(info, actor_id):
                    is_risky_dict[actor_id] = False
                    break

    return is_risky_dict



def roi_two_stage_inference(start_frame, clean_state=False, model = None):

    device = torch.device('cuda')
    data_path = './roi_two_stage/inference/test_data'

    # start_frame = 100
    image_size = (360, 640)
    camera_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


    #c args = read_args()

    png_list = []
    for index in range(4,-1,-1):
        save_id = start_frame - index
        png_list.append(f'{save_id:08d}.json')

    build_tracking(start_frame, data_path)



    confidence_go, risk_id_list, score_list = train(model, png_list, image_size, camera_transforms, device, data_path=data_path, clean_state=clean_state)
    is_risky_dict = read_log(confidence_go, risk_id_list, score_list)

    # print(dict(zip(risk_id_list, score_list)))
    # print(is_risky_dict)
    #print(f"confidence go: {confidence_go:.4f}")
    # for 
    output_list = []
    for key, value in is_risky_dict.items():
        if value == True:
            output_list.append(key)
    return output_list

    # print(output_list)
        






