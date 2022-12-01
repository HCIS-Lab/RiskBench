import copy
import json
import os
import time
import numpy as np
import torch

# from inference.instance_to_box import produce_boxes
from roi_two_stage.inference.behavior_tool import build_tracking
from single_stage.inference.test_gat import train
from torchvision import transforms
from single_stage.models.GAT_LSTM import GAT_LSTM as Model
import gdown




def read_testdata(data_path='inference/test_data'):
    img_path = os.path.join(data_path, 'rgb', 'front')
    imgs = os.listdir(img_path)
    imgs.sort()

    return imgs


def load_weight(model):
    checkpoint = './single_stage/inference/model_weight/all/2022-10-30_010032_w_dataAug_attn/inputs-camera-epoch-20.pth'


    if not os.path.exists('./single_stage/inference/model_weight/all/2022-10-30_010032_w_dataAug_attn/'):
        os.mkdir('./single_stage/inference/model_weight/all/2022-10-30_010032_w_dataAug_attn/')

    if not os.path.isfile(checkpoint):
        print("Download single stage weight")
        url = "https://drive.google.com/u/4/uc?id=1ZeTmPax75ivcW-XO4yZLgQWDAp08V2Ax&export=download"
        gdown.download(url, checkpoint)


    state_dict = torch.load('./single_stage/inference/model_weight/all/2022-10-30_010032_w_dataAug_attn/inputs-camera-epoch-20.pth')
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


# def read_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--inputs', default='camera', type=str)
#     parser.add_argument('--cause', default='all',
#                         type=str, required=False)
#     parser.add_argument('--model',
#                         default='model_weight/all/2022-10-30_010032_w_dataAug_attn/inputs-camera-epoch-20.pth',
#                         type=str)
#     parser.add_argument('--gpu', default='1', type=str)
#     parser.add_argument('--time_steps', default=5, type=int, required=False)
#     parser.add_argument('--time_sample', default=1, type=int, required=False)
#     parser.add_argument('--threshold', default=0.5, type=float)
#     parser.add_argument('--fusion', default='attn',
#                         choices=['avg', 'gcn', 'attn'], type=str)
#     parser.add_argument('--vis', action='store_true', default=False)
#     parser.add_argument('--clean_state', action='store_true', default=False)
#     parser.add_argument('--show_process', action='store_true', default=False)

#     args = parser.parse_args()

#     return args


def read_log(confidence_go, risk_id_list, score_list, sweeping=1, diff_threshold=0):

    risk_id_list = list(map(str, risk_id_list))
    file_name = "./single_stage/inference/temp_weight/roi_history.json"
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
        return info["confidence_go"]<0.5 and info["score"][actor_id] >= diff_threshold

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



def single_stage(start_frame, clean_state=False):

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
    
    png_list = []
    for index in range(4,-1,-1):
        save_id = start_frame - index
        png_list.append(f'{save_id:08d}.json')
    build_tracking(start_frame, data_path)

    model = Model('camera', time_steps = 5, pretrained=False).to(device)

    model = load_weight(model)
    model.train(False)


    confidence_go, risk_id_list, score_list = train(model, png_list, image_size,
                                                    camera_transforms, device, data_path, clean_state)


    is_risky_dict = read_log(confidence_go, risk_id_list, score_list)

    # # print(dict(zip(risk_id_list, score_list)))
    # print(is_risky_dict)

    # print(f"confidence go: {confidence_go:.4f}")
    output_list = []
    for key, value in is_risky_dict.items():
        if value == True:
            output_list.append(key)
    return output_list



# if __name__ == '__main__':



#     model = Model('camera', args.time_steps, pretrained=False).to(device)

#     model = load_weight(model)
#     model.train(False)

#     #start = time.time()
#     confidence_go, risk_id_list, score_list = train(args, model, all_test, image_size,
#                                                     camera_transforms, device, data_path=data_path)
#     #end = time.time()

#     is_risky_dict = read_log(confidence_go, risk_id_list, score_list)

#     # print(dict(zip(risk_id_list, score_list)))
#     print(is_risky_dict)

#     print(f"confidence go: {confidence_go:.4f}")
#     #print(f"{end-start:.2f}s")