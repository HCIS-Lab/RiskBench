import os
import os.path as osp
import sys
import cv2
import json
import argparse
import numpy as np
import shutil
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from .instance_to_box import produce_boxes
from .behavior_tool import build_tracking

from single_stage.models.GAT_LSTM import GAT_LSTM as Model

sys.path.insert(0, '../')

# python test_jacky.py --cause interactive --time_steps 5 --show_process --vis --frame 95


def to_device(x, device):
    return x.unsqueeze(0).to(device)


def read_testdata(args, data_path='test_data'):

    test_frame = args.frame
    T = args.time_steps
    test_set = []
    img_path = osp.join(data_path, 'rgb', 'front')

    for frame in range(test_frame-T+1, test_frame+1):
        img = osp.join(img_path, f"{frame:08d}.png")

        if not osp.isfile(img):
            print(f"{img} not exist!!!")
            exit()

        test_set.append(osp.join(img_path, img))

    return test_set


def normalize_box(trackers, width, height):
    normalized_trackers = trackers.copy()

    # normalized_trackers[:, :, 3] = normalized_trackers[:,
    #                                                    :, 1] + normalized_trackers[:, :, 3]
    # normalized_trackers[:, :, 2] = normalized_trackers[:,
    #                                                    :, 0] + normalized_trackers[:, :, 2]

    tmp = normalized_trackers[:, :, 0] / width
    normalized_trackers[:, :, 0] = normalized_trackers[:, :, 1] / height
    normalized_trackers[:, :, 1] = tmp
    tmp = trackers[:, :, 2] / width
    normalized_trackers[:, :, 2] = normalized_trackers[:, :, 3] / height
    normalized_trackers[:, :, 3] = tmp

    return normalized_trackers


def find_tracker(tracking, start, end):

    width = 1280
    height = 720

    # t_array saves timestamps

    t_array = tracking[:, 0]
    tracking_index = tracking[np.where(t_array == end)[0], 1]
    # num_object = len(tracking_index)
    num_object = 60
    time_sample = 1

    trackers = np.zeros(
        [int((end-start)/time_sample+1), 60, 4])  # Tx(N+1)x4
    trackers[:, 0, :] = np.array(
        [0.0, 0.0, width, height])  # Ego bounding box

    for t in range(start, end+1, time_sample):
        current_tracking = tracking[np.where(t_array == t)[0]]
        for i, object_id in enumerate(tracking_index):

            if object_id in current_tracking[:, 1]:
                bbox = current_tracking[np.where(
                    current_tracking[:, 1] == object_id)[0], 2:6]
                bbox[:, 0] = np.clip(bbox[:, 0], 0, 1279)
                bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, 1279)
                bbox[:, 1] = np.clip(bbox[:, 1], 0, 719)
                bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, 719)
                trackers[int((t-start)/time_sample), i+1, :] = bbox

    trackers.astype(np.int32)
    normalized_trackers = normalize_box(trackers, width, height)

    return trackers, normalized_trackers, tracking_index


def visualize_result(data_path, frame_id, tracker, filename, gt):
    """
        gt: center_x, center_y, w, h
    """

    width, height = 1, 1

    camera_name = f'{frame_id:08}.png'
    camera_path = osp.join(data_path, 'rgb/front', camera_name)
    frame = cv2.imread(camera_path)
    box = tracker[-1]  # x1,y1,x2,y2

    gt_x1 = (gt[0]-0.5*gt[2])*width
    gt_x2 = (gt[0]+0.5*gt[2])*width
    gt_y1 = (gt[1]-0.5*gt[3])*height
    gt_y2 = (gt[1]+0.5*gt[3])*height

    cv2.rectangle(frame, (int(gt_x1), int(gt_y1)),
                  (int(gt_x2), int(gt_y2)), (0, 0, 255), 8)

    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
        box[2]), int(box[3])), (0, 255, 0), 3)

    cv2.imwrite(f'{filename}.png', frame)

    # cv2.imshow('filename.png', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_all_score(data_path, frame_id, att_score_lst, trackers, filename, tracking_id, confidence_go, vis=True):

    width, height = 1, 1
    colors_BGR = [(220, 20, 60), (240, 128, 128), (255, 182, 193), (0, 255, 0), (127, 255, 212), (103, 255, 255),
                  (0, 255, 255), (0, 191, 255), (30, 144, 255), (0, 0, 153), (204, 51, 153), (255, 0, 255), (0, 0, 255)]
    colors_RGB = [c[::-1] for c in colors_BGR]

    camera_name = str(frame_id).zfill(8)+'.png'
    camera_path = osp.join(data_path, 'rgb/front', camera_name)
    frame = cv2.imread(camera_path)
    risk_score = dict()

    for idx, score in enumerate(att_score_lst):
        color = colors_BGR[idx % len(colors_BGR)]

        risk_score[str(tracking_id[idx])] = np.float(score)
        box = trackers[-1][idx+1]  # x1,y1,x2,y2
        cv2.rectangle(frame, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, f'{round(score, 4)}', (int(box[0]), int(
            box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(filename+'_all.png', frame)

    if vis:
        cv2.imshow(filename+'_all.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def train(model, all_test, image_size, camera_transforms, device, data_path, clean_state):



    time_steps = 5 
    softmax = nn.Softmax(dim=1).to(device)

    
    with torch.set_grad_enabled(False):
        
        temp_weight = f"./single_stage/inference/temp_weight/interactive"
        if clean_state:
            if os.path.isdir(temp_weight):
                shutil.rmtree(temp_weight)
            if os.path.isfile("./single_stage/inference/temp_weight/roi_history.json"):
                os.remove("./single_stage/inference/temp_weight/roi_history.json")
        if not os.path.isdir(temp_weight):
            os.mkdir(temp_weight)

        frame_id = int(all_test[-1].split('/')[-1].split('.')[0])

        tracking_results = np.load(osp.join(data_path, 'tracking.npy'))
        
        start_time = 0
        use_mask = True
        time_sample = 1# time_sample

        et = int(frame_id)
        st = et - (time_steps-1)*time_sample
        
 
        trackers, normalized_trackers, tracking_id = find_tracker(tracking_results, st, et)
        
        normalized_trackers = torch.from_numpy(normalized_trackers.astype(np.float32)).to(device)
        normalized_trackers = normalized_trackers.unsqueeze(0)

        num_box = len(trackers[0])
        camera_inputs = []

####
        if os.path.isfile(f"{temp_weight}/wo_hx_{st-1}.npy"):
            hx = torch.tensor(
                np.load(f'{temp_weight}/wo_hx_{st-1}.npy')).to(device)
            cx = torch.tensor(
                np.load(f'{temp_weight}/wo_cx_{st-1}.npy')).to(device)
        else:
            hx = torch.zeros((60, 128)).to(device)
            cx = torch.zeros((60, 128)).to(device)
###
        # read image
        for l in range(st, et+1, time_sample):

            # camera_name = 'output{}.png'.format(str(l-1 + start_time))
            camera_name = str(l).zfill(8)+'.png'
            camera_path = osp.join(data_path, 'rgb/front', camera_name)

            # save for later usage in intervention
            read_image = Image.open(camera_path).convert('RGB')
            camera_input = camera_transforms(read_image)
            camera_input = np.array(camera_input)
            camera_inputs.append(camera_input)

        camera_inputs = torch.Tensor(camera_inputs)  # (t, c, w, h)
        camera_inputs = camera_inputs.view(1, time_steps, 3, 360, 640).to(device)

        
        # Test Model
        # vel, att_score_lst = model(camera_inputs, normalized_trackers, device)
        #         # Test Model
        vel, att_score_lst, hx, cx = model(camera_inputs, normalized_trackers, device, hx=hx.clone(), cx=cx.clone())



        np.save(f'{temp_weight}/wo_hx_{et}.npy',
                hx.cpu().detach().numpy())
        np.save(f'{temp_weight}/wo_cx_{et}.npy',
                cx.cpu().detach().numpy())

        # Reshape and remove ego's attention
        att_score_lst = att_score_lst[-1][1: len(tracking_id)+1].view(-1).tolist()

        # Apply Softmax on vel result
        confidence_go = softmax(vel).to('cpu').numpy()[0][0]

        
        ##### Find each object's attention score #####

        # print("object id:", tracking_id)
        # print("att_score:", att_score_lst)
        # print("confidence_go:", confidence_go)


    #if att_score_lst:
    #    threshold = 0.5 #args.threshold
        # cause_object_id = np.argmax(np.array(action_logits)[:, 0])
        # action_logits = action_logits - confidence_go
        # action_logits = np.clip(action_logits, 0, 1)

        # if args.show_process:
            # print(action_logits)
        #print("======================================")
        #print(f"Scenario s_go: {confidence_go}")

        # if confidence_go > threshold:
        #     print(
        #         f"Warning: Confidence go score (s_go) more than {threshold} without intervention!!!")

        # '''
        # if args.cause == 'obstacle' and tracking_id[cause_object_id-1] in gt_obstacle:
        #     print(f"Risky object id: {gt_obstacle}")
        # else:
        #     print(f"Risky object id: {tracking_id[cause_object_id-1]}")
        # '''

        # max_idx = att_score_lst.argmax()

        # print(f"Risky object id: {tracking_id[max_idx]}")
        # print(f"Risky object score: {att_score_lst[max_idx]}")
        # print("======================================")

        # if args.vis:
        #     filename = osp.join("inference/vis", str(frame_id))
        #     # visualize_result(data_path, frame_id, trackers[:, cause_object_id], filename, [
        #     #     center_x, center_y, w, h])
        #     draw_all_score(data_path, frame_id, att_score_lst, trackers,
        #                     filename, tracking_id, confidence_go, args.vis)

        # if tracking_id[cause_object_id-1] in gt_obstacle:
        #     return confidence_go, gt_obstacle, action_logits[cause_object_id][0]
        # else:
        #     return confidence_go, tracking_id[cause_object_id-1], action_logits[cause_object_id][0]

    return confidence_go, tracking_id, att_score_lst   
    

        

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epoch', default='1', type=str)
#     parser.add_argument('--inputs', default='camera', type=str)
#     parser.add_argument('--cause', default='interactive',
#                         type=str, required=True)
#     parser.add_argument('--model',
#                         default='inference/model_weight/all/2022-9-1_183046_w_dataAug_attn/inputs-camera-epoch-20.pth',
#                         type=str)
#     parser.add_argument('--gpu', default='1', type=str)
#     parser.add_argument('--time_steps', default=5, type=int, required=True)
#     parser.add_argument('--time_sample', default=1, type=int, required=False)
#     parser.add_argument('--frame', type=int, required=True)
#     parser.add_argument('--threshold', default=0.5, type=float)
#     parser.add_argument('--partial_conv', default=True, type=bool)
#     parser.add_argument('--fusion', default='attn',
#                         choices=['avg', 'gcn', 'attn'], type=str)
#     parser.add_argument('--vis', action='store_true', default=False)
#     parser.add_argument('--clean_state', action='store_true', default=True)
#     parser.add_argument('--show_process', action='store_true', default=False)

#     args = parser.parse_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = Model(args.inputs, time_steps=args.time_steps, pretrained=False, partialConv=args.partial_conv,
#                   fusion=args.fusion).to(device)

#     # load pretrained weight
#     state_dict = torch.load(args.model)
#     state_dict_copy = {}

#     for key in state_dict.keys():
#         state_dict_copy[key[7:]] = state_dict[key]

#     model.load_state_dict(state_dict_copy)
#     model.train(False)

#     image_size = (360, 640)
#     camera_transforms = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

#     all_test = read_testdata(args, data_path='inference/test_data')
#     produce_boxes(data_path='inference/test_data')
#     build_tracking(data_path='inference/test_data')
#     train(args, model, all_test, image_size,
#           camera_transforms, device, data_path='inference/test_data')
