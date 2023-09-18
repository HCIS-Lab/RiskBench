from roi_two_stage.models import GCN as Model
import os
import os.path as osp
import sys
import cv2
import json
import argparse
import numpy as np
import shutil
import PIL.Image as Image


import torch
import torch.nn as nn
from torchvision import transforms



def to_device(x, device):
    return x.unsqueeze(0).to(device)




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
        [int((end-start)/time_sample+1), num_object+1, 4])  # Tx(N+1)x4
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


def draw_all_score(data_path, frame_id, action_logits, trackers, filename, tracking_id, confidence_go, vis=True):

    width, height = 1, 1
    colors_BGR = [(220, 20, 60), (240, 128, 128), (255, 182, 193), (0, 255, 0), (127, 255, 212), (103, 255, 255),
                  (0, 255, 255), (0, 191, 255), (30, 144, 255), (0, 0, 153), (204, 51, 153), (255, 0, 255), (0, 0, 255)]
    colors_RGB = [c[::-1] for c in colors_BGR]

    camera_name = str(frame_id).zfill(8)+'.png'
    camera_path = osp.join(data_path, 'rgb/front', camera_name)
    frame = cv2.imread(camera_path)
    risk_score = dict()

    for idx, score in enumerate(action_logits[1:]):
        color = colors_BGR[idx % len(colors_BGR)]

        risk_score[str(tracking_id[idx])] = np.float(score[0])
        box = trackers[-1][idx+1]  # x1,y1,x2,y2
        cv2.rectangle(frame, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, f'{score[0]:.4}', (int(box[0]), int(
            box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(filename+'_all.png', frame)

    if vis:
        cv2.imshow(filename+'_all.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def train(model, all_test, image_size, camera_transforms, device, data_path='inference/test_data', clean_state = True):

    # if args.vis:
    #     if not os.path.isdir("./inference/vis"):
    #         os.makedirs("./inference/vis")

    time_steps = 5 # args.time_steps
    softmax = nn.Softmax(dim=1).to(device)

    with torch.set_grad_enabled(False):

        # id_to_class = {}
        # for bbox_json in os.listdir(osp.join(data_path, 'bbox/front')):
        #     bbox_file = open(osp.join(data_path, 'bbox/front', bbox_json))
        #     bbox_info = json.load(bbox_file)
        #     bbox_file.close()
        #     for bbox in bbox_info:
        #         id_to_class[bbox["actor_id"]] = bbox["class"]

        temp_weight = f"./roi_two_stage/inference/temp_weight/interactive"#{args.cause}"
        if clean_state:
            if os.path.isdir(temp_weight):
                shutil.rmtree(temp_weight)
            if os.path.isfile("./roi_two_stage/inference/temp_weight/roi_history.json"):
                os.remove("./roi_two_stage/inference/temp_weight/roi_history.json")
        if not os.path.isdir(temp_weight):
            os.mkdir(temp_weight)

        frame_id = int(all_test[-1].split('/')[-1].split('.')[0])

        gt_obstacle = []

        # if args.cause == "obstacle":
        #     obstacle_info = open(osp.join(data_path, 'obstacle_info.json'))
        #     obstacle_id = json.load(obstacle_info)[
        #         "interactive_obstacle_list"].keys()
        #     obstacle_info.close()
        #     gt_obstacle = list(map(int, obstacle_id))

        tracking_results = np.load(osp.join(data_path, 'tracking.npy'))

        start_time = 0
        use_mask = True
        time_sample = 1 #args.time_sample

        et = int(frame_id)
        st = et - (time_steps-1)*time_sample

        trackers, normalized_trackers, tracking_id = find_tracker(
            tracking_results, st, et)
        normalized_trackers = torch.from_numpy(
            normalized_trackers.astype(np.float32)).to(device)
        normalized_trackers = normalized_trackers.unsqueeze(0)

        num_box = len(trackers[0])
        camera_inputs = []
        action_logits = []

        if os.path.isfile(f"{temp_weight}/wo_hx_{st-1}.npy"):
            hx = torch.tensor(
                np.load(f'{temp_weight}/wo_hx_{st-1}.npy')).to(device)
            cx = torch.tensor(
                np.load(f'{temp_weight}/wo_cx_{st-1}.npy')).to(device)
        else:
            hx = torch.zeros((num_box, 512)).to(device)
            cx = torch.zeros((num_box, 512)).to(device)

        # without intervention
        for l in range(st, et+1, time_sample):

            camera_name = str(l + start_time).zfill(8)+'.png'
            camera_path = osp.join(data_path, 'rgb/front', camera_name)

            # save for later usage in intervention
            read_image = Image.open(camera_path).convert('RGB')
            camera_inputs.append(read_image)

            camera_input = camera_transforms(read_image)
            camera_input = np.array(camera_input)

            camera_input = to_device(torch.from_numpy(
                camera_input.astype(np.float32)), device)
            mask = torch.ones(
                (1, 3, image_size[0], image_size[1])).to(device)

            # assign index for RoIAlign
            # box_ind : (BxN)
            box_ind = np.array(
                [np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
            box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

            #if args.partial_conv:
            camera_input = model.backbone.features(camera_input, mask)
            #else:
            #    camera_input = model.backbone.features(camera_input)

            # camera_input: 1xCxHxW
            # normalized_trackers: 1xNx4
            # ROIAlign : BxNx1280
            time_sample = 1
            tracker = normalized_trackers[:,
                                          (l - st)//time_sample].contiguous()
            box_ind = box_ind.view(1, num_box)
            feature_input = model.cropFeature(
                camera_input, tracker, box_ind)
            # check feature_input
            feature_input = feature_input.view(-1, 1280)

            hx, cx = model.step(feature_input, hx, cx)

        np.save(f'{temp_weight}/wo_hx_{et}.npy',
                hx.cpu().detach().numpy())
        np.save(f'{temp_weight}/wo_cx_{et}.npy',
                cx.cpu().detach().numpy())

        updated_feature, _ = model.message_passing(
            hx, normalized_trackers)  # BxH
        vel = model.vel_classifier(model.drop(updated_feature))
        confidence_go = softmax(vel).to('cpu').numpy()[0][0]

        obstacle_idx = []
        obstacle_region = None

        for i in range(len(tracking_id)):
            if tracking_id[i] in gt_obstacle:
                obstacle_idx.append(i+1)

        # with intervention
        for i in range(len(tracking_id)+1):
            if i == 0:
                action_logits.append([0.0, 1.0])
                continue

            if i in obstacle_idx:
                if obstacle_region is None:
                    id_idx = obstacle_idx
                    obstacle_region = i
                else:
                    action_logits.append(action_logits[obstacle_region])
                    continue
            else:
                id_idx = [i]

            if os.path.isfile(f"{temp_weight}/wo_hx_{st-1}.npy"):
                hx = torch.tensor(
                    np.load(f'{temp_weight}/wo_hx_{st-1}.npy')).to(device)
                cx = torch.tensor(
                    np.load(f'{temp_weight}/wo_cx_{st-1}.npy')).to(device)
            else:
                hx = torch.zeros((num_box, 512)).to(device)
                cx = torch.zeros((num_box, 512)).to(device)

            #  trackers: Tx(N+1)x4 (x1, y1, w, h ) without normalization
            #  normalized_trackers: : Tx(N+1)x4 (y1, x1, y2, x2 ) with normalization
            trackers, normalized_trackers, tracking_id = find_tracker(
                tracking_results, st, et)

            normalized_trackers = torch.from_numpy(
                normalized_trackers.astype(np.float32)).to(device)
            normalized_trackers = normalized_trackers.unsqueeze(0)

            for l in range(st, et+1, time_sample):
                camera_input = np.array(
                    camera_inputs[(l - st)//time_sample])

                for j in id_idx:
                    camera_input[int(trackers[(l - st)//time_sample, j, 1]):int(trackers[(l - st)//time_sample, j, 3]),
                                 int(trackers[(l - st)//time_sample, j, 0]):int(trackers[(l - st)//time_sample, j, 2]), :] = 0

                camera_input = Image.fromarray(np.uint8(camera_input))
                np_camera_input = np.array(camera_input)

                camera_input = camera_transforms(camera_input)
                camera_input = np.array(camera_input)

                camera_input = to_device(torch.from_numpy(
                    camera_input.astype(np.float32)), device)

                # assign index for RoIAlign
                # box_ind : (BxN)
                box_ind = np.array(
                    [np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                box_ind = torch.from_numpy(
                    box_ind.astype(np.int32)).to(device)

                if not use_mask:
                    mask = torch.ones(
                        (1, 3, image_size[0], image_size[1])).to(device)

                else:
                    mask = np.ones((1, 3, image_size[0], image_size[1]))
                    for j in id_idx:
                        x1 = int(
                            trackers[(l - st) // time_sample, j, 1]/720*image_size[0])  # x1
                        x2 = int(
                            trackers[(l - st) // time_sample, j, 3]/720*image_size[0])  # x2
                        y1 = int(
                            trackers[(l - st) // time_sample, j, 0]/1280*image_size[1])  # y1
                        y2 = int(
                            trackers[(l - st) // time_sample, j, 2]/1280*image_size[1])  # y2
                        mask[:, :, x1:x2, y1:y2] = 0

                    mask = torch.from_numpy(
                        mask.astype(np.float32)).to(device)

                # if args.partial_conv:
                camera_input = model.backbone.features(
                    camera_input, mask)
                #else:
                #    camera_input = model.backbone.features(camera_input)

                tracker = normalized_trackers[:,
                                              (l - st)//time_sample].contiguous()
                tracker[:, id_idx, :] = 0

                box_ind = box_ind.view(1, num_box)
                feature_input = model.cropFeature(
                    camera_input, tracker, box_ind)

                # check feature_input
                feature_input = feature_input.view(-1, 1280)
                hx, cx = model.step(feature_input, hx, cx)

            intervened_trackers = torch.ones(
                (1, time_steps, num_box, 4)).to(device)
            intervened_trackers[:, :, id_idx, :] = 0.0

            intervened_trackers = intervened_trackers * normalized_trackers

            updated_feature, _ = model.message_passing(
                hx, intervened_trackers)  # BxH

            vel = model.vel_classifier(model.drop(updated_feature))
            confidence = softmax(vel).to('cpu').numpy()[0]
            action_logits.append(confidence)  # Nx2
            # print(session, start, end, i, trackers[:,i ], action_logits[i])

        if action_logits:
            threshold = 0.5 #args.threshold
            cause_object_id = np.argmax(np.array(action_logits)[:, 0])
            # action_logits = action_logits - confidence_go
            action_logits = np.clip(action_logits, 0, 1)

            # if args.show_process:
            #     # print(action_logits)
            #     print("======================================")
            #     print(f"Scenario s_go: {confidence_go}")

            #     if confidence_go > threshold:
            #         print(
            #             f"Warning: Confidence go score (s_go) more than {threshold} without intervention!!!")

            #     if args.cause == 'obstacle' and tracking_id[cause_object_id-1] in gt_obstacle:
            #         print(f"Risky object id: {gt_obstacle}")
            #     else:
            #         print(f"Risky object id: {tracking_id[cause_object_id-1]}")

            #     print(
            #         f"Risky object score: {action_logits[cause_object_id][0]}")
            #     print("======================================")

            # if args.vis:
            #     filename = osp.join("inference/vis", str(frame_id))
            #     # visualize_result(data_path, frame_id, trackers[:, cause_object_id], filename, [
            #     #     center_x, center_y, w, h])
            #     draw_all_score(data_path, frame_id, action_logits, trackers,
            #                    filename, tracking_id, confidence_go, args.vis)

            # if tracking_id[cause_object_id-1] in gt_obstacle:
            #     return confidence_go, gt_obstacle, action_logits[cause_object_id][0]
            # else:
            #     return confidence_go, tracking_id[cause_object_id-1], action_logits[cause_object_id][0]

            return confidence_go, tracking_id, action_logits[1:, 0]


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
#     parser.add_argument('--clean_state', action='store_true', default=False)
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
