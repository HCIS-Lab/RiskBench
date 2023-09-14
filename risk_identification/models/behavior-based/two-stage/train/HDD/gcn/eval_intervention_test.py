
import os
import os.path as osp
import sys
import cv2
import json
import copy
import argparse
import csv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, '../../../')
from models import GCN as Model
import config as cfg

# python eval_intervention_test.py --cause interactive --time_steps 5 --vis --show_process


def vis_test(img, center_x=None, center_y=None, w=None, h=None):

    width, height = 1, 1

    if w != None:
        print(center_x, center_y, w, h)

        gt_x1 = (center_x-0.5*w)*width
        gt_x2 = (center_x+0.5*w)*width
        gt_y1 = (center_y-0.5*h)*height
        gt_y2 = (center_y+0.5*h)*height

        cv2.rectangle(img, (int(gt_x1), int(gt_y1)),
                      (int(gt_x2), int(gt_y2)), (0, 0, 255), 8)

    cv2.imshow('filename.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_device(x, device):
    return x.unsqueeze(0).to(device)


def read_testdata(data_type):
    test_set = []
    data_root = f'/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/{data_type}'

    for basic_scene in os.listdir(data_root):
        basic_scene_path = osp.join(data_root, basic_scene, 'variant_scenario')

        for var_scene in os.listdir(basic_scene_path):
            var_scene_path = osp.join(basic_scene_path, var_scene)

            if basic_scene[:2] == '5_':
                test_set.append(var_scene_path)

    return test_set


def load_weight(model, checkpoint):

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)
    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default='1', type=str)
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='interactive',
                        type=str, required=True)
    parser.add_argument('--model',
                        default='snapshots/all/2022-9-1_183046_w_dataAug_attn/inputs-camera-epoch-20.pth',
                        type=str)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--time_steps', default=3, type=int, required=True)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--fusion', default='attn',
                        choices=['avg', 'gcn', 'attn'], type=str)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--show_process', action='store_true', default=False)
    parser.add_argument('--closest', action='store_true', default=False)

    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args.inputs, time_steps=args.time_steps, pretrained=False, partialConv=args.partial_conv,
                  fusion=args.fusion).to(device)
    model = load_weight(model, args.model)


    model.train(False)
    softmax = nn.Softmax(dim=1).to(device)

    image_size = (360, 640)
    camera_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                      [0.5, 0.5, 0.5],),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    time_steps = args.time_steps
    time_sample = 1  # 10
    visualize = args.vis

    if visualize:
        year, month, day, hour, minute, second = get_current_time()
        formated_time = f'{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}'
        vis_save_path = f'./vis/{args.cause}/{formated_time}'
        if not os.path.isdir(vis_save_path):
            os.makedirs(vis_save_path)

        if not os.path.isdir("RA"):
            os.makedirs("RA")

        with open(f"RA/{formated_time}.json", "w") as f:
            json.dump({}, f, indent=4)

    def plot_vel(pred, target, plot_name):
        t = len(pred)
        timestamp = range(1, t+1)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(timestamp, pred, marker='o', label="Prediction")
        ax.plot(timestamp, target, marker='o', label="Target")
        # Place a legend to the right of this smaller subplot.
        ax.legend(loc='upper right')
        fig.savefig(plot_name)  # save the figure to file
        plt.close(fig)

    def normalize_box(trackers, width, height):
        normalized_trackers = trackers.copy()

        # normalized_trackers[:, :, 2] = normalized_trackers[:,
        #                                                    :, 0] + normalized_trackers[:, :, 2]
        # normalized_trackers[:, :, 3] = normalized_trackers[:,
        #                                                    :, 1] + normalized_trackers[:, :, 3]

        tmp = normalized_trackers[:, :, 0] / width
        normalized_trackers[:, :, 0] = normalized_trackers[:, :, 1] / height
        normalized_trackers[:, :, 1] = tmp

        tmp = normalized_trackers[:, :, 2] / width
        normalized_trackers[:, :, 2] = normalized_trackers[:, :, 3] / height
        normalized_trackers[:, :, 3] = tmp

        return normalized_trackers

    def find_tracker(tracking, start, end):

        width = 1280
        height = 720

        # t_array saves timestamps

        t_array = tracking[:, 0]
        tracking_index = tracking[np.where(t_array == end)[0], 1]
        num_object = len(tracking_index)

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

    def visualize_result(var_scene_path, frame_id, tracker, filename, gt):
        """
            gt: center_x, center_y, w, h
        """

        width, height = 1, 1

        camera_name = f'{frame_id:08}.png'
        camera_path = osp.join(var_scene_path, 'rgb/front', camera_name)
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

    def draw_all_score(var_scene_path, frame_id, action_logits, trackers, filename, scenario_id_weather, tracking_id, confidence_go):

        width, height = 1, 1
        colors_BGR = [(220, 20, 60), (240, 128, 128), (255, 182, 193), (0, 255, 0), (127, 255, 212), (103, 255, 255),
                      (0, 255, 255), (0, 191, 255), (30, 144, 255), (0, 0, 153), (204, 51, 153), (255, 0, 255), (0, 0, 255)]
        colors_RGB = [c[::-1] for c in colors_BGR]

        camera_name = str(frame_id).zfill(8)+'.png'
        camera_path = osp.join(var_scene_path, 'rgb/front', camera_name)
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

        # {'10_t1-7_1_p_c_r_1_0' : {58:  {100: 50, 103: 60, 101:20}}, \
        #   {59: {, },  60:{,},  ...}, '10_i-1_1_p_c_l_1_j' : {58:{,}}, ...}
        with open(f"RA/{formated_time}.json") as f:
            vision_RA = json.load(f)
        vision_RA.setdefault(str(scenario_id_weather), {})
        vision_RA[str(scenario_id_weather)][frame_id] = risk_score

        with open(f"RA/{formated_time}.json", "w") as f:
            json.dump(vision_RA, f, indent=4)

        cv2.imwrite(filename+'_all.png', frame)
        # cv2.imshow('filename.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        object_id = tracking_id
        object_score = np.array(np.array(action_logits)[1:, 0])
        x = np.arange(len(object_id))

        plt.clf()
        plt.bar(x, object_score, color=np.array(
            colors_RGB[:len(object_id)])/255.)
        plt.plot(x, [confidence_go]*len(x), color='black',
                 label='confidence_go', linewidth=0.7)
        if len(object_id) > 8:
            plt.xticks(x, object_id, rotation=45)
        else:
            plt.xticks(x, object_id)

        plt.xlabel('Object id')
        plt.ylim(0.0, 1.0)
        plt.title(f'Confidence_go score:{confidence_go:.2f}')
        plt.legend()
        plt.savefig(filename+'_bar.png')
        # plt.show()

    def check_close(var_scene_path, scenario_id, frame_id, data, gt_obstacle, obstacle_trans=None):
        actor_id_list = []
        for actor in data:
            actor_id_list.append(actor["actor_id"])

        actor_and_trans = {}
        csv_path = os.path.join(
            var_scene_path, 'trajectory_frame', f"{scenario_id}.csv")
        with open(csv_path) as csvfile:
            rows = csv.DictReader(csvfile)

            for row in rows:
                if int(row["FRAME"]) == frame_id:
                    actor_and_trans[row["TRACK_ID"]] = {
                        "x": float(row["X"]), "y": float(row["Y"])}

        ego_x, ego_y = actor_and_trans["player"]["x"], actor_and_trans["player"]["y"]
        gt_x, gt_y = actor_and_trans[str(
            gt_obstacle[0])]["x"], actor_and_trans[str(gt_obstacle[0])]["y"]
        gt_dis = (gt_x-ego_x)**2 + (gt_y-ego_y)**2
        min_dis = (99999, 0)    # (dis, id)

        for id in actor_and_trans:
            if id.isdigit() and int(id) in actor_id_list:
                x, y = actor_and_trans[id]["x"], actor_and_trans[id]["y"]
                dis = (x-ego_x)**2 + (y-ego_y)**2
                # if dis < 100 and int(id) not in gt_obstacle:
                #     return True
                if dis < min_dis[0]:
                    min_dis = (dis, int(id))

        # return False

        if obstacle_trans is not None:
            for id in obstacle_trans:
                x, y = obstacle_trans[id]["transform"]
                dis = (x-ego_x)**2 + (y-ego_y)**2
                if dis < min_dis[0]:
                    min_dis = (dis, int(id))

        if min_dis[1] in gt_obstacle:
            return False
        else:
            return True

    all_test = read_testdata(args.cause)
    correct = {"total_correct": 0, "low_correct": 0,
               "mid_correct": 0, "high_correct": 0}
    cnt = {"total_correct": 0, "low_correct": 0,
           "mid_correct": 0, "high_correct": 0}
    threshold = 0.5
    diff_threshold = 0
    TP, FN, FP, TN = 0, 0, 0, 0

    for var_scene_path in all_test:
        with torch.set_grad_enabled(False):

            # session = None
            weather = var_scene_path.split('/')[-1]
            random_actor = weather.split('_')[1]
            scenario_id = var_scene_path.split('/')[-3]

            dyn_desc = open(
                osp.join(var_scene_path, 'dynamic_description.json'))
            data = json.load(dyn_desc)
            dyn_desc.close()

            gt_obstacle = []
            obstacle_trans = None

            if args.cause != 'obstacle':
                for key in data.keys():
                    if key.isdigit():
                        gt_obstacle.append(data[key])
                        break

            # read testing data (behavior change frame)
            if args.cause == 'obstacle':
                behavior_change_path = osp.join(
                    var_scene_path, 'obstacle_info.json')
                behavior_change_file = open(behavior_change_path, 'r')
                obstacle_info = json.load(behavior_change_file)
                behavior_change_file.close()

                start, end = obstacle_info["interactive frame"]
                frame_id = (start + end)//2

                gt_obstacle = list(
                    map(int, list(obstacle_info["interactive_obstacle_list"].keys())))
                obstacle_trans = obstacle_info["interactive_obstacle_list"]

            elif args.cause == 'collision':

                frame_id = -1
                bbox_folder = os.path.join(var_scene_path, 'bbox', 'front')
                frames = os.listdir(bbox_folder)
                frames.sort()

                for frame in frames[::-1]:
                    bbox_path = os.path.join(bbox_folder, frame)
                    bbox_json = open(bbox_path, 'r')
                    bbox_data = json.load(bbox_json)
                    bbox_json.close()

                    for actor in bbox_data:
                        if actor['actor_id'] == gt_obstacle[0]:
                            frame_id = int(frame.split('.')[0])-5
                            break

                    if frame_id != -1:
                        break

                if frame_id == -1:
                    print(
                        f"Error: No interactive frame in collision/{scenario_id}/{weather}!!!")
                    continue

            else:
                behavior_change_path = var_scene_path.split('variant_scenario')[
                    0]+'behavior_annotation.txt'

                behavior_change_file = open(behavior_change_path, 'r')
                temp = behavior_change_file.readlines()
                frame_id = (int(temp[1].strip()) + int(temp[0].strip()))//2

            if args.show_process:
                print("===================================================")
                print(var_scene_path.split('/')
                      [-3], var_scene_path.split('/')[-1])
                # print(gt_obstacle)

            # read bbox of risk object
            bbox_path = osp.join(var_scene_path, 'bbox/front')
            frame_name = f'{frame_id:08}.json'
            json_file = open(osp.join(bbox_path, frame_name))
            data = json.load(json_file)
            json_file.close()

            if args.closest and not check_close(var_scene_path, scenario_id, frame_id, data, gt_obstacle, obstacle_trans):
                continue

            if args.cause != 'obstacle':
                for actor in data:
                    if actor['actor_id'] == gt_obstacle[0]:
                        bbox = actor['box']
                        break

                center_x = float((bbox[0]+bbox[2])/2)     # width
                center_y = float((bbox[1]+bbox[3])/2)     # height
                w = float(bbox[2]-bbox[0])
                h = float(bbox[3]-bbox[1])

            tracking_results = np.load(
                osp.join(var_scene_path, 'tracking.npy'))

            start_time = 0  # 59
            use_mask = True

            pred_metrics = []
            target_metrics = []

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

            hx = torch.zeros((num_box, 512)).to(device)
            cx = torch.zeros((num_box, 512)).to(device)

            # without intervention
            for l in range(st, et+1, time_sample):

                # camera_name = 'output{}.png'.format(str(l-1 + start_time))
                camera_name = str(l + start_time).zfill(8)+'.png'
                camera_path = osp.join(
                    var_scene_path, 'rgb/front', camera_name)

                # save for later usage in intervention
                read_image = Image.open(camera_path).convert('RGB')
                camera_inputs.append(read_image)

                camera_input = camera_transforms(
                    read_image)
                camera_input = np.array(camera_input)

                #########################
                # vis_test(np.array(read_image), center_x, center_y, w, h)
                #########################

                camera_input = to_device(torch.from_numpy(
                    camera_input.astype(np.float32)), device)
                mask = torch.ones(
                    (1, 3, image_size[0], image_size[1])).to(device)

                # assign index for RoIAlign
                # box_ind : (BxN)
                box_ind = np.array(
                    [np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

                if args.partial_conv:
                    camera_input = model.backbone.features(camera_input, mask)
                else:
                    camera_input = model.backbone.features(camera_input)

                # camera_input: 1xCxHxW
                # normalized_trackers: 1xNx4
                # ROIAlign : BxNx1280
                tracker = normalized_trackers[:,
                                              (l - st)//time_sample].contiguous()
                box_ind = box_ind.view(1, num_box)
                feature_input = model.cropFeature(
                    camera_input, tracker, box_ind)
                # check feature_input
                feature_input = feature_input.view(-1, 1280)

                hx, cx = model.step(feature_input, hx, cx)

            updated_feature, _ = model.message_passing(
                hx, normalized_trackers)  # BxH
            vel = model.vel_classifier(model.drop(updated_feature))
            confidence_go = softmax(vel).to('cpu').numpy()[0][0]

            obstacle_idx = []
            obstacle_region = None

            for i in range(num_box-1):
                if tracking_id[i] in gt_obstacle:
                    obstacle_idx.append(i+1)

            # with intervention
            for i in range(num_box):
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
                                trackers[(l - st) // time_sample, j, 1]/720*image_size[0])
                            x2 = int(
                                trackers[(l - st) // time_sample, j, 3]/720*image_size[0])
                            y1 = int(
                                trackers[(l - st) // time_sample, j, 0]/1280*image_size[1])
                            y2 = int(
                                trackers[(l - st) // time_sample, j, 2]/1280*image_size[1])
                            
                            mask[:, :, x1:x2, y1:y2] = 0

                        mask = torch.from_numpy(
                            mask.astype(np.float32)).to(device)


                    if args.partial_conv:
                        camera_input = model.backbone.features(
                            camera_input, mask)
                    else:
                        camera_input = model.backbone.features(camera_input)

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
                action_logits.append(softmax(vel).to('cpu').numpy()[0])  # Nx2
                # print(session, start, end, i, trackers[:,i ], action_logits[i])

            if action_logits:
                cause_object_id = np.argmax(np.array(action_logits)[:, 0])
                # action_logits = action_logits - confidence_go
                action_logits = np.clip(action_logits, 0, 1)
                risky_score = action_logits[cause_object_id][0]

                scenario_id_weather = scenario_id + \
                    '_' + weather

                if visualize:
                    filename = vis_save_path + '/' + \
                        scenario_id_weather + str(frame_id)
                    visualize_result(var_scene_path, frame_id, trackers[:, cause_object_id], filename, [
                        center_x, center_y, w, h])
                    draw_all_score(var_scene_path, frame_id, action_logits, trackers,
                                   filename, scenario_id_weather, tracking_id, confidence_go)

                cnt["total_correct"] += 1
                cnt[f"{random_actor}_correct"] += 1

                if (confidence_go > threshold) or (risky_score-confidence_go < diff_threshold):
                    FN += 1
                else:
                    if (args.cause != 'obstacle' and tracking_id[cause_object_id-1] == gt_obstacle[0]) or \
                            (args.cause == 'obstacle' and cause_object_id in obstacle_idx):
                        TP += 1
                        correct["total_correct"] += 1
                        correct[f"{random_actor}_correct"] += 1
                    else:
                        FP += 1

                if args.show_process:
                    print(f"Scenario s_go: {confidence_go}")
                    print(f'Sample: {cnt["total_correct"]}/{len(all_test)}')

                    # if cnt["low_correct"] != 0:
                    #     print(
                    #         f'Random_actor==low precision: {correct["low_correct"]}/{cnt["low_correct"]} {(correct["low_correct"]/cnt["low_correct"])*100:.2f}%')

                    # if cnt["mid_correct"] != 0:
                    #     print(
                    #         f'Random_actor==mid precision: {correct["mid_correct"]}/{cnt["mid_correct"]} {(correct["mid_correct"]/cnt["mid_correct"])*100:.2f}%')

                    # if cnt["high_correct"] != 0:
                    #     print(
                    #         f'Random_actor==high precision: {correct["high_correct"]}/{cnt["high_correct"]} {(correct["high_correct"]/cnt["high_correct"])*100:.2f}%')

                    print(
                        f'Total Recall: {(correct["total_correct"]/cnt["total_correct"])*100:.2f}%')

    # if cnt["low_correct"] != 0:
    #     print(
    #         f'Random_actor==low precision: {correct["low_correct"]}/{cnt["low_correct"]} {(correct["low_correct"]/cnt["low_correct"])*100:.2f}%')

    # if cnt["mid_correct"] != 0:
    #     print(
    #         f'Random_actor==mid precision: {correct["mid_correct"]}/{cnt["mid_correct"]} {(correct["mid_correct"]/cnt["mid_correct"])*100:.2f}%')

    # if cnt["high_correct"] != 0:
    #     print(
    #         f'Random_actor==high precision: {correct["high_correct"]}/{cnt["high_correct"]} {(correct["high_correct"]/cnt["high_correct"])*100:.2f}%')

    print(
        f'Total Recall: {(correct["total_correct"]/cnt["total_correct"])*100:.2f}%')

    n = cnt["total_correct"]
    print(
        f"TP: [{TP}, {TP/n:.4f}], FN: [{FN}, {FN/n:.4f}], FP: [{FP}, {FP/n:.4f}]")
