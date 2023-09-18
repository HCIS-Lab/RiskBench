from torchvision import transforms
import torch.nn as nn
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import argparse
import json
import cv2
import copy
import os.path as osp
import os
import config as cfg
from models.GAT_LSTM import GAT_LSTM as Model
import sys
sys.path.insert(0, '../../../')


def vis_test(img, center_x=None, center_y=None, w=None, h=None):
    # width = 1280.0
    # height = 720.0

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


towns = '10'  # '05' or '10'


def read_testdata():

    test_set = []
    data_type = ["interactive", "obstacle", "collision", "non-interactive"]

    for _type in data_type:
        data_root = f'/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/{_type}'

        for basic_scene in os.listdir(data_root):
            basic_scene_path = osp.join(
                data_root, basic_scene, 'variant_scenario')

            for var_scene in os.listdir(basic_scene_path):
                var_scene_path = osp.join(basic_scene_path, var_scene)

                if basic_scene[:2] == '10':     # 5_ or 10
                    test_set.append(var_scene_path)

    return test_set


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)
    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def load_weight(model, checkpoint):

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default='1', type=str)
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='all', type=str)
    parser.add_argument('--model',
                        default='snapshots/all/2022-10-21_002608_w_dataAug_attn/inputs-camera-epoch-20.pth',
                        type=str)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--vis', action='store_true', default=False)

    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args.inputs, time_steps=5, pretrained=False).to(device)
    model = load_weight(model, args.model)

    model.train(False)
    softmax = nn.Softmax(dim=1).to(device)

    camera_transforms = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                      [0.5, 0.5, 0.5],),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    time_steps = 5
    time_sample = 1  # 10
    visualize = args.vis

    if visualize:
        year, month, day, hour, minute, second = get_current_time()
        formated_time = f'{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}'
        vis_save_path = f'./vis/{args.cause}/{formated_time}'
        if not os.path.isdir(vis_save_path):
            os.makedirs(vis_save_path)

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

        # normalized_trackers[:, :, 3] = normalized_trackers[:,
        #                                                   :, 1] + normalized_trackers[:, :, 3]
        # normalized_trackers[:, :, 2] = normalized_trackers[:,
        #                                                   :, 0] + normalized_trackers[:, :, 2]

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

    def visualize_result(test_sample, frame_id, tracker, filename, gt):
        """
            gt: center_x, center_y, w, h
        """
        # width = 1280.0
        # height = 720.0
        width, height = 1, 1

        camera_name = f'{frame_id:08}.png'
        camera_path = osp.join(test_sample, 'rgb/front', camera_name)
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

    def draw_all_score(test_sample, frame_id, action_logits, trackers, filename, scenario_id_weather, tracking_id, confidence_go):
        # width = 1280.0
        # height = 720.0
        width, height = 1, 1
        colors_BGR = [(220, 20, 60), (240, 128, 128), (255, 182, 193), (0, 255, 0), (127, 255, 212), (103, 255, 255),
                      (0, 255, 255), (0, 191, 255), (30, 144, 255), (0, 0, 153), (204, 51, 153), (255, 0, 255), (0, 0, 255)]
        colors_RGB = [c[::-1] for c in colors_BGR]

        camera_name = str(frame_id).zfill(8)+'.png'
        camera_path = osp.join(test_sample, 'rgb/front', camera_name)
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
        plt.xticks(x, object_id)
        plt.xlabel('Object id')
        plt.ylim(0.0, 1.0)
        plt.title('Risk score')
        plt.legend()
        plt.savefig(filename+'_bar.png')
        # plt.show()

    all_test = read_testdata()
    correct = 0
    all_roi_result = dict()
    a = 1

    for cnt, test_sample in enumerate(all_test, 1):
        with torch.set_grad_enabled(False):

            # session = None
            folder = test_sample
            scenario_id = test_sample.split('/')[-3]

            dyn_desc = open(osp.join(folder, 'dynamic_description.json'))
            data = json.load(dyn_desc)
            dyn_desc.close()

            gt_cause_id = data['player']

            print("===================================================")
            print(test_sample)
            print(test_sample.split('/')[-3], test_sample.split('/')[-1])
            print(gt_cause_id)

            rgb_path = osp.join(test_sample, 'rgb/front')
            all_frame = sorted(os.listdir(rgb_path))

            first_frame_id = int(all_frame[0].split('.')[0])
            last_frame_id = int(all_frame[-1].split('.')[0])

            tracking_results = []
            tracking_results = np.load(osp.join(test_sample, 'tracking.npy'))

            each_roi_result = dict()

            for frame_id in range(first_frame_id+(time_steps-1)*time_sample, last_frame_id):

                et = int(frame_id)
                st = et - (time_steps-1)*time_sample
                pred_metrics = []
                target_metrics = []

                trackers, normalized_trackers, tracking_id = find_tracker(
                    tracking_results, st, et)

                normalized_trackers = torch.from_numpy(
                    normalized_trackers.astype(np.float32)).to(device)
                normalized_trackers = normalized_trackers.unsqueeze(0)
                num_box = len(trackers[0])

                camera_inputs = []
                for l in range(st, et+1, time_sample):

                    # camera_name = 'output{}.png'.format(str(l-1 + start_time))
                    camera_name = str(l).zfill(8)+'.png'
                    camera_path = osp.join(
                        test_sample, 'rgb/front', camera_name)

                    # save for later usage in intervention
                    read_image = Image.open(camera_path).convert('RGB')

                    camera_input = camera_transforms(read_image)
                    camera_input = np.array(camera_input)

                    # camera_input = torch.from_numpy(camera_input.astype(np.float32))
                    camera_inputs.append(camera_input)

                camera_inputs = torch.Tensor(camera_inputs)  # (t, c, w, h)
                camera_inputs = camera_inputs.view(
                    1, time_steps, 3, 360, 640).to(device)

                vel, att_score_lst = model(
                    camera_inputs, normalized_trackers, device)

                # Reshape and remove ego's attention
                att_score_lst = att_score_lst[-1][1: len(
                    tracking_id)+1].view(-1).tolist()

                # Apply Softmax on vel result
                confidence_go = softmax(vel).to('cpu').numpy()[0][0]

                ##### Find each object's attention score #####

                # print("object id:", tracking_id)
                # print("att_score:", att_score_lst)
                # print("confidence_go:", confidence_go)

                frame_result = dict()
                for idx in range(len(tracking_id)):
                    frame_result[str(tracking_id[idx])] = str(
                        att_score_lst[idx])

                frame_result["scenario_go"] = str(confidence_go)
                each_roi_result[frame_id] = frame_result

            scenario_name = test_sample.split(
                '/')[-3] + "_" + test_sample.split('/')[-1]
            all_roi_result[scenario_name] = each_roi_result

    ##### Store all frame, all object's risk score in json file #####
    roi_file_name = "RA_" + args.cause + \
        f"_town{towns}_timesteps=" + str(time_steps) + ".json"
    if not os.path.isdir("roi"):
        os.makedirs("roi")

    roi_path = os.path.join("roi", roi_file_name)

    with open(roi_path, "w") as f:
        json.dump(all_roi_result, f, indent=2, sort_keys=True)
