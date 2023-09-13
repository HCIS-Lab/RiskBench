import sys

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import math
import torch
import torch.utils.data as data
import re
from collections import defaultdict
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--model", default='model_controller')
    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--memories_path",
                        default='carla_dataset_all/')
    parser.add_argument('--type', default='obstacle',
                        type=str, required=True)
    parser.add_argument('--future_length', default=30,
                        type=int, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--data_path', type=str, default='test_data')
    return parser.parse_args()


class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.
    The building class is merged into the background class
    0:background 1:street 2:sidewalk, 3:building 4: vegetation ---> 0:background 1:street 2:sidewalk, 3: vegetation
    """

    def __init__(self, train, name, weather_name, num_time, vehicle, f_frame, now_frame, pred_len):

        self.pasts = []           # [len_past, 2]
        self.presents = []        # position in complete scene
        self.futures = []         # [len_future, 2]
        points = np.vstack((vehicle['X'], vehicle['Y'])).T
        temp_past = points[num_time:num_time + 20].copy()
        origin = temp_past[-1]
        temp_future = points[num_time + 20:num_time + 20 + pred_len].copy()
        id = vehicle['TRACK_ID'].values[0]
        temp_past = temp_past - origin
        temp_future = temp_future - origin

        self.pasts = torch.FloatTensor(temp_past)
        self.presents = torch.FloatTensor(origin)
        self.track_id = id
        self.now_frame = now_frame

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        return self.pasts[idx], self.futures[idx], self.presents[idx]


class NoFutureDataset(data.Dataset):
    def __init__(self, past):
        self.pasts = past

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        return self.pasts[idx]


class Validator():
    def __init__(self, config, vehicle_list):
        ################################
        data_type = config.type
        self.load_memory = config.saved_memory
        self.future_len = config.future_length
        specific_frame = config.frame
        data_path = config.data_path
        ################################

        self.pasts = []
        self.presents = []
        self.futures = []

        data_val_list = []
        all_txt_list = []
        for filename in sorted(os.listdir(data_path + '/bbox/front/')):
            all_txt_list.append(
                int(filename.split(".")[0]))
        bbox_time_list = np.array(all_txt_list)
        bbox_first_frame = np.min(bbox_time_list)

        first_frame = vehicle_list[0]['FRAME'].iloc[0]
        for val_vehicle_num in range(len(vehicle_list)):
            for real_frame in range(1):
                if specific_frame - bbox_first_frame >= 20:
                    data_val_list.append(TrackDataset(train=False, name='scenario_name', weather_name='weather_type',
                                                      num_time=0, vehicle=vehicle_list[val_vehicle_num], f_frame=first_frame, now_frame=int(specific_frame - 20), pred_len=self.future_len))
        #self.data_val = np.array(data_val_list)
        self.data_val = data_val_list
        val_pasts = []
        for i in range(len(self.data_val)):  # i th file
            val_pasts.append(
                np.array(data_val_list[i].pasts[0] - data_val_list[i].pasts[0][-1]))
        val_pasts = torch.FloatTensor(val_pasts)
        self.batch_val = NoFutureDataset(
            past=val_pasts)
        self.val_loader = DataLoader(
            self.batch_val, batch_size=1, num_workers=0, shuffle=False)
        print('dataset created')
        # load model to evaluate
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = 1
        self.mem_n2n.future_len = config.future_length
        self.mem_n2n.past_len = 20
        self.EuclDistance = nn.PairwiseDistance(p=2)
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
            self.mem_n2n.share_memory()
        self.start_epoch = 0
        self.config = config

    def test_model(self):
        """
        Memory selection and evaluation!
        :return: None
        """
        self._memory_writing(self.config.saved_memory)
        all_prediction = self.evaluate(
            self.val_loader).cpu()
        offset = 0
        df_list = []
        for i in range(len(self.data_val)):
            origin = self.data_val[i].presents
            pasts = self.data_val[i].pasts
            prediction = all_prediction[offset]
            offset += 1

            for t in range(self.future_len + 20):
                if t < 20:
                    x = float(pasts[t][0] + origin[0])
                    y = float(pasts[t][1] + origin[1])
                    lis = [self.data_val[i].now_frame +
                           t, self.data_val[i].track_id, x, y]
                    df_list.append(lis)
                    # list = [str(self.data_val[i][j][k].now_frame + t), '\t', str(
                    #    self.data_val[i][j][k].track_id), '\t', str(x), '\t', str(y), '\n']
                    # f.writelines(list)
                else:
                    pred_x = float(
                        prediction[0][t - 20][0] + origin[0])
                    pred_y = float(
                        prediction[0][t - 20][1] + origin[1])
                    lis = [self.data_val[i].now_frame +
                           t, self.data_val[i].track_id, pred_x, pred_y]
                    df_list.append(lis)
                    # list = [str(self.data_val[i][j][k].now_frame + t), '\t', str(
                    #    self.data_val[i][j][k].track_id), '\t', str(pred_x), '\t', str(pred_y), '\n']
                    # f.writelines(list)
        df = pd.DataFrame(df_list, columns=['FRAME', 'TRACK_ID', 'X', 'Y'])
        return df

    def evaluate(self, loader):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :return: dictionary of performance metrics
        """
        all_pred = torch.tensor([]).cuda()
        with torch.no_grad():
            for step, (past) in enumerate(loader):
                past = Variable(past)
                if self.config.cuda:
                    past = past.cuda()
                pred = self.mem_n2n(past.unsqueeze(0))
                all_pred = torch.cat((all_pred, pred), axis=0)
        return all_pred

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        if self.config.saved_memory:
            print("memory_loading")
            self.mem_n2n.memory_past = torch.load(
                self.config.memories_path + 'memory_past.pt')
            self.mem_n2n.memory_fut = torch.load(
                self.config.memories_path + 'memory_fut.pt')


def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle


def obstacle_collision(car_length, car_width, obs_length, obs_width, ego_x, ego_y, ego_x_next, ego_y_next, obs_x, obs_y, obs_yaw, vehicle_length, vehicle_width, specific_frame, pred_t, now_id):
    center_distance_vector = [obs_x - ego_x, obs_y - ego_y]
    ego_vector_square = math.sqrt(
        (ego_x_next - ego_x)**2 + (ego_y_next - ego_y)**2)
    ego_cos = (ego_x_next - ego_x) / ego_vector_square
    ego_sin = (ego_y_next - ego_y) / ego_vector_square
    ego_axisX_1 = ego_cos * car_length / 2
    ego_axisY_1 = ego_sin * car_length / 2
    ego_axisX_2 = ego_sin * car_width / 2
    ego_axisY_2 = - ego_cos * car_width / 2
    obs_yaw = (float(obs_yaw) + 90.0) * np.pi / 180
    obs_rec = [obs_x, obs_y, obs_width, obs_length, obs_yaw]
    obs_cos = math.cos(obs_yaw)
    obs_sin = math.sin(obs_yaw)
    obs_axisX_1 = obs_cos * obs_length / 2
    obs_axisY_1 = obs_sin * obs_length / 2
    obs_axisX_2 = obs_sin * obs_width / 2
    obs_axisY_2 = - obs_cos * obs_width / 2
    if abs(center_distance_vector[0] * obs_cos + center_distance_vector[1] * obs_cos) <=\
            abs(ego_axisX_1 * obs_cos + ego_axisY_1 * obs_sin) + abs(ego_axisX_2 * obs_cos + ego_axisY_2 * obs_sin) + vehicle_length / 2 and \
            abs(center_distance_vector[0] * obs_sin - center_distance_vector[1] * obs_cos) <=\
            abs(ego_axisX_1 * obs_cos - ego_axisY_1 * obs_cos) + abs(ego_axisX_2 * obs_sin - ego_axisY_2 * obs_cos) + vehicle_width / 2 and \
            abs(center_distance_vector[0] * ego_cos + center_distance_vector[1] * ego_sin) <=\
            abs(obs_axisX_1 * ego_cos + obs_axisY_1 * ego_sin) + abs(obs_axisX_2 * ego_cos + obs_axisY_2 * ego_sin) + vehicle_length / 2 and \
            abs(center_distance_vector[0] * ego_sin - center_distance_vector[1] * ego_cos) <=\
            abs(obs_axisX_1 * ego_cos - obs_axisY_1 * ego_cos) + abs(obs_axisX_2 * ego_sin + obs_axisY_2 * ego_cos) + vehicle_width / 2:
        return [specific_frame, int(now_id)]


def main(config):
    data_path = config.data_path
    data_type = config.type
    specific_frame = config.frame
    future_len = config.future_length
    scenario_name = '10_i-1_1_c_f_f_1_rl'

    # 設定車輛 行人 障礙物面積

    vehicle_length = 4.7
    vehicle_width = 2
    pedestrian_length = 0.8
    pedestrian_width = 0.8
    agent_area = [[vehicle_length, vehicle_width],
                  [pedestrian_length, pedestrian_width]]
    trafficcone_width = 0.85
    barrier_length = 1.25
    barrier_width = 0.375
    warning_length = 3
    warning_width = 2.33
    object_type_list = ['static.prop.trafficcone01',
                        'static.prop.streetbarrier', 'static.prop.trafficwarning', 'vehicle']
    object_area = [[trafficcone_width, trafficcone_width], [barrier_length, barrier_width], [
        warning_length, warning_width], [vehicle_length, vehicle_width]]

    # 由於raw_data看不出ego car是誰，所以這邊用dynamic_description.json找出那個ID，並在dataframe中把他的type改成EGO，事後用這個找
    # 但如果你有ego car id，可以直接用traj_df.loc[ #EGO_ID == variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
    traj_df = pd.read_csv(data_path + '/trajectory_frame/test.csv')
    with open(data_path + '/dynamic_description.json') as f:
        data = json.load(f)
        variant_ego_id = str(data['player'])
        traj_df.loc[traj_df.TRACK_ID ==
                    variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
    # 偶爾沒有obstacle資料夾 偶爾沒有obstacle_list 偶爾obstacle_list是空的
    # 若要即時蒐集obstacle 此段可刪
    has_obstacle = True
    if not os.path.exists(data_path + '/obstacle'):
        has_obstacle = False
    elif not os.path.exists(data_path + '/obstacle/obstacle_list.txt'):
        has_obstacle = False
    else:
        if os.path.getsize(data_path + '/obstacle/obstacle_list.txt') == 0:
            has_obstacle = False
    # 自己幫Obstacle_list取ID
    # 若要即時蒐集obstacle 此段需改 right_id_obj作為即時看到的obstacle資訊 要和即時看到的vehicle資訊concat在一起
    # ex: vehicle資訊可能是
    # FRAME TRACK_ID OBJECT_TYPE X Y
    # 50    12345   vehicle     0   0
    # 50    23456   pedestrian     1   1
    # 當看到新的obstacle id 需要取出他的X Y 做成和上述相同的格式 (obstacle有transform可以取出X跟Y)
    # ex: obstacle資訊可能是
    # FRAME TRACK_ID OBJECT_TYPE X Y
    # 50    56789   static.prop.trafficcone01     2   2
    # 50    45678   static.prop.trafficwarning     3   3
    # 之後跟obstacle對ID有關的部分就不需要了
    if has_obstacle and data_type == 'obstacle':
        object_type_list = [
            'static.prop.trafficcone01', 'static.prop.streetbarrier', 'static.prop.trafficwarning', 'vehicle']
        all_object_num = 0
        class_num = np.array([0, 0, 0, 0])
        class_0_pos = []
        class_1_pos = []
        class_2_pos = []
        class_3_pos = []
        class_all_pos = []
        obj_dict = defaultdict(list)
        with open(os.path.join(data_path + '/obstacle/', 'obstacle_list.txt')) as f:
            for line in f.readlines():
                all_object_num += 1
                s = line.split('\t')
                object_name = s[0]
                pos = s[1].split(',')
                x = (pos[0].split('='))[1]
                y = (pos[1].split('='))[1]
                z_t = pos[2].split('=')
                z = (z_t[1].split(')'))[0]
                if object_name == object_type_list[0]:
                    object_id = 100000 + class_num[0]
                    class_num[0] += 1
                    class_0_pos.append((x, y, z))
                elif object_name == object_type_list[1]:
                    object_id = 200000 + class_num[1]
                    class_num[1] += 1
                    class_1_pos.append((x, y, z))
                elif object_name == object_type_list[2]:
                    object_id = 300000 + class_num[2]
                    class_num[2] += 1
                    class_2_pos.append((x, y, z))
                elif (object_name.split('.'))[0] == object_type_list[3]:
                    object_id = 400000 + class_num[3]
                    class_num[3] += 1
                    class_3_pos.append((x, y, z))
            class_all_pos.append(class_0_pos)
            class_all_pos.append(class_1_pos)
            class_all_pos.append(class_2_pos)
            class_all_pos.append(class_3_pos)
            mid_time_index = int(
                len(pd.unique(traj_df['FRAME'].values)) / 2)
            mid_time = pd.unique(traj_df['FRAME'].values)[
                mid_time_index]
            mid_mask = (traj_df.FRAME == mid_time) & (
                traj_df.OBJECT_TYPE == 'AGENT')
            mid_pos_x = float(traj_df[mid_mask]['X'])
            mid_pos_y = float(traj_df[mid_mask]['Y'])
            for t in range(len(class_all_pos)):
                for i in range(class_num[t]):
                    for j in pd.unique(traj_df['FRAME'].values):
                        dist_x = float(
                            class_all_pos[t][i][0]) - mid_pos_x
                        dist_y = float(
                            class_all_pos[t][i][1]) - mid_pos_y
                        if abs(dist_x) <= 50 and abs(dist_y) <= 50:
                            obj_dict['FRAME'].append(j)
                            obj_dict['TRACK_ID'].append(
                                100000 * (t + 1) + i)
                            obj_dict['OBJECT_TYPE'].append(
                                object_type_list[t])
                            obj_dict['X'].append(
                                class_all_pos[t][i][0])
                            obj_dict['Y'].append(
                                class_all_pos[t][i][1])
                            obj_dict['CITY_NAME'].append(
                                traj_df['CITY_NAME'].values[0])
            obj_df = pd.DataFrame(obj_dict)
            if scenario_name.split('_')[0] != '5' and scenario_name.split('_')[0] != '10':
                print("No GT")
                return False
            with open(data_path + '/obstacle_info.json') as f:
                obj_data = json.load(f)
                obj_d = obj_data['interactive_obstacle_list']
            right_id_obj = pd.DataFrame()
            f = open(
                data_path + '/GT.txt', 'w')
            for track_id, remain_df in obj_df.groupby('TRACK_ID'):
                for item in obj_d.items():
                    dis_x = float(remain_df.X.values[0]) - \
                        item[1]['transform'][0]
                    dis_y = float(remain_df.Y.values[0]) - \
                        item[1]['transform'][1]
                    if abs(dis_x) < 1 and abs(dis_y) < 1:
                        lines = [
                            str(remain_df.TRACK_ID.values[0]), '\t', str(item[0]), '\n']
                        f.writelines(lines)
                        remain_df.loc[:, 'TRACK_ID'] = item[0]
                        break
                right_id_obj = pd.concat(
                    [right_id_obj, remain_df], axis=0)
            f.close()
            traj_df = pd.concat([traj_df, right_id_obj])
            traj_df.reset_index(drop=True)
    right_id_obj_list = []

    # 這是找出obstacle GT的ID
    # 若要即時蒐集obstacle 此段可刪
    if data_type == 'obstacle':
        with open(data_path + '/obstacle_info.json') as f:
            obj_data = json.load(f)
            obj_d = obj_data['interactive_obstacle_list']
            # label by 邦元
            #last_frame = obj_data['interactive frame'][1]
        for item in obj_d.items():
            right_id_obj_list.append(item[0])

    risky_vehicle_list = []
    # vehicle_id_list儲存是車輛的ID，pedestrian_id_list儲存行人的ID，需要這個是因為車輛和行人的面積不同
    vehicle_id_list = []
    pedestrian_id_list = []
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df.OBJECT_TYPE.values[0] == 'vehicle':
            vehicle_id_list.append(remain_df.TRACK_ID.values[0])
        elif remain_df.OBJECT_TYPE.values[0] == 'pedestrian':
            pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
    all_txt_list = []
    # 從bbox的第一個資料時間點當作起始的frame，至少需要起始frame + 20(也就是有20個frame數的資料)，才能做預測
    for filename in sorted(os.listdir(data_path + '/bbox/front/')):
        all_txt_list.append(
            int(filename.split(".")[0]))
    bbox_time_list = np.array(all_txt_list)
    bbox_first_frame = np.min(bbox_time_list)
    if specific_frame - bbox_first_frame < 20:
        print("There is no enough data")
        return False
    #print(bbox_first_frame, vehicle_list[0])

    traj_df['X'] = traj_df['X'].astype(float)
    traj_df['Y'] = traj_df['Y'].astype(float)
    vehicle_list = []
    filter = (traj_df.OBJECT_TYPE != ('AGENT'))
    traj_df = traj_df[filter].reset_index(drop=True)
    filter = (traj_df.OBJECT_TYPE != ('actor.vehicle'))
    traj_df = traj_df[filter].reset_index(drop=True)
    filter = (traj_df.OBJECT_TYPE != ('actor.pedestrian'))
    traj_df = traj_df[filter].reset_index(drop=True)

    all_txt_list = []
    for filename in sorted(os.listdir(data_path + '/bbox/front/')):
        all_txt_list.append(
            int(filename.split(".")[0]))
    bbox_time_list = np.array(all_txt_list)
    bbox_first_frame = np.min(bbox_time_list)
    filter = (traj_df.FRAME >= int(bbox_first_frame))
    traj_df = traj_df[filter].reset_index(drop=True)

    mid_time_index = int(
        len(pd.unique(traj_df['FRAME'].values)) / 2)
    mid_time = pd.unique(traj_df['FRAME'].values)[mid_time_index]
    mid_mask = (traj_df.FRAME == specific_frame) & (
        traj_df.OBJECT_TYPE == 'EGO')
    mid_pos_x = float(traj_df[mid_mask]['X'])
    mid_pos_y = float(traj_df[mid_mask]['Y'])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        filter = (remain_df.FRAME > (
            specific_frame - 20))
        remain_df = remain_df[filter].reset_index(drop=True)
        remain_df = remain_df.reset_index(drop=True)
        actor_pos_x = float(remain_df.loc[20, 'X'])
        actor_pos_y = float(remain_df.loc[20, 'Y'])
        dist_x = actor_pos_x - mid_pos_x
        dist_y = actor_pos_y - mid_pos_y
        if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
            vehicle_list.append(remain_df)

    # 準備好資料之後，丟進mantra的模型
    # 我把丟進mantra模型的地方，改成跟kalman_filter一樣的vehicle_list
    v = Validator(config, vehicle_list)
    print('start evaluation')
    temp_df = v.test_model()

    # 預測結束，把結果拿去計算未來會不會碰撞
    vehicle_list = []
    for track_id, remain_df in temp_df.groupby('TRACK_ID'):
        vehicle_list.append(remain_df)
    ego_prediction = np.zeros((future_len, 2))
    for n in range(len(vehicle_list)):
        vl = vehicle_list[n].to_numpy()
        now_id = vl[0][1]
        if int(now_id) == int(variant_ego_id):
            ego_now_pos_x = vl[0][2]
            ego_now_pos_y = vl[0][3]
            for pred_t in range(future_len - 1):
                real_pred_x = vl[pred_t + 20][2]
                real_pred_x_next = vl[pred_t + 21][2]
                real_pred_y = vl[pred_t + 20][3]
                real_pred_y_next = vl[pred_t + 21][3]
                ego_prediction[pred_t][0] = real_pred_x
                ego_prediction[pred_t][1] = real_pred_y
                if pred_t == int(future_len - 2):
                    ego_prediction[pred_t +
                                   1][0] = real_pred_x_next
                    ego_prediction[pred_t +
                                   1][1] = real_pred_y_next
    for n in range(len(vehicle_list)):
        #ego, agent, other = 0, 0, 0
        vl = vehicle_list[n].to_numpy()
        # vl : frame, id, x, y
        now_id = vl[0][1]
        if str(int(now_id)) in pedestrian_id_list:
            agent_type = 1
        elif str(int(now_id)) in vehicle_id_list:
            agent_type = 0
        elif int(now_id) == int(variant_ego_id):
            agent_type = 0

        actor_pos_x = vl[0][2]
        actor_pos_y = vl[0][3]
        dist_x = actor_pos_x - ego_now_pos_x
        dist_y = actor_pos_y - ego_now_pos_y
        if abs(dist_x) > 37.5 or abs(dist_y) > 37.5:
            continue
        for pred_t in range(future_len - 1):
            if int(now_id) == int(variant_ego_id):
                continue
            real_pred_x = vl[pred_t + 20][2]
            real_pred_x_next = vl[pred_t + 21][2]
            real_pred_y = vl[pred_t + 20][3]
            real_pred_y_next = vl[pred_t + 21][3]

            # 這個環節是為了從ID中找出那個障礙物是哪一種類別(如柵欄或是交通錐)，要找這個是因為每個障礙物面積都不一樣，為了帶進函式obstacle_collision去計算是否碰撞
            # 若即時拿obstacle資料，可以直接拿type是哪一種，跟object_type_list對照找出對應的面積
            if data_type == 'obstacle' and (int(now_id) >= 100000 or str(now_id) in right_id_obj_list):
                obs_type = int(now_id) / 100000 - 1
                obs_index = int(now_id) % 100000
                if str(now_id) in right_id_obj_list:
                    with open(data_path + '/GT.txt') as f:
                        for line in f.readlines():
                            s = line.split('\t')
                            if int(now_id) == int(s[1]):
                                obs_type = int(
                                    s[0]) / 100000 - 1
                                obs_index = int(
                                    s[0]) % 100000
                # 這個環節是從obstacle_list去拿出障礙物的yaw(因為計算碰撞需要障礙物角度)，要找這個是因為當初存資料沒有考慮到yaw
                # 若即時拿obstacle資料，可以從obstacle的transform拿到yaw
                with open(os.path.join(data_path + data_type + '/' + scenario_name + '/obstacle/', 'obstacle_list.txt')) as f:
                    count = -1
                    for line in f.readlines():
                        s = line.split('\t')
                        object_name = s[0]
                        if int(obs_type) == 3:
                            object_name = object_name.split('.')[
                                0]
                        if object_name != object_type_list[int(obs_type)]:
                            continue
                        count += 1
                        if int(obs_index) == count:
                            pos = s[1].split(',')
                            yaw = (pos[4].split('='))[1]
                            x = (pos[0].split('='))[1]
                            y = (pos[1].split('='))[1]

                temp = obstacle_collision(vehicle_length, vehicle_width, object_area[int(obs_type)][0], object_area[int(
                    obs_type)][1], ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], yaw, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                if temp != None:
                    risky_vehicle_list.append(temp)
            else:
                # 這邊是計算和其他車輛有沒有碰撞
                center_distance_vector = [
                    real_pred_x - ego_prediction[pred_t][0], real_pred_y - ego_prediction[pred_t][1]]
                vehicle_vector_square = math.sqrt(
                    (real_pred_x_next - real_pred_x)**2 + (real_pred_y_next - real_pred_y)**2)
                vehicle_cos = (real_pred_x_next -
                               real_pred_x) / vehicle_vector_square
                vehicle_sin = (real_pred_y_next -
                               real_pred_y) / vehicle_vector_square
                vehicle_axisX_1 = vehicle_cos * \
                    agent_area[agent_type][0] / 2
                vehicle_axisY_1 = vehicle_sin * \
                    agent_area[agent_type][0] / 2
                vehicle_axisX_2 = vehicle_sin * \
                    agent_area[agent_type][1] / 2
                vehicle_axisY_2 = - vehicle_cos * \
                    agent_area[agent_type][1] / 2
                ego_vector_square = math.sqrt((ego_prediction[pred_t + 1][0] - ego_prediction[pred_t][0])**2 + (
                    ego_prediction[pred_t + 1][1] - ego_prediction[pred_t][1])**2)
                ego_cos = (
                    ego_prediction[pred_t + 1][0] - ego_prediction[pred_t][0]) / ego_vector_square
                ego_sin = (
                    ego_prediction[pred_t + 1][1] - ego_prediction[pred_t][1]) / ego_vector_square
                ego_axisX_1 = ego_cos * \
                    agent_area[agent_type][0] / 2
                ego_axisY_1 = ego_sin * \
                    agent_area[agent_type][0] / 2
                ego_axisX_2 = ego_sin * \
                    agent_area[agent_type][1] / 2
                ego_axisY_2 = - ego_cos * \
                    agent_area[agent_type][1] / 2
                if abs(center_distance_vector[0] * vehicle_cos + center_distance_vector[1] * vehicle_cos) <=\
                        abs(ego_axisX_1 * vehicle_cos + ego_axisY_1 * vehicle_sin) + abs(ego_axisX_2 * vehicle_cos + ego_axisY_2 * vehicle_sin) + agent_area[agent_type][0] / 2 and \
                        abs(center_distance_vector[0] * vehicle_sin - center_distance_vector[1] * vehicle_cos) <=\
                        abs(ego_axisX_1 * vehicle_cos - ego_axisY_1 * vehicle_cos) + abs(ego_axisX_2 * vehicle_sin - ego_axisY_2 * vehicle_cos) + agent_area[agent_type][1] / 2 and \
                        abs(center_distance_vector[0] * ego_cos + center_distance_vector[1] * ego_sin) <=\
                        abs(vehicle_axisX_1 * ego_cos + vehicle_axisY_1 * ego_sin) + abs(vehicle_axisX_2 * ego_cos + vehicle_axisY_2 * ego_sin) + agent_area[agent_type][0] / 2 and \
                        abs(center_distance_vector[0] * ego_sin - center_distance_vector[1] * ego_cos) <=\
                        abs(vehicle_axisX_1 * ego_cos - vehicle_axisY_1 * ego_cos) + abs(vehicle_axisX_2 * ego_sin + vehicle_axisY_2 * ego_cos) + agent_area[agent_type][1] / 2:
                    risky_vehicle_list.append(
                        [specific_frame, int(vl[0][1])])

    # 這個環節是統整出當下有多少risky id，會給出一個list，可能有複數個
    file_d = {}
    for frame, id in risky_vehicle_list:
        if frame in file_d:
            if str(id) in file_d[frame]:
                continue
            else:
                file_d[frame].append(str(id))
        else:
            file_d[frame] = [str(id)]
    if specific_frame in file_d:
        risky_id = file_d[specific_frame]
    else:
        risky_id = None
    # print(file_d)
    print(scenario_name,
          ' risky_id:',  risky_id)


if __name__ == '__main__':
    # python mantra_one_step.py --type interactive --future_length 30 --frame 157

    # For interactive test data, 157 can find risky id (10_i-1_1_c_f_f_1_rl)
    config = parse_config()
    main(config)
