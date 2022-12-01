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
    def __init__(self, vehicle_list, frame):
        ################################
        data_type =  "interactive"#config.type
        self.load_memory = True #config.saved_memory
        self.future_len = 30 #config.future_length
        specific_frame = frame #config.frame
        # data_path = config.data_path
        ################################

        self.pasts = []
        self.presents = []
        self.futures = []

        data_val_list = []
        all_txt_list = []

        # for filename in sorted(os.listdir(data_path + '/bbox/front/')):
        #     all_txt_list.append(
        #         int(filename.split(".")[0]))
        # bbox_time_list = np.array(all_txt_list)
        # bbox_first_frame = np.min(bbox_time_list)

        #first_frame = vehicle_list[0]['FRAME'].iloc[0]
        for val_vehicle_num in range(len(vehicle_list)):
            #for real_frame in range(1):
            #if specific_frame - bbox_first_frame >= 20:
            data_val_list.append(TrackDataset(train=False, name='scenario_name', weather_name='weather_type',
                                                    num_time=0, vehicle=vehicle_list[val_vehicle_num], f_frame=0, now_frame=int(specific_frame - 20), pred_len=self.future_len))
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

        import gdown
        if not os.path.exists("./mantra/"):
            os.mkdir("./mantra/")

        if not os.path.isfile("./mantra/model_controller"):
            print("Download mantra weight")
            url = "https://drive.google.com/u/4/uc?id=1wiC0P5Idc3p6Pjl_6uNBcjBslc8-Z3_F&export=download"
            gdown.download(url, "./mantra/model_controller")

        self.mem_n2n = torch.load("./mantra/model_controller")


        self.mem_n2n.num_prediction = 1
        self.mem_n2n.future_len = 30 #config.future_length
        self.mem_n2n.past_len = 20
        self.EuclDistance = nn.PairwiseDistance(p=2)
        if True: #config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
            self.mem_n2n.share_memory()
        self.start_epoch = 0
        # self.config = config

    def test_model(self):
        """
        Memory selection and evaluation!
        :return: None
        """
        self._memory_writing(True) #self.config.saved_memory)
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
                if True: # self.config.cuda:
                    past = past.cuda()
                pred = self.mem_n2n(past.unsqueeze(0))
                all_pred = torch.cat((all_pred, pred), axis=0)
        return all_pred

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        if True: #self.config.saved_memory:
            import gdown
            if not os.path.exists('./mantra/carla_dataset_all/'):
                os.mkdir('./mantra/carla_dataset_all/')

            if not os.path.isfile('./mantra/carla_dataset_all/memory_past.pt'):
                print("Download mantra memory_past weight")
                url = "https://drive.google.com/u/0/uc?id=1Kn7JrIkgV0bExfb2ljD6324CtxOHBZ3L&export=download"
                gdown.download(url, './mantra/carla_dataset_all/memory_past.pt')
            if not os.path.isfile('./mantra/carla_dataset_all/memory_fut.pt'):
                print("Download mantra memory_fut weight")
                url = "https://drive.google.com/u/0/uc?id=1DGGxG_23WuHCNr3KqXbFBo3-NobqkxTe&export=download"
                gdown.download(url, './mantra/carla_dataset_all/memory_fut.pt')

            print("memory_loading")
            self.mem_n2n.memory_past = torch.load( './mantra/carla_dataset_all/memory_past.pt' ) 
            self.mem_n2n.memory_fut = torch.load( './mantra/carla_dataset_all/memory_fut.pt')


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


def mantra_inference(vehicle_list, specific_frame, variant_ego_id, pedestrian_id_list, vehicle_id_list ):

    vehicle_length = 4.7
    vehicle_width = 2
    pedestrian_length = 0.8
    pedestrian_width = 0.8
    agent_area = [[vehicle_length, vehicle_width],
                [pedestrian_length, pedestrian_width]]




    #config
    future_len = 20

    v = Validator(vehicle_list,  specific_frame )
    print('start evaluation')
    temp_df = v.test_model()

    vehicle_list = []
    for track_id, remain_df in temp_df.groupby('TRACK_ID'):
        vehicle_list.append(remain_df)

    risky_vehicle_list = []
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

    agent_type = 0
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
        risky_id = []

    return risky_id