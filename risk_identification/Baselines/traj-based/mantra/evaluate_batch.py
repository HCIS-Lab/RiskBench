from mmap import PROT_READ
import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import datetime
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_invariance
import index_qualitative
from torch.autograd import Variable
import csv
import time
import tqdm
import pdb
import pandas as pd
import matplotlib.image as mpimg
import re
import mantra_evaluate
import mantra
import math
import sys

from torchvision import models
from thop.profile import profile

from torchsummary import summary


class Validator():
    def __init__(self, config):
        """
        class to evaluate Memnet
        :param config: configuration parameters (see test.py)
        """
        self.data_type = config.data_type
        self.load_memory = config.saved_memory
        self.future_len = config.future_len
        self.memories_path = config.memories_path
        if config.evaluate_or_inference == 'evaluate':
            self.metric_or_predict = 'metric'
        else:
            self.metric_or_predict = 'predict'
        self.val_or_test = config.val_or_test
        self.load_non_interactive = False
        if self.metric_or_predict == 'metric':
            residual_frame = self.future_len + 20
        elif self.metric_or_predict == 'predict':
            residual_frame = 20
        self.dict_metrics = {}
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test/' + self.name_test + '_' + \
            config.info + '_' + self.data_type + '_' + \
            self.val_or_test + '_' + \
            str(float(self.future_len / 20)) + 's_' + \
            self.metric_or_predict + '/' + self.data_type
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        print('creating dataset...')
        #TRAIN_DIR = 'data_carla_risk_all/train'
        TRAIN_DIR = os.path.join(config.dataset_file, 'train')
        if self.val_or_test == 'val':
            VAL_DIR = os.path.join(config.dataset_file, 'val', self.data_type)
            #VAL_DIR = 'data_carla_risk_all/val/' + self.data_type
        elif self.val_or_test == 'test':
            VAL_DIR = os.path.join(config.dataset_file, 'test', self.data_type)
            #VAL_DIR = 'data_carla_risk_all/test/' + self.data_type
        self.pasts = []
        self.presents = []
        self.futures = []

        if not self.load_memory:
            dir_list = os.listdir(TRAIN_DIR)
            for type in dir_list:
                for scenario_name in os.listdir(TRAIN_DIR + '/' + type):
                    print("write_memory", scenario_name)
                    weather_file = TRAIN_DIR + '/' + type + \
                        '/' + scenario_name + '/variant_scenario'
                    for weather_type in os.listdir(weather_file):
                        traj_df = pd.read_csv(
                            weather_file + '/' + weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                        vehicle_list = []
                        filter = (traj_df.OBJECT_TYPE != ('AGENT'))
                        traj_df = traj_df[filter].reset_index(drop=True)
                        filter = (traj_df.OBJECT_TYPE != ('actor.vehicle'))
                        traj_df = traj_df[filter].reset_index(drop=True)
                        filter = (traj_df.OBJECT_TYPE != ('actor.pedestrian'))
                        traj_df = traj_df[filter].reset_index(drop=True)
                        for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
                            if obj_type == 'EGO':
                                vehicle_list.append(remain_df)
                        len_time = len(vehicle_list[0]['FRAME'])
                        data_train_vehicle_list = []
                        for train_vehicle_num in range(len(vehicle_list)):
                            vehicle = vehicle_list[train_vehicle_num]
                            points = np.vstack((vehicle['X'], vehicle['Y'])).T

                            for t in range(len_time):
                                if len_time - t > 50:
                                    temp_past = points[t:t + 20].copy()
                                    temp_future = points[t + 20:t + 50].copy()
                                    origin = temp_past[-1]
                                    temp_past = temp_past - origin
                                    temp_future = temp_future - origin
                                    unit_y_axis = torch.Tensor([0, -1])
                                    vector = temp_past[-5]
                                    if int(vector[0]) == 0:
                                        angle = 0
                                    elif vector[0] > 0.0:
                                        angle = np.rad2deg(
                                            mantra.angle_vectors(vector, unit_y_axis))
                                    else:
                                        angle = - \
                                            np.rad2deg(mantra.angle_vectors(
                                                vector, unit_y_axis))
                                    matRot_track = cv2.getRotationMatrix2D(
                                        (0, 0), angle, 1)
                                    past_rot = cv2.transform(
                                        temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
                                    future_rot = cv2.transform(
                                        temp_future.reshape(-1, 1, 2), matRot_track).squeeze()
                                    self.pasts.append(past_rot)
                                    self.futures.append(future_rot)
                                    self.presents.append(origin)
            self.pasts = torch.FloatTensor(self.pasts)
            self.futures = torch.FloatTensor(self.futures)
            self.presents = torch.FloatTensor(self.presents)
            self.data_train = mantra.CarlaDataset(
                past=self.pasts, future=self.futures, present=self.presents)
            self.train_loader_one_array = DataLoader(
                self.data_train, batch_size=512, num_workers=16, shuffle=True)
        data_val_list = []
        Val_List = sorted(os.listdir(VAL_DIR))
        for scenario_name in Val_List:

            if str(self.future_len) == '60' and scenario_name == '5_i-1_0_0_0_r_0_0':
                continue
            if str(self.future_len) == '60' and scenario_name == '5_i-7_1_0_0_l_0_0':
                continue
            if str(self.future_len) == '60' and scenario_name == '5_t1-2_0_0_0_r_0_0':
                continue
            if str(self.future_len) == '60' and scenario_name == '5_t2-2_0_c_u_f_0':
                continue
            if str(self.future_len) == '60' and scenario_name == '5_t2-7_0_c_r_f_0':
                continue
            if str(self.future_len) == '60' and scenario_name == '5_t3-2_0_c_u_f_0':
                continue
            weather_file = VAL_DIR + '/' + scenario_name + '/variant_scenario'
            for weather_type in sorted(os.listdir(weather_file)):
                if str(self.future_len) == '60' and scenario_name == '5_s-9_0_m_sl_sr_0_1' and weather_type == 'WetCloudySunset_mid_':
                    continue
                print(scenario_name, weather_type)
                traj_df = pd.read_csv(
                    weather_file + '/' + weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                vehicle_list = []
                filter = (traj_df.OBJECT_TYPE != ('AGENT'))
                traj_df = traj_df[filter].reset_index(drop=True)
                filter = (traj_df.OBJECT_TYPE != ('actor.vehicle'))
                traj_df = traj_df[filter].reset_index(drop=True)
                filter = (traj_df.OBJECT_TYPE != ('actor.pedestrian'))
                traj_df = traj_df[filter].reset_index(drop=True)

                all_txt_list = []
                for filename in sorted(os.listdir(VAL_DIR + '/' + self.data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/')):
                    all_txt_list.append(
                        int(filename.split(".")[0]))
                bbox_time_list = np.array(all_txt_list)
                bbox_first_frame = np.min(bbox_time_list)
                filter = (traj_df.FRAME >= int(bbox_first_frame))
                traj_df = traj_df[filter].reset_index(drop=True)

                mid_time_index = int(
                    len(pd.unique(traj_df['FRAME'].values)) / 2)
                mid_time = pd.unique(traj_df['FRAME'].values)[mid_time_index]
                mid_mask = (traj_df.FRAME == mid_time) & (
                    traj_df.OBJECT_TYPE == 'EGO')
                mid_pos_x = float(traj_df[mid_mask]['X'])
                mid_pos_y = float(traj_df[mid_mask]['Y'])
                for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                    remain_df = remain_df.reset_index(drop=True)
                    actor_pos_x = remain_df.loc[mid_time_index, 'X']
                    actor_pos_y = remain_df.loc[mid_time_index, 'Y']
                    dist_x = actor_pos_x - mid_pos_x
                    dist_y = actor_pos_y - mid_pos_y
                    if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                        vehicle_list.append(remain_df)
                first_frame = vehicle_list[0]['FRAME'].iloc[0]
                len_time = len(vehicle_list[0]['FRAME'])
                frame_filter = (traj_df.OBJECT_TYPE == ('EGO'))
                frame_df = traj_df[frame_filter].reset_index(drop=True)
                data_val_vehicle_list = []
                for val_vehicle_num in range(len(vehicle_list)):
                    data_val_time_list = []
                    val_time_num = 0
                    for real_frame in frame_df['FRAME'].values:
                        if len_time - val_time_num > residual_frame:
                            data_val_time_list.append(mantra_evaluate.TrackDataset(train=False, name=scenario_name, weather_name=weather_type,
                                                      num_time=val_time_num, vehicle=vehicle_list[val_vehicle_num], f_frame=first_frame, now_frame=real_frame, pred_len=self.future_len))
                            val_time_num += 1
                    data_val_vehicle_list.append(data_val_time_list)
                data_val_list.append(data_val_vehicle_list)
        self.data_val = np.array(data_val_list)
        if self.metric_or_predict == 'metric':
            val_pasts = []
            val_futures = []
            val_presents = []
            for i in range(len(self.data_val)):  # i th file
                for k in range(len(self.data_val[i][0])):  # j th vechicle
                    for j in range(len(self.data_val[i])):  # k th second
                        origin = self.data_val[i][j][k].pasts[0][-1]
                        val_pasts.append(
                            np.array(self.data_val[i][j][k].pasts[0]))
                        val_futures.append(
                            np.array(self.data_val[i][j][k].futures[0]))
                        val_presents.append(
                            np.array(self.data_val[i][j][k].presents[0]))
            val_pasts = torch.FloatTensor(val_pasts)
            val_futures = torch.FloatTensor(val_futures)
            val_presents = torch.FloatTensor(val_presents)
            self.batch_val = mantra.CarlaDataset(
                past=val_pasts, future=val_futures, present=val_presents)
            self.val_loader = DataLoader(
                self.batch_val, batch_size=256, num_workers=12, shuffle=False)
        elif self.metric_or_predict == 'predict':
            val_pasts = []
            val_futures = []
            val_presents = []
            for i in range(len(self.data_val)):  # i th file
                for k in range(len(self.data_val[i][0])):  # j th vechicle
                    for j in range(len(self.data_val[i])):  # k th second
                        origin = self.data_val[i][j][k].pasts[0][-1]
                        val_pasts.append(
                            np.array(self.data_val[i][j][k].pasts[0] - origin))
            val_pasts = torch.FloatTensor(val_pasts)
            self.batch_val = mantra.NoFutureDataset(
                past=val_pasts)
            self.val_loader = DataLoader(
                self.batch_val, batch_size=256, num_workers=12, shuffle=False)
        print('dataset created')
        # load model to evaluate
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = config.preds
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len
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
        # populate the memory
        finished_scenarios = 0
        all_prediction = self.evaluate(
            self.val_loader).cpu()
        if self.metric_or_predict == 'metric':
            self.save_results(self.dict_metrics)
        offset = 0
        for i in range(len(self.data_val)):
            name = self.data_val[i][0][0].file_name.split('.')[0]
            weather_file = self.data_val[i][0][0].weather_type
            train_or_val = self.data_val[i][0][0].train_val
            print(name, weather_file)
            col_path = self.folder_test + name + '/' + weather_file + '/'
            if not os.path.exists(col_path):
                os.makedirs(col_path)
            for k in range(len(self.data_val[i][0])):
                f = open(self.folder_test + name + '/' + weather_file + '/' + str(self.data_val[i][0][k].now_frame) + '-' + str(
                    self.data_val[i][0][k].now_frame + self.future_len + 19) + '.txt', 'w')
                for j in range(len(self.data_val[i])):
                    origin = self.data_val[i][j][k].presents
                    pasts = self.data_val[i][j][k].pasts
                    prediction = all_prediction[offset]
                    offset += 1
                    for t in range(self.future_len + 20):
                        if t < 20:
                            x = float(pasts[0][t][0] + origin[0][0])
                            y = float(pasts[0][t][1] + origin[0][1])
                            list = [str(self.data_val[i][j][k].now_frame + t), '\t', str(
                                self.data_val[i][j][k].track_id), '\t', str(x), '\t', str(y), '\n']
                            f.writelines(list)
                        else:
                            pred_x = float(
                                prediction[0][t - 20][0] + origin[0][0])
                            pred_y = float(
                                prediction[0][t - 20][1] + origin[0][1])
                            list = [str(self.data_val[i][j][k].now_frame + t), '\t', str(
                                self.data_val[i][j][k].track_id), '\t', str(pred_x), '\t', str(pred_y), '\n']
                            f.writelines(list)
                f.close()
            finished_scenarios += 1
            print("finished_scenarios:", finished_scenarios)

    def save_results(self, dict_metrics_test):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param dict_metrics_train: dictionary with train metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')

        self.file.write("model:" + self.config.model + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("memory size: " +
                        str(len(self.mem_n2n.memory_past)) + '\n')

        self.file.write(
            "error 1s: " + str(dict_metrics_test['horizon10s']) + 'm \n')
        self.file.write(
            "error 2s: " + str(dict_metrics_test['horizon20s']) + 'm \n')
        self.file.write(
            "error 3s: " + str(dict_metrics_test['horizon30s']) + 'm \n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s']) + 'm \n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s']) + 'm \n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s']) + 'm \n')
        self.file.write("future_len:" + str(self.future_len) + 'm \n')
        if str(self.future_len) == '60':
            self.file.write(
                "error 6s: " + str(dict_metrics_test['horizon60s']) + 'm \n')
            self.file.write(
                "ADE 6s: " + str(dict_metrics_test['ADE_6s']) + 'm \n')
        self.file.close()

    def evaluate(self, loader):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :return: dictionary of performance metrics
        """
        self.mem_n2n.eval()
        all_pred = torch.tensor([]).cuda()
        with torch.no_grad():
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0
            ADE_6s = horizon60s = 0
            if self.metric_or_predict == 'metric':
                for step, (past, future, presents) in enumerate(loader):
                    past = Variable(past)
                    future = Variable(future)
                    if self.config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                    pred = self.mem_n2n(past)
                    all_pred = torch.cat((all_pred, pred), axis=0)
                    future_rep = future.unsqueeze(1).repeat(
                        1, self.config.preds, 1, 1)
                    distances = torch.norm(pred - future_rep, dim=3)
                    mean_distances = torch.mean(distances, dim=2)
                    index_min = torch.argmin(mean_distances, dim=1)
                    distance_pred = distances[torch.arange(
                        0, len(index_min)), index_min]
                    horizon10s += sum(distance_pred[:, 9])
                    horizon20s += sum(distance_pred[:, 19])
                    horizon30s += sum(distance_pred[:, 29])
                    ADE_1s += sum(torch.mean(distance_pred[:, :10], dim=1))
                    ADE_2s += sum(torch.mean(distance_pred[:, :20], dim=1))
                    ADE_3s += sum(torch.mean(distance_pred[:, :30], dim=1))
                    if str(self.future_len) == '60':
                        horizon60s += sum(distance_pred[:, 59])
                        ADE_6s += sum(torch.mean(distance_pred[:, :60], dim=1))
                print("final ADE:", ADE_3s, horizon30s, len(loader.dataset))
                self.dict_metrics['ADE_1s'] = round(
                    (ADE_1s / len(loader.dataset)).item(), 3)
                self.dict_metrics['ADE_2s'] = round(
                    (ADE_2s / len(loader.dataset)).item(), 3)
                self.dict_metrics['ADE_3s'] = round(
                    (ADE_3s / len(loader.dataset)).item(), 3)
                self.dict_metrics['horizon10s'] = round(
                    (horizon10s / len(loader.dataset)).item(), 3)
                self.dict_metrics['horizon20s'] = round(
                    (horizon20s / len(loader.dataset)).item(), 3)
                self.dict_metrics['horizon30s'] = round(
                    (horizon30s / len(loader.dataset)).item(), 3)
                if str(self.future_len) == '60':
                    self.dict_metrics['ADE_6s'] = round(
                        (ADE_6s / len(loader.dataset)).item(), 3)
                    self.dict_metrics['horizon60s'] = round(
                        (horizon60s / len(loader.dataset)).item(), 3)
            elif self.metric_or_predict == 'predict':
                for step, (past) in enumerate(loader):
                    past = Variable(past)
                    if self.config.cuda:
                        past = past.cuda()
                    pred = self.mem_n2n(past)
                    all_pred = torch.cat((all_pred, pred), axis=0)

        return all_pred

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if self.load_memory:
            print("memory_loading")
            if not self.load_non_interactive:
                self.mem_n2n.memory_past = torch.load(
                    self.memories_path + 'memory_past.pt')
                self.mem_n2n.memory_fut = torch.load(
                    self.memories_path + 'memory_fut.pt')
            else:
                self.mem_n2n.memory_past = torch.load(
                    self.memories_path + 'memory_past.pt')
                self.mem_n2n.memory_fut = torch.load(
                    self.memories_path + 'memory_fut.pt')
        else:
            print("memory_writing")
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (past, future, present) in enumerate(tqdm.tqdm(self.train_loader_one_array)):
                    past = Variable(past)
                    future = Variable(future)
                    if config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                    self.mem_n2n.write_in_memory(past, future)

                # save memory
                torch.save(self.mem_n2n.memory_past,
                           self.memories_path + 'memory_past.pt')
                torch.save(self.mem_n2n.memory_fut,
                           self.memories_path + 'memory_fut.pt')
