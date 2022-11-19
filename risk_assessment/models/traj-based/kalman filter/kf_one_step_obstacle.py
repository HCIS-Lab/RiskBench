import sys

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import math
import torch
from collections import defaultdict
import argparse


class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


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
        return [str(specific_frame), int(now_id)]


def main(data_type, future_len, specific_frame, data_path):
    future_len = int(future_len)
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
    folder = data_path + data_type + '/'
    for scenario_name in sorted(os.listdir(folder)):
        has_obstacle = True
        if not os.path.exists(folder + scenario_name + '/obstacle'):
            has_obstacle = False
        elif not os.path.exists(folder + scenario_name + '/obstacle/obstacle_list.txt'):
            has_obstacle = False
        else:
            if os.path.getsize(folder + scenario_name + '/obstacle/obstacle_list.txt') == 0:
                has_obstacle = False
        for weather_type in sorted(os.listdir(folder + scenario_name + '/variant_scenario/')):
            ego_list = []
            actor_list = []
            actor_id_list = []
            if data_type == 'interactive' or data_type == 'collision':
                traj_df = pd.read_csv(folder + scenario_name + '/variant_scenario/' +
                                      weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
                    if obj_type == 'AGENT':
                        ego_list.append(remain_df)
                    elif obj_type == 'actor.vehicle' or obj_type == 'actor.pedestrian':
                        actor_list.append(remain_df)
                        actor_id_list.append(remain_df.TRACK_ID.values[0])
                with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/dynamic_description.json') as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    variant_actor_id = str(data[actor_id_list[0]])
                    traj_df.loc[traj_df.TRACK_ID ==
                                variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
                    traj_df.loc[traj_df.TRACK_ID ==
                                variant_actor_id, 'OBJECT_TYPE'] = 'ACTOR'
            if data_type == 'non-interactive':
                traj_df = pd.read_csv(folder + scenario_name + '/variant_scenario/' +
                                      weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/dynamic_description.json') as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    traj_df.loc[traj_df.TRACK_ID ==
                                variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
                    traj_df.to_csv(data_path + data_type + '/' + scenario_name + '/variant_scenario/' +
                                   weather_type + '/trajectory_frame/' + scenario_name + '.csv', index=False)
            if has_obstacle and data_type == 'obstacle':
                traj_df = pd.read_csv(folder + scenario_name + '/variant_scenario/' +
                                      weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/dynamic_description.json') as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    traj_df.loc[traj_df.TRACK_ID ==
                                variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
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
                    with open(os.path.join(folder + scenario_name + '/obstacle/', 'obstacle_list.txt')) as f:
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
                        # Now obstacle GT only on 5 and 10, otherwise change data generator
                        if scenario_name.split('_')[0] != '5' and scenario_name.split('_')[0] != '10':
                            continue
                        with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/obstacle_info.json') as f:
                            obj_data = json.load(f)
                            obj_d = obj_data['interactive_obstacle_list']
                        right_id_obj = pd.DataFrame()
                        f = open(
                            data_path + data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/GT.txt', 'w')
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
            if data_type == 'obstacle':
                with open(data_path + data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/obstacle_info.json') as f:
                    obj_data = json.load(f)
                    obj_d = obj_data['interactive_obstacle_list']
                    # label by 邦元
                    #last_frame = obj_data['interactive frame'][1]
                with open(data_path + data_type + '_GT.json') as f:
                    GT = json.load(f)
                    f.close()
                frame_id = GT[scenario_name][weather_type]['nearset_frame']
                for item in obj_d.items():
                    right_id_obj_list.append(item[0])
            risky_vehicle_list = []
            # get ego and actor id
            actor_id_list = []
            vehicle_id_list = []
            pedestrian_id_list = []
            for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
                if obj_type == 'actor.vehicle' or obj_type == 'actor.pedestrian':
                    actor_id_list.append(remain_df.TRACK_ID.values[0])
            for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                if remain_df.OBJECT_TYPE.values[0] == 'vehicle':
                    vehicle_id_list.append(remain_df.TRACK_ID.values[0])
                elif remain_df.OBJECT_TYPE.values[0] == 'pedestrian':
                    pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
            with open(data_path + data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/dynamic_description.json') as f:
                data = json.load(f)
                variant_ego_id = str(data['player'])
                if data_type == 'interactive' or data_type == 'collision':
                    variant_actor_id = str(data[actor_id_list[0]])
                else:
                    variant_actor_id = '0'
            vehicle_list = []
            filter = (traj_df.OBJECT_TYPE != ('AGENT'))
            traj_df = traj_df[filter].reset_index(drop=True)
            filter = (traj_df.OBJECT_TYPE != ('actor.vehicle'))
            traj_df = traj_df[filter].reset_index(drop=True)
            filter = (traj_df.OBJECT_TYPE != ('actor.pedestrian'))
            traj_df = traj_df[filter].reset_index(drop=True)
            all_txt_list = []
            for filename in sorted(os.listdir(data_path + data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/')):
                all_txt_list.append(
                    int(filename.split(".")[0]))
            bbox_time_list = np.array(all_txt_list)
            bbox_first_frame = np.min(bbox_time_list)
            filter = (traj_df.FRAME >= int(bbox_first_frame))
            traj_df = traj_df[filter].reset_index(drop=True)
            mid_mask = (traj_df.FRAME == specific_frame) & (
                traj_df.OBJECT_TYPE == 'EGO')
            mid_pos_x = float(traj_df[mid_mask]['X'])
            mid_pos_y = float(traj_df[mid_mask]['Y'])
            for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                filter = (remain_df.FRAME > (
                    specific_frame - 20))
                remain_df = remain_df[filter].reset_index(drop=True)
                remain_df = remain_df.reset_index(drop=True)
                actor_pos_x = float(
                    remain_df.loc[20, 'X'])
                actor_pos_y = float(
                    remain_df.loc[20, 'Y'])
                dist_x = actor_pos_x - mid_pos_x
                dist_y = actor_pos_y - mid_pos_y
                if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                    vehicle_list.append(remain_df)
            if scenario_name == '5_t3-9_0_m_u_l_0' and (weather_type == 'MidRainyNight_high_' or weather_type == 'MidRainyNoon_low_'):
                bbox_first_frame = 64
            frame_filter = (traj_df.OBJECT_TYPE == ('EGO'))
            frame_df = traj_df[frame_filter].reset_index(drop=True)
            if specific_frame - bbox_first_frame < 20:
                print("There is no enough data")
                break
            #print(bbox_first_frame, vehicle_list[0])
            for frame_index in range(1):
                d = dict()
                d['scenario_id'] = scenario_name
                for val_vehicle_num in range(len(vehicle_list)):
                    now_id = vehicle_list[val_vehicle_num].TRACK_ID[0]
                    if int(now_id) == int(variant_ego_id):
                        x = vehicle_list[val_vehicle_num].X.to_numpy()
                        y = vehicle_list[val_vehicle_num].Y.to_numpy()
                        kf = KalmanFilter()
                        x_pred = []
                        y_pred = []
                        count = 0
                        for x_now, y_now in zip(x[frame_index:frame_index + 20], y[frame_index:frame_index + 20]):
                            count += 1
                            if count <= 20:
                                pred = kf.Estimate(x_now, y_now)
                                x_pred.append(pred[0])
                                y_pred.append(pred[1])
                        for temp in range(future_len):
                            pred = kf.Estimate(
                                x_pred[temp + 19], y_pred[temp + 19])
                            x_pred.append(pred[0])
                            y_pred.append(pred[1])
                        ego_prediction = np.zeros((len(x_pred), 2))
                        for pred_t in range(len(x_pred) - 21):
                            real_pred_x = x_pred[pred_t + 20]
                            real_pred_x_next = x_pred[pred_t + 21]
                            real_pred_y = y_pred[pred_t + 20]
                            real_pred_y_next = y_pred[pred_t + 21]
                            ego_prediction[pred_t][0] = real_pred_x
                            ego_prediction[pred_t][1] = real_pred_y
                            if pred_t == int(len(x_pred) - 22):
                                ego_prediction[pred_t +
                                               1][0] = real_pred_x_next
                                ego_prediction[pred_t +
                                               1][1] = real_pred_y_next
                for val_vehicle_num in range(len(vehicle_list)):
                    vl = vehicle_list[val_vehicle_num].to_numpy()
                    x = vehicle_list[val_vehicle_num].X.to_numpy()
                    y = vehicle_list[val_vehicle_num].Y.to_numpy()
                    x = x.astype(float)
                    y = y.astype(float)
                    kf = KalmanFilter()
                    x_pred = []
                    y_pred = []
                    count = 0
                    for x_now, y_now in zip(x[frame_index:frame_index + 20], y[frame_index:frame_index + 20]):
                        count += 1
                        if count <= 20:
                            pred = kf.Estimate(x_now, y_now)
                            x_pred.append(pred[0])
                            y_pred.append(pred[1])
                    for temp in range(future_len):
                        pred = kf.Estimate(
                            x_pred[temp + 19], y_pred[temp + 19])
                        x_pred.append(pred[0])
                        y_pred.append(pred[1])
                    now_id = int(vehicle_list[val_vehicle_num].TRACK_ID[0])
                    if str(int(now_id)) in pedestrian_id_list:
                        agent_type = 1
                    elif str(int(now_id)) in vehicle_id_list:
                        agent_type = 0
                    for pred_t in range(future_len - 1):
                        if int(now_id) == int(variant_ego_id):
                            continue
                        real_pred_x = x_pred[pred_t + 20]
                        real_pred_x_next = x_pred[pred_t + 21]
                        real_pred_y = y_pred[pred_t + 20]
                        real_pred_y_next = y_pred[pred_t + 21]

                        #################

                        if str(now_id) in obs_list:
                            if vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.trafficcone01':
                                temp = obstacle_collision(vehicle_length, vehicle_width, 0.85, 0.85, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                          pred_t + 1][0], ego_prediction[pred_t + 1][1], float(x[0]), float(y[0]), 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                            elif vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.streetbarrier':
                                temp = obstacle_collision(vehicle_length, vehicle_width, 1.25, 0.375, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                          pred_t + 1][0], ego_prediction[pred_t + 1][1], float(x[0]), float(y[0]), 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                            elif vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.trafficwarning':
                                temp = obstacle_collision(vehicle_length, vehicle_width, 3, 2.33, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                          pred_t + 1][0], ego_prediction[pred_t + 1][1], float(x[0]), float(y[0]), 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                            if temp != None:
                                risky_vehicle_list.append(temp)
                        ########################
                            """

                            if data_type == 'obstacle' and (now_id >= 100000 or str(now_id) in right_id_obj_list):
                                obs_type = now_id / 100000 - 1
                                obs_index = now_id % 100000
                                if str(now_id) in right_id_obj_list:
                                    with open(data_path + data_type + '/' + scenario_name + '/variant_scenario/' + weather_type + '/GT.txt') as f:
                                        for line in f.readlines():
                                            s = line.split('\t')
                                            if int(now_id) == int(s[1]):
                                                obs_type = int(
                                                    s[0]) / 100000 - 1
                                                obs_index = int(s[0]) % 100000
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
                                            # print(pos)
                                            pos = s[1].split(',')
                                            yaw = (pos[4].split('='))[1]
                                temp = obstacle_collision(vehicle_length, vehicle_width, object_area[int(obs_type)][0], object_area[int(
                                    obs_type)][1], ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[pred_t + 1][0], ego_prediction[pred_t + 1][1], float(x[0]), float(y[0]), float(yaw), vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                                if temp != None:
                                    risky_vehicle_list.append(temp)
                        """
                        else:
                            center_distance_vector = [
                                real_pred_x - ego_prediction[pred_t][0], real_pred_y - ego_prediction[pred_t][1]]
                            vehicle_vector_square = math.sqrt(
                                (real_pred_x_next - real_pred_x)**2 + (real_pred_y_next - real_pred_y)**2)
                            if vehicle_vector_square == 0:
                                vehicle_vector_square = 2147483647
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
                                    [str(specific_frame), int(now_id)])
            file_d = {}
            d = {}
            for frame, id in risky_vehicle_list:
                if frame in file_d:
                    if str(id) in file_d[frame]:
                        continue
                    else:
                        file_d[frame].append(str(id))
                else:
                    file_d[frame] = [str(id)]
            # check all prediction result
            # print(file_d)
            if str(specific_frame) in file_d:
                risky_id = file_d[str(specific_frame)]
            else:
                risky_id = None
            print(scenario_name, ' ', weather_type,
                  ' risky_id:',  risky_id)


if __name__ == '__main__':
    # python kf_one_step.py --type obstacle --future_length 30 --frame 160

    # For obstacle test data, 153~172 can find risky id
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='obstacle',
                        type=str, required=True)
    parser.add_argument('--future_length', default='30',
                        type=str, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--data_path', type=str, default='../test_data/')
    args = parser.parse_args()
    main(args.type, args.future_length, args.frame, args.data_path)
