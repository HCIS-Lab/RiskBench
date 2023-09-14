import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import sys
import threading as td
import json
import cv2
import matplotlib.patches as mpatches
import argparse


def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    # 若是車輛靜止不動 ego_vec為[0 0]
    if v1[0] < 0.01 and v1[1] < 0.01:
        v1_u = [1.0, 0.1]
    else:
        v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 因為np.arccos給的範圍只有0~pi (180度)，但若cos值為-0.5，不論是120度或是240度，都會回傳120度，因此考慮三四象限的情況，強制轉180度到一二象限(因車輛和行人面積的對稱性，故可這麼做)
    if v1_u[1] < 0:
        v1_u = v1_u * (-1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle


def obstacle_collision(car_length, car_width, obs_length, obs_width, ego_x, ego_y, ego_x_next, ego_y_next, obs_x, obs_y, obs_yaw, vehicle_length, vehicle_width, filename, pred_t, vl):
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
    x1 = float(np.cos(obs_rec[4])*(-obs_rec[2]/2) -
               np.sin(obs_rec[4])*(-obs_rec[3]/2) + obs_rec[0])
    x2 = float(np.cos(obs_rec[4])*(obs_rec[2]/2) -
               np.sin(obs_rec[4])*(-obs_rec[3]/2) + obs_rec[0])
    x3 = float(np.cos(obs_rec[4])*(-obs_rec[2]/2) -
               np.sin(obs_rec[4])*(obs_rec[3]/2) + obs_rec[0])
    x4 = float(np.cos(obs_rec[4])*(obs_rec[2]/2) -
               np.sin(obs_rec[4])*(obs_rec[3]/2) + obs_rec[0])
    y1 = float(np.sin(obs_rec[4])*(-obs_rec[2]/2) +
               np.cos(obs_rec[4])*(-obs_rec[3]/2) + obs_rec[1])
    y2 = float(np.sin(obs_rec[4])*(obs_rec[2]/2) +
               np.cos(obs_rec[4])*(-obs_rec[3]/2) + obs_rec[1])
    y3 = float(np.sin(obs_rec[4])*(-obs_rec[2]/2) +
               np.cos(obs_rec[4])*(obs_rec[3]/2) + obs_rec[1])
    y4 = float(np.sin(obs_rec[4])*(obs_rec[2]/2) +
               np.cos(obs_rec[4])*(obs_rec[3]/2) + obs_rec[1])
    plt.plot([x1, x2, x4, x3, x1], [y1, y2, y4, y3, y1],
             '-', color='purple', markersize=3)
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
        plt.plot([obs_x, ego_x], [obs_y, ego_y],
                 '-.', color='red', markersize=1)
        return [str(int(filename.split(".")[0].split("_")[0].split("-")[0]) + pred_t + 20), int(vl[0][1])]


def main(future_len, data_source):
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
    all_type = ['interactive', 'obstacle', 'collision', 'non-interactive']
    for data_type in all_type:
        folder = data_source + '/' + data_type + '/'
        for scenario_name in sorted(os.listdir(folder)):
            for weather_type in sorted(os.listdir(folder + scenario_name + '/')):
                last_frame = 999
                if scenario_name != '10_i-1_1_c_l_r_1_0' or weather_type != 'MidRainyNight_low_':
                    continue
                if scenario_name == '10_i-1_1_t_r_f_1_rl' and weather_type == 'ClearNoon_low_':
                    continue
                if scenario_name == '10_i-1_1_t_r_f_1_rl' and weather_type == 'WetCloudyNoon_mid_':
                    continue
                sav_path = folder + scenario_name + \
                    '/' + weather_type + '/prediction_test'
                lane_feature = np.load(folder + scenario_name +
                                       '/variant_scenario/' + weather_type + '/topology/' + scenario_name + '.npy', allow_pickle=True)
                if not os.path.exists(sav_path):
                    os.makedirs(sav_path)
                col_path = folder + scenario_name + '/' + \
                    weather_type + '/collsion_description'
                if not os.path.exists(col_path):
                    os.makedirs(col_path)

                right_id_obj_list = []

                if data_type == 'interactive' or data_type == 'collision':
                    with open(data_type + '_GT.json') as f:
                        GT = json.load(f)
                        f.close()
                    frame_id = GT[scenario_name][weather_type]
                    for frame, id in frame_id.items():
                        last_frame = int(frame)
                elif data_type == 'obstacle':
                    with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/obstacle_info.json') as f:
                        obj_data = json.load(f)
                        obj_d = obj_data['interactive_obstacle_list']
                        last_frame = obj_data['interactive frame'][1]

                    for item in obj_d.items():
                        right_id_obj_list.append(item[0])

                if scenario_name == '5_s-9_0_m_sl_sr_0_1' and weather_type == 'ClearSunset_low_':
                    last_frame = 149
                if scenario_name == '5_t2-2_0_f_sl' and weather_type == 'CloudySunset_high_':
                    last_frame = 213

                all_txt_list = []
                for filename in sorted(os.listdir(folder + scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/')):
                    all_txt_list.append(
                        int(filename.split(".")[0]))
                bbox_time_list = np.array(all_txt_list)
                bbox_first_frame = np.min(bbox_time_list)

                # get first frame
                origin_df = pd.read_csv(folder + scenario_name +
                                        '/variant_scenario/' + weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                # get ego and actor id
                actor_id_list = []
                vehicle_id_list = []
                pedestrian_id_list = []
                for obj_type, remain_df in origin_df.groupby('OBJECT_TYPE'):
                    if obj_type == 'actor.vehicle':
                        actor_id_list.append(remain_df.TRACK_ID.values[0])
                        vehicle_id_list.append(remain_df.TRACK_ID.values[0])
                        actor_agent_type = 0
                    if obj_type == 'actor.pedestrian':
                        actor_id_list.append(remain_df.TRACK_ID.values[0])
                        pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
                        actor_agent_type = 1
                for track_id, remain_df in origin_df.groupby('TRACK_ID'):
                    if remain_df.OBJECT_TYPE.values[0] == 'vehicle':
                        vehicle_id_list.append(remain_df.TRACK_ID.values[0])
                    elif remain_df.OBJECT_TYPE.values[0] == 'pedestrian':
                        pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
                dynamic_path = folder + scenario_name + '/variant_scenario/' + \
                    weather_type + '/dynamic_description.json'
                with open(dynamic_path) as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    if data_type == 'interactive' or data_type == 'collision':
                        variant_actor_id = str(data[actor_id_list[0]])
                    else:
                        variant_actor_id = '0'
                # get 75*75 region
                filter = (origin_df.OBJECT_TYPE != ('AGENT'))
                origin_df = origin_df[filter].reset_index(drop=True)
                mid_time_index = int(
                    len(pd.unique(origin_df['FRAME'].values)) / 2)
                mid_time = pd.unique(origin_df['FRAME'].values)[mid_time_index]
                mid_mask = (origin_df.FRAME == mid_time) & (
                    origin_df.TRACK_ID == str(variant_ego_id))
                mid_pos_x = float(origin_df[mid_mask]['X'])
                mid_pos_y = float(origin_df[mid_mask]['Y'])

                frame_filter = (origin_df.OBJECT_TYPE == ('EGO'))
                frame_df = origin_df[frame_filter].reset_index(drop=True)
                if scenario_name == '5_t3-9_0_m_u_l_0' and (weather_type == 'MidRainyNight_high_' or weather_type == 'MidRainyNoon_low_'):
                    bbox_first_frame = 64
                first_data_df = pd.read_csv(os.path.join(folder + scenario_name + '/' + weather_type + '/' + str(
                    int(bbox_first_frame)) + '-' + str(int(bbox_first_frame) + 19 + future_len) + '.txt', ), sep='\t', header=None)
                first_data_list = []
                for track_id, remain_df in first_data_df.groupby(1):
                    if int(track_id) == int(variant_ego_id):
                        first_data_list.append(remain_df)
                forever_present_x = int(
                    first_data_list[0].to_numpy()[20][2]) * (-1)
                forever_present_y = int(
                    first_data_list[0].to_numpy()[20][3])
                risky_vehicle_list = []
                for filename in sorted(os.listdir(folder + scenario_name + '/' + weather_type + '/')):
                    if filename == 'prediction_test':
                        continue
                    if filename == 'prediction':
                        continue
                    elif filename == 'collsion_description':
                        continue

                    if int(filename.split(".")[0].split("-")[0]) < bbox_first_frame or int(filename.split(".")[0].split("-")[0]) + 20 > last_frame:
                        continue
                    if True:
                        print(scenario_name, weather_type,
                              filename.split(".")[0])
                        traj_df = pd.read_csv(os.path.join(
                            folder + scenario_name + '/' + weather_type + '/', filename), sep='\t', header=None)
                        vehicle_list = []
                        for track_id, remain_df in traj_df.groupby(1):
                            vehicle_list.append(remain_df)
                        ego_prediction = np.zeros((future_len, 2))
                        collision_list = []
                        fig, ax = plt.subplots()
                        ax.set_xlabel("x axis(m)")
                        ax.set_ylabel("y axis(m)")
                        ego, = ax.plot(
                            [0, 0], label="detected collision", color='red', linestyle='--')
                        first_legend = ax.legend(
                            handles=[ego], loc='lower left', bbox_to_anchor=(0.6, 0.3))
                        ax.add_artist(first_legend)
                        black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
                                                         edgecolor='black', label='agents')
                        black_legend = ax.legend(
                            handles=[black_patch], loc='lower left', bbox_to_anchor=(0.6, 0.2))
                        ax.add_artist(black_legend)
                        darkgreen_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
                                                             edgecolor='lime', label='ego')
                        darkgreen_legend = ax.legend(
                            handles=[darkgreen_patch], loc='lower left', bbox_to_anchor=(0.6, 0.1))
                        ax.add_artist(darkgreen_legend)
                        purple_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
                                                          edgecolor='violet', label='risk')
                        ax.legend(
                            handles=[purple_patch], loc='lower left', bbox_to_anchor=(0.6, 0))
                        d = dict()
                        d['scenario_id'] = scenario_name
                        for features in lane_feature:
                            xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
                                :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
                            plt.plot(xs * (-1), ys, '-.', color='lightgray')
                            x_s, y_s = np.vstack((features[1][:, :2], features[1][-1, 3:5]))[
                                :, 0], np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 1]
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
                                    if pred_t % 3 == 0:
                                        plt.scatter([(-1) * real_pred_x_next], [
                                            real_pred_y_next], color='lime', s=3)
                                    ego_prediction[pred_t][0] = real_pred_x
                                    ego_prediction[pred_t][1] = real_pred_y
                                    if pred_t == int(future_len - 2):
                                        ego_prediction[pred_t +
                                                       1][0] = real_pred_x_next
                                        ego_prediction[pred_t +
                                                       1][1] = real_pred_y_next
                        for n in range(len(vehicle_list)):
                            vl = vehicle_list[n].to_numpy()
                            # vl : frame, id, x, y
                            now_id = vl[0][1]
                            if str(int(now_id)) in pedestrian_id_list:
                                agent_type = 1
                            elif str(int(now_id)) in vehicle_id_list:
                                agent_type = 0
                            elif int(now_id) == int(variant_ego_id):
                                agent_type = 0
                            elif int(now_id) == int(variant_actor_id):
                                agent_type = actor_agent_type
                            actor_pos_x = vl[0][2]
                            actor_pos_y = vl[0][3]
                            dist_x = actor_pos_x - ego_now_pos_x
                            dist_y = actor_pos_y - ego_now_pos_y
                            if abs(dist_x) > 37.5 or abs(dist_y) > 37.5:
                                continue
                            for t in range(19):
                                real_past_x = vl[t][2]
                                real_past_x_next = vl[t + 1][2]
                                real_past_y = vl[t][3]
                                real_past_y_next = vl[t + 1][3]
                                if t == 18 and int(now_id) < 100000 and str(now_id) not in right_id_obj_list:
                                    ego_vec = [real_past_y_next - real_past_y,
                                               real_past_x_next * (-1) - real_past_x * (-1)]
                                    angle = np.rad2deg(
                                        angle_vectors(ego_vec, [1, 0]))
                                    transform_angle = (
                                        float(angle)) * np.pi / 180
                                    ego_rec = [real_past_x_next, real_past_y_next, agent_area[agent_type]
                                               [1], agent_area[agent_type][0], transform_angle]
                                    x_1 = float(np.cos(
                                        ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                                    x_2 = float(np.cos(
                                        ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                                    x_3 = float(np.cos(
                                        ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                                    x_4 = float(np.cos(
                                        ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                                    y_1 = float(np.sin(
                                        ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                                    y_2 = float(np.sin(
                                        ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                                    y_3 = float(np.sin(
                                        ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                                    y_4 = float(np.sin(
                                        ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                                if int(now_id) == int(variant_ego_id):
                                    if t == 18:
                                        plt.plot([x_1 * (-1), x_2 * (-1), x_4 * (-1), x_3 * (-1), x_1 * (-1)], [
                                            y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
                                elif int(now_id) == int(variant_actor_id):
                                    if t == 18:
                                        plt.plot([x_1 * (-1), x_2 * (-1), x_4 * (-1), x_3 * (-1), x_1 * (-1)], [
                                            y_1, y_2, y_4, y_3, y_1], color='violet')
                                elif str(int(now_id)) in vehicle_id_list:
                                    if t == 18 and int(now_id) < 100000 and int(now_id) != int(variant_ego_id) and str(now_id) not in right_id_obj_list:
                                        plt.plot([x_1 * (-1), x_2 * (-1), x_4 * (-1), x_3 * (-1), x_1 * (-1)], [
                                            y_1, y_2, y_4, y_3, y_1], color='black')
                                elif str(int(now_id)) in pedestrian_id_list:
                                    if t == 18 and int(now_id) < 100000 and int(now_id) != int(variant_ego_id) and str(now_id) not in right_id_obj_list:
                                        plt.plot([x_1 * (-1), x_2 * (-1), x_4 * (-1), x_3 * (-1), x_1 * (-1)], [
                                            y_1, y_2, y_4, y_3, y_1], color='black')
                            for pred_t in range(future_len - 1):
                                if int(now_id) == int(variant_ego_id):
                                    continue
                                real_pred_x = vl[pred_t + 20][2]
                                real_pred_x_next = vl[pred_t + 21][2]
                                real_pred_y = vl[pred_t + 20][3]
                                real_pred_y_next = vl[pred_t + 21][3]
                                if int(now_id) == int(variant_actor_id):
                                    plt.plot([real_pred_x * (-1), real_pred_x_next * (-1)], [
                                        real_pred_y, real_pred_y_next], '--o', color='violet', markersize=1)
                                elif str(int(now_id)) in vehicle_id_list:
                                    plt.plot([real_pred_x * (-1), real_pred_x_next * (-1)], [
                                        real_pred_y, real_pred_y_next], '--o', color='black', markersize=1)
                                elif str(int(now_id)) in pedestrian_id_list:
                                    plt.plot([real_pred_x * (-1), real_pred_x_next * (-1)], [
                                        real_pred_y, real_pred_y_next], '--o', color='black', markersize=1)
                                if data_type == 'obstacle' and (now_id >= 100000 or str(now_id) in right_id_obj_list):
                                    obs_type = now_id / 100000 - 1
                                    obs_index = now_id % 100000
                                    if str(now_id) in right_id_obj_list:
                                        with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/GT.txt') as f:
                                            for line in f.readlines():
                                                s = line.split('\t')
                                                if int(now_id) == int(s[1]):
                                                    obs_type = int(
                                                        s[0]) / 100000 - 1
                                                    obs_index = int(
                                                        s[0]) % 100000
                                    with open(os.path.join(folder + scenario_name + '/obstacle/', 'obstacle_list.txt')) as f:
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
                                        obs_type)][1], ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], yaw, vehicle_length, vehicle_width, filename, pred_t, vl)
                                    if temp != None:
                                        risky_vehicle_list.append(temp)
                                else:
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
                                        risky_vehicle_list.append([str(int(filename.split(".")[0].split(
                                            "_")[0].split("-")[0]) + pred_t + 20), int(vl[0][1])])
                                        plt.plot([real_pred_x * (-1), ego_prediction[pred_t][0] * (-1)], [
                                            real_pred_y, ego_prediction[pred_t][1]], '-.', color='red', markersize=1)
                        plt.xlim(forever_present_x - 37,
                                 forever_present_x + 37)
                        plt.ylim(forever_present_y - 37,
                                 forever_present_y + 37)
                        plt.savefig(sav_path + '/' +
                                    str(filename.split(".")[0]) + '.png', dpi=500)
                        plt.close()
                    temp_d = {}
                    d = {}
                    for frame, id in risky_vehicle_list:
                        if frame in temp_d:
                            if str(id) in temp_d[frame]:
                                continue
                            else:
                                temp_d[frame].append(str(id))
                        else:
                            temp_d[frame] = [str(id)]
                    if len(risky_vehicle_list) != 0:
                        with open(col_path + '/' + scenario_name + '.json', 'w') as f:
                            json.dump(temp_d, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--future_length', default='30',
                        type=str, required=True)
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()
    main(args.future_length, args.data_path)
