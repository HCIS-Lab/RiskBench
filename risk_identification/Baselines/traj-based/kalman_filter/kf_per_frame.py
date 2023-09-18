import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import math
import torch
import argparse

# Instantiate OCV kalman filter


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


def obstacle_collision(car_length, car_width, obs_length, obs_width, ego_x, ego_y, ego_x_next, ego_y_next, obs_x, obs_y, obs_yaw, vehicle_length, vehicle_width, filename, pred_t, now_id):
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
        return [str(filename + 20), int(now_id)]


def main(future_len, val_or_test, data_path):
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
    for type in sorted(os.listdir(os.path.join(data_path, val_or_test))):
        scenario_file = os.path.join(data_path, val_or_test, type)
        ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = 0
        dict_metrics = {}
        finished_scenario = -1
        for scenario_name in sorted(os.listdir(scenario_file)):
            weather_file = scenario_file + '/' + scenario_name + '/variant_scenario'
            for weather_type in sorted(os.listdir(weather_file)):
                if scenario_name != '10_s-2_0_c_sl_f_0' or weather_type != 'WetNight_high_':
                    continue
                last_frame = 999
                if weather_type.split('_')[1] != 'high':
                    continue
                right_id_obj_list = []
                if type == 'interactive' or type == 'collision':
                    with open(type + '_GT.json') as f:
                        GT = json.load(f)
                        f.close()
                    frame_id = GT[scenario_name][weather_type]
                    for frame, id in frame_id.items():
                        last_frame = int(frame)
                elif type == 'obstacle':
                    with open(scenario_file + '/' + scenario_name + '/variant_scenario/' + weather_type + '/obstacle_info.json') as f:
                        obj_data = json.load(f)
                        obj_d = obj_data['interactive_obstacle_list']
                        last_frame = obj_data['interactive frame'][1]
                    for item in obj_d.items():
                        right_id_obj_list.append(item[0])

                if scenario_name == '5_s-9_0_m_sl_sr_0_1' and weather_type == 'ClearSunset_low_':
                    last_frame = 149  # 165
                if scenario_name == '5_t2-2_0_f_sl' and weather_type == 'CloudySunset_high_':
                    last_frame = 213  # 215
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'ClearNight_low_':
                    last_frame = 119  # 122
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'ClearNoon_mid_':
                    last_frame = 117  # 121
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'MidRainyNoon_high_':
                    last_frame = 119  # 122
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'WetCloudyNight_mid_':
                    last_frame = 119  # 125
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'WetNight_low_':
                    last_frame = 117  # 123
                if scenario_name == '10_i-1_1_t_f_l_0' and weather_type == 'WetNoon_low_':
                    last_frame = 118  # 121
                # 10_i-1_1_t_f_l_0 last one weather is OK

                risky_vehicle_list = []
                sav_path = scenario_file + '/' + \
                    scenario_name + '/variant_scenario/' + weather_type + '/prediction_test'
                lane_feature = np.load(scenario_file + '/' + scenario_name +
                                       '/variant_scenario/' + weather_type + '/topology/' + scenario_name + '.npy', allow_pickle=True)
                if not os.path.exists(sav_path):
                    os.makedirs(sav_path)
                col_path = scenario_file + '/' + type + '/' + scenario_name + '/variant_scenario/' + \
                    weather_type + '/collsion_description'
                if not os.path.exists(col_path):
                    os.makedirs(col_path)

                traj_df = pd.read_csv(
                    weather_file + '/' + weather_type + '/trajectory_frame/' + scenario_name + '.csv')

                # get ego and actor id
                actor_id_list = []
                vehicle_id_list = []
                pedestrian_id_list = []
                for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
                    if obj_type == 'actor.vehicle':
                        actor_id_list.append(remain_df.TRACK_ID.values[0])
                        vehicle_id_list.append(remain_df.TRACK_ID.values[0])
                        actor_agent_type = 0
                    if obj_type == 'actor.pedestrian':
                        actor_id_list.append(remain_df.TRACK_ID.values[0])
                        pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
                        actor_agent_type = 1
                for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                    if remain_df.OBJECT_TYPE.values[0] == 'vehicle':
                        vehicle_id_list.append(remain_df.TRACK_ID.values[0])
                    elif remain_df.OBJECT_TYPE.values[0] == 'pedestrian':
                        pedestrian_id_list.append(remain_df.TRACK_ID.values[0])
                if val_or_test == 'val':
                    dynamic_path = scenario_file + '/' + \
                        scenario_name + '/variant_scenario/' + \
                        weather_type + '/dynamic_description.json'
                elif val_or_test == 'test':
                    dynamic_path = scenario_file + '/' + \
                        scenario_name + '/variant_scenario/' + \
                        weather_type + '/dynamic_description.json'
                with open(dynamic_path) as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    if type == 'interactive' or type == 'collision':
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
                for filename in sorted(os.listdir(scenario_file + '/' + scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/')):
                    all_txt_list.append(
                        int(filename.split(".")[0]))
                bbox_time_list = np.array(all_txt_list)
                bbox_first_frame = np.min(bbox_time_list)
                filter = (traj_df.FRAME >= int(bbox_first_frame))
                traj_df = traj_df[filter].reset_index(drop=True)

                if type == 'non-interactive':
                    last_frame = int(traj_df.FRAME.values[-1])

                scenario_horizon = last_frame - bbox_first_frame

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

                if scenario_name == '5_t3-9_0_m_u_l_0' and (weather_type == 'MidRainyNight_high_' or weather_type == 'MidRainyNoon_low_'):
                    bbox_first_frame = 64

                frame_filter = (traj_df.OBJECT_TYPE == ('EGO'))
                frame_df = traj_df[frame_filter].reset_index(drop=True)
                forever_present_x = int(frame_df.X.to_numpy()[20])
                forever_present_y = int(frame_df.Y.to_numpy()[20])
                for frame_index in range(scenario_horizon - 20):
                    print(type, scenario_name, weather_type,
                          str(bbox_first_frame + frame_index))

                    plt.figure(dpi=500)
                    plt.xlabel("x axis(m)")
                    plt.ylabel("y axis(m)")
                    ego, = plt.plot([0, 0], [0, 0], '--o',
                                    color='darkgreen', markersize=1)
                    agent, = plt.plot([0, 0], [0, 0], '--o',
                                      color='purple', markersize=1)
                    other, = plt.plot([0, 0], [0, 0], '--o',
                                      color='black', markersize=1)
                    plt.legend([ego, agent, other], ["ego", "agent", "other"])
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

                            # if no ground truth future, metric cannot be calculated
                            if (frame_index + 20 + future_len) < scenario_horizon:
                                finished_scenario += 1
                                future_numpy = np.vstack(
                                    (x[20:20 + future_len], y[20:20 + future_len])).T
                                future = torch.from_numpy(
                                    future_numpy).unsqueeze(0)
                                predict_numpy = np.vstack(
                                    (np.squeeze(np.array(x_pred[20:20 + future_len])), np.squeeze(np.array(y_pred[20:20 + future_len])))).T
                                predict = torch.from_numpy(
                                    predict_numpy).unsqueeze(0)
                                future_rep = future.unsqueeze(1).repeat(
                                    1, 1, 1, 1)
                                distances = torch.norm(
                                    predict - future_rep, dim=3)
                                mean_distances = torch.mean(distances, dim=2)
                                index_min = torch.argmin(mean_distances, dim=1)
                                distance_pred = distances[torch.arange(
                                    0, len(index_min)), index_min]
                                horizon10s += sum(distance_pred[:, 9])
                                horizon20s += sum(distance_pred[:, 19])
                                horizon30s += sum(distance_pred[:, 29])
                                ADE_1s += sum(torch.mean(
                                    distance_pred[:, :10], dim=1))
                                ADE_2s += sum(torch.mean(
                                    distance_pred[:, :20], dim=1))
                                ADE_3s += sum(torch.mean(
                                    distance_pred[:, :30], dim=1))
                            ego_prediction = np.zeros((len(x_pred), 2))
                            for pred_t in range(len(x_pred) - 21):
                                real_pred_x = x_pred[pred_t + 20]
                                real_pred_x_next = x_pred[pred_t + 21]
                                real_pred_y = y_pred[pred_t + 20]
                                real_pred_y_next = y_pred[pred_t + 21]
                                plt.plot([real_pred_x, real_pred_x_next], [
                                    real_pred_y, real_pred_y_next], '--o', color='lime', markersize=1)
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
                        elif int(now_id) == int(variant_ego_id):
                            agent_type = 0
                        elif int(now_id) == int(variant_actor_id):
                            agent_type = actor_agent_type

                        for t in range(19):
                            real_past_x = x[t + frame_index]
                            real_past_x_next = x[t + frame_index + 1]
                            real_past_y = y[t + frame_index]
                            real_past_y_next = y[t + frame_index + 1]
                            if t == 18 and int(now_id) < 100000 and str(now_id) not in right_id_obj_list:
                                ego_vec = [real_past_y_next - real_past_y,
                                           real_past_x_next - real_past_x]
                                angle = np.rad2deg(
                                    angle_vectors(ego_vec, [1, 0]))
                                transform_angle = (float(angle)) * np.pi / 180
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
                                plt.plot([real_past_x, real_past_x_next], [
                                    real_past_y, real_past_y_next], '--o', color='darkgreen', markersize=1)

                                if t == 18:
                                    plt.plot([x_1, x_2, x_4, x_3, x_1], [
                                        y_1, y_2, y_4, y_3, y_1], '-',  color='green', markersize=3)
                            elif int(now_id) == int(variant_actor_id):
                                plt.plot([real_past_x, real_past_x_next], [
                                    real_past_y, real_past_y_next], '--o', color='purple', markersize=1)
                                if t == 18:
                                    plt.plot([x_1, x_2, x_4, x_3, x_1], [
                                        y_1, y_2, y_4, y_3, y_1], color='purple')
                            elif str(int(now_id)) in vehicle_id_list:
                                plt.plot([real_past_x, real_past_x_next], [
                                    real_past_y, real_past_y_next], '--o', color='black', markersize=1)
                                if t == 18 and int(now_id) < 100000 and int(now_id) != int(variant_ego_id) and str(now_id) not in right_id_obj_list:
                                    plt.plot([x_1, x_2, x_4, x_3, x_1], [
                                        y_1, y_2, y_4, y_3, y_1], color='black')
                            elif str(int(now_id)) in pedestrian_id_list:
                                plt.plot([real_past_x, real_past_x_next], [
                                    real_past_y, real_past_y_next], '--o', color='darkblue', markersize=1)
                                if t == 18 and int(now_id) < 100000 and int(now_id) != int(variant_ego_id) and str(now_id) not in right_id_obj_list:
                                    plt.plot([x_1, x_2, x_4, x_3, x_1], [
                                        y_1, y_2, y_4, y_3, y_1], color='darkblue')
                        for pred_t in range(future_len - 1):
                            if int(now_id) == int(variant_ego_id):
                                continue
                            real_pred_x = x_pred[pred_t + 20]
                            real_pred_x_next = x_pred[pred_t + 21]
                            real_pred_y = y_pred[pred_t + 20]
                            real_pred_y_next = y_pred[pred_t + 21]
                            if int(now_id) == int(variant_actor_id):
                                plt.plot([real_pred_x, real_pred_x_next], [
                                    real_pred_y, real_pred_y_next], '--o', color='violet', markersize=1)
                            elif str(int(now_id)) in vehicle_id_list:
                                plt.plot([real_pred_x, real_pred_x_next], [
                                    real_pred_y, real_pred_y_next], '--o', color='brown', markersize=1)
                            elif str(int(now_id)) in pedestrian_id_list:
                                plt.plot([real_pred_x, real_pred_x_next], [
                                    real_pred_y, real_pred_y_next], '--o', color='blue', markersize=1)
                            if type == 'obstacle' and (now_id >= 100000 or str(now_id) in right_id_obj_list):
                                obs_type = now_id / 100000 - 1
                                obs_index = now_id % 100000
                                if str(now_id) in right_id_obj_list:
                                    with open(scenario_file + '/' + scenario_name + '/variant_scenario/' + weather_type + '/GT.txt') as f:
                                        for line in f.readlines():
                                            s = line.split('\t')
                                            if int(now_id) == int(s[1]):
                                                obs_type = int(
                                                    s[0]) / 100000 - 1
                                                obs_index = int(s[0]) % 100000
                                with open(os.path.join(scenario_file + '/' + scenario_name + '/obstacle/', 'obstacle_list.txt')) as f:
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
                                temp = obstacle_collision(vehicle_length, vehicle_width, object_area[int(obs_type)][0], object_area[int(
                                    obs_type)][1], ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[pred_t + 1][0], ego_prediction[pred_t + 1][1], float(x[0]), float(y[0]), float(yaw), vehicle_length, vehicle_width, bbox_first_frame + frame_index, pred_t, now_id)
                                if temp != None:
                                    risky_vehicle_list.append(temp)
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
                                        [str(bbox_first_frame + frame_index + 20), int(now_id)])
                                    plt.plot([real_pred_x, ego_prediction[pred_t][0]], [
                                        real_pred_y, ego_prediction[pred_t][1]], '-.', color='red', markersize=1)
                    plt.xlim(forever_present_x - 37, forever_present_x + 37)
                    plt.ylim(forever_present_y - 37, forever_present_y + 37)
                    for features in lane_feature:
                        xs, ys = np.vstack((features[0][:, : 2], features[0][-1, 3: 5]))[
                            :, 0], np.vstack((features[0][:, : 2], features[0][-1, 3: 5]))[:, 1]
                        plt.plot(xs, ys, '-.', color='gray')
                    plt.savefig(sav_path + '/' +
                                str(bbox_first_frame + frame_index) + '-' + str(bbox_first_frame + frame_index + 19 + future_len) + '.png')
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
        dict_metrics[type + '_ADE_1s'] = round(
            (ADE_1s / finished_scenario).item(), 3)
        dict_metrics[type + '_ADE_2s'] = round(
            (ADE_2s / finished_scenario).item(), 3)
        dict_metrics[type + '_ADE_3s'] = round(
            (ADE_3s / finished_scenario).item(), 3)
        dict_metrics[type + '_horizon10s'] = round(
            (horizon10s / finished_scenario).item(), 3)
        dict_metrics[type + '_horizon20s'] = round(
            (horizon20s / finished_scenario).item(), 3)
        dict_metrics[type + '_horizon30s'] = round(
            (horizon30s / finished_scenario).item(), 3)
        file = open('kalman_filter_' + str(float(future_len / 20)) + 's_' + val_or_test + "/results_" +
                    str(float(future_len/20)) + "s.txt", "a+")
        file.write(
            type + " error 1s: " + str(dict_metrics[type + '_horizon10s']) + 'm \n')
        file.write(
            type + " error 2s: " + str(dict_metrics[type + '_horizon20s']) + 'm \n')
        file.write(
            type + " error 3s: " + str(dict_metrics[type + '_horizon30s']) + 'm \n')
        file.write(
            type + " ADE 1s: " + str(dict_metrics[type + '_ADE_1s']) + 'm \n')
        file.write(
            type + " ADE 2s: " + str(dict_metrics[type + '_ADE_2s']) + 'm \n')
        file.write(
            type + " ADE 3s: " + str(dict_metrics[type + '_ADE_3s']) + 'm \n')
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--future_length', default='30',
                        type=str, required=True)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--val_or_test', type=str, default='test')
    args = parser.parse_args()
    main(args.future_length, args.val_or_test, args.data_path)
