import os
import csv
import json
import argparse
import sys
import torch
import numpy as np
import math
import pandas as pd
from collections import defaultdict
from attrdict import AttrDict
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path



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


def get_generator(checkpoint):

    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=30 ,#new_args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args,  ego, traj_txt_list, generator, num_samples, ):
    # data = []
    # with open(txt_path) as f:
    #     for line in f.readlines():
    #         line = line.strip().split('\t')
    #         line = [float(i) for i in line]
    #         data.append(line)



    data = []
    #print(ego, traj_txt_list)
    ego_np = np.array(ego)
    # print(ego_np)
    other_np = np.array(traj_txt_list)
    # print(other_np)
    #other_np = np.sort(other_np, order='name')
    data = np.concatenate((ego_np, other_np))


    #data = np.array(data)
    frames = np.unique(data[:, 0]).tolist()
    frames = [int(i) for i in frames]
    #frames = frames.astype(np.int)
    frames.sort()
    frames = [str(i) for i in frames]
    # print(frames)
    frame_data = []
    ped_num = None
    for frame in frames:
        # print(frame, data[:, 0])
        data_now = data[float(frame) == data[:, 0], :]
        # print(data_now)
        #data_now = data[data[:, 0] == frame, :]
        if ped_num == None:
            ped_num = data_now.shape[0]
        if data_now.shape[0] == ped_num:
            frame_data.append(data_now)
    obs_traj = np.zeros((20, ped_num, 2))
    obs_traj_rel = np.zeros((20, ped_num, 2))
    seq_start_end = np.array([[0, ped_num]])
    ped_list = np.zeros((20, ped_num))
    for i, frame in enumerate(frame_data):
        if i > 19:
            break

        frame_sort = frame[frame[:, 1].argsort()]
        for j, ped in enumerate(frame_sort):
            obs_traj[i, j] = np.array([ped[2], ped[3]])
            ped_list[i, j] = ped[1]
    obs_traj_rel[:, 1:] = \
        obs_traj[:, 1:] - obs_traj[:, :-1]

    obs_traj = torch.from_numpy(obs_traj).to(torch.float).cuda()
    obs_traj_rel = torch.from_numpy(obs_traj_rel).to(torch.float).cuda()
    seq_start_end = torch.from_numpy(seq_start_end).to(torch.int).cuda()

    pred_traj_fake_rel = generator(
        obs_traj, obs_traj_rel, seq_start_end
    )
    pred_traj_fake = relative_to_abs(
        pred_traj_fake_rel, obs_traj[-1]
    )
    new_ped_list = np.sort(ped_list[0])
    batch_num = 1
    df_list = []
    obs_traj_new = obs_traj.reshape(
        20, batch_num, -1, 2).permute(1, 0, 2, 3)
    pred_traj_fake_new = pred_traj_fake.reshape(
        args.pred_len, batch_num, -1, 2).permute(1, 0, 2, 3)
    traj_mix = torch.cat(
        (obs_traj_new, pred_traj_fake_new), 1).detach().cpu().numpy()
    for i, frame_now in enumerate(traj_mix[0]):
        for j, actor in enumerate(frame_now):
            df_list.append([i, int(new_ped_list[j]), actor[0], actor[1]])
    df = pd.DataFrame(df_list, columns=['FRAME', 'TRACK_ID', 'X', 'Y'])

    return df



def socal_gan_inference(vehicle_list, specific_frame, variant_ego_id, pedestrian_id_list, vehicle_id_list, _args, generator ):


    future_len = 30 

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





    df = pd.concat(vehicle_list)
    traj_txt_list = []
    ego_info = []
    for index, row in df.iterrows():
        if row[0] == 'FRAME' or row[2] == 'AGENT' or row[2].split('.')[0] == 'actor':
            continue
        if row[1] == variant_ego_id:
            ego_info.append([row[0], row[1], row[3], row[4]])
            continue
        traj_row_list = []
        traj_row_list.append(row[0])
        traj_row_list.append(row[1])
        traj_row_list.append(row[3])
        traj_row_list.append(row[4])
        traj_txt_list.append(traj_row_list)



    # f = open('./sgan/temp.txt', 'w')
    # for ego_traj in ego_info:
    #     ego_traj_str = str(ego_traj[0]) + '\t' + str(ego_traj[1]) + \
    #         '\t' + str(ego_traj[2]) + '\t' + str(ego_traj[3]) + '\n'
    #     f.write(ego_traj_str)
    # for traj in traj_txt_list:
    #     traj_str = str(traj[0]) + '\t' + str(traj[1]) + \
    #         '\t' + str(traj[2]) + '\t' + str(traj[3]) + '\n'
    #     f.write(traj_str)
    # f.close()

    # social gan開始inference
    #checkpoint = torch.load("./social_gan/gan_test_with_model_all.pt")
    #generator = get_generator(checkpoint)



    #_args = AttrDict(checkpoint['args'])
    #_args.dataset_name = "interactive" 
    #_args.skip = 1
    #_args.pred_len = 30 

    #temp_df = evaluate(_args, './sgan/temp.txt', generator, 1 )
    temp_df = evaluate(_args, ego_info, traj_txt_list, generator, 1)






    # 計算碰撞與否

    risky_vehicle_list = []
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

            # 這個環節是為了從ID中找出那個障礙物是哪一種類別(如柵欄或是交通錐)，要找這個是因為每個障礙物面積都不一樣，為了帶進函式obstacle_collision去計算是否碰撞
            # 若即時拿obstacle資料，可以直接拿type是哪一種，跟object_type_list對照找出對應的面積
           
 
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
        risky_id = []
    # print(file_d)
    print(' risky_id:',  risky_id)
    return risky_id


