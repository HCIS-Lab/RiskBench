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
import time

scenario_name = '10_i-1_1_c_f_f_1_rl'


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--data_path', type=str, default='test_data')
    parser.add_argument('--model_path', type=str,
                        default='gan_test_with_model_all.pt')
    parser.add_argument('--infer_data', type=str)
    parser.add_argument('--type', default='obstacle', type=str)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--pred_len', default=30, type=int)
    return parser.parse_args()


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


def get_generator(checkpoint, new_args):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=new_args.pred_len,
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


def evaluate(args, ego, traj_txt_list, generator, num_samples, infer_file):
    data = []
    #print(ego, traj_txt_list)
    ego_np = np.array(ego)
    # print(ego_np)
    other_np = np.array(traj_txt_list)
    # print(other_np)
    #other_np = np.sort(other_np, order='name')
    data = np.concatenate((ego_np, other_np))
    # print(data)

    # print(data[i])

    # with open('test_data/trajectory_frame/temp.txt') as f:
    #    for line in f.readlines():
    #        line = line.strip().split('\t')
    #        line = [float(i) for i in line]
    #        data.append(line)
    #data = np.array(data)
    #data = np.array(traj_txt_list)
    frames = np.unique(data[:, 0]).tolist()
    frames = [int(i) for i in frames]
    #frames = frames.astype(np.int)
    frames.sort()
    frames = [str(i) for i in frames]
    frame_data = []

    ped_num = None
    for frame in frames:
        data_now = data[frame == data[:, 0], :]
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

    print(df)
    return df


def main(config):

    data_path = config.data_path
    data_type = config.type
    specific_frame = config.frame
    future_len = config.pred_len

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
            # last_frame = obj_data['interactive frame'][1]
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
    # print(bbox_first_frame, vehicle_list[0])

    # 把Basic scenario存到的ID拿掉
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

    # vehicle_list丟到preprocessing, 生成social gan專用的txt檔

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
    """
    f = open(data_path + '/trajectory_frame/temp.txt', 'w')
    for ego_traj in ego_info:
        ego_traj_str = str(ego_traj[0]) + '\t' + str(ego_traj[1]) + \
            '\t' + str(ego_traj[2]) + '\t' + str(ego_traj[3]) + '\n'
        f.write(ego_traj_str)
    for traj in traj_txt_list:
        traj_str = str(traj[0]) + '\t' + str(traj[1]) + \
            '\t' + str(traj[2]) + '\t' + str(traj[3]) + '\n'
        f.write(traj_str)
    f.close()
    """
    # social gan開始inference
    checkpoint = torch.load(config.model_path)
    start_time = time.time()
    generator = get_generator(checkpoint, config)
    print(time.time() - start_time)
    _args = AttrDict(checkpoint['args'])
    _args.dataset_name = config.type
    _args.skip = 1
    _args.pred_len = config.pred_len
    # txt_path = data_path + '/trajectory_frame/temp.txt'

    temp_df = evaluate(_args, ego_info, traj_txt_list, generator,
                       config.num_samples, config.infer_data)

    # 計算碰撞與否
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
        # ego, agent, other = 0, 0, 0
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
    # python social_gan_one_step.py --type interactive --pred_len 30 --frame 107
    config = parse_config()
    main(config)
