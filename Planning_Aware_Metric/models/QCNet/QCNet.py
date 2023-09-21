import os
from itertools import chain
from itertools import compress
from typing import Any, Dict, List,Optional

import numpy as np
import pandas as pd
import math
import torch
from torch_geometric.data import HeteroData
from models.QCNet.predictors import QCNet

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
        return now_id
    
def get_agent_features(df: pd.DataFrame, num_historical_steps: int, dim=2) -> Dict[str, Any]:
    '''
    df: FRAME, TRACK_ID, OBJECT_TYPE (vehicle, pedestrian, obstacle), X, Y, VELOCITY_X, VELOCITY_Y, YAW
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_types = ['vehicle', 'pedestrian', 'obstacle']
    vector_repr = True
    predict_unseen_agents = False
    if not predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
        # historical_df = df[df['timestep'] < self.num_historical_steps]
        agent_ids = list(df['TRACK_ID'].unique())
        df = df[df['TRACK_ID'].isin(agent_ids)]
    else:
        agent_ids = list(df['TRACK_ID'].unique())
    df.loc[df.OBJECT_TYPE == 'EGO', 'OBJECT_TYPE'] = 'vehicle'
    df.loc[df.OBJECT_TYPE == 'static.prop.trafficcone01', 'OBJECT_TYPE'] = 'obstacle'
    df.loc[df.OBJECT_TYPE == 'static.prop.streetbarrier', 'OBJECT_TYPE'] = 'obstacle'
    df.loc[df.OBJECT_TYPE == 'static.prop.trafficwarning', 'OBJECT_TYPE'] = 'obstacle'

    num_agents = len(agent_ids)

    # initialization
    valid_mask = torch.zeros(num_agents, num_historical_steps, dtype=torch.bool).to(device)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool).to(device)
    predict_mask = torch.zeros(num_agents, num_historical_steps, dtype=torch.bool).to(device)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8).to(device)
    position = torch.zeros(num_agents, num_historical_steps, dim, dtype=torch.float).to(device)
    heading = torch.zeros(num_agents, num_historical_steps, dtype=torch.float).to(device)
    velocity = torch.zeros(num_agents, num_historical_steps, dim, dtype=torch.float).to(device)

    for track_id, track_df in df.groupby('TRACK_ID'):
        agent_idx = agent_ids.index(track_id)
        # agent_steps = track_df['timestep'].values

        # valid_mask[agent_idx, agent_steps] = True
        valid_mask[agent_idx, :] = True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        # predict_mask[agent_idx, agent_steps] = True
        predict_mask[agent_idx, :] = True
        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: num_historical_steps] = (
                    valid_mask[agent_idx, :num_historical_steps - 1] &
                    valid_mask[agent_idx, 1: num_historical_steps])
            valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :num_historical_steps] = False
        if not current_valid_mask[agent_idx]:
            predict_mask[agent_idx, num_historical_steps:] = False

        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = agent_types.index(track_df['OBJECT_TYPE'].values[0])
        # position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
        #                                                                     track_df['position_y'].values],
        #                                                                     axis=-1)).float()
        # heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
        # velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
        #                                                                     track_df['velocity_y'].values],
        #                                                                     axis=-1)).float()
        position[agent_idx, :, :2] = torch.from_numpy(np.stack([track_df['X'].values,
                                                                            track_df['Y'].values],
                                                                            axis=-1)).float()
        heading[agent_idx, :] = torch.from_numpy(track_df['YAW'].values).float()
        velocity[agent_idx, :, :2] = torch.from_numpy(np.stack([track_df['VELOCITY_X'].values,
                                                                            track_df['VELOCITY_Y'].values],
                                                                            axis=-1)).float()

    return {
        'num_nodes': num_agents,
        'valid_mask': valid_mask.to(device),
        'predict_mask': predict_mask.to(device),
        'id': agent_id,
        'type': agent_type.to(device),
        'position': position.to(device),
        'heading': heading.to(device),
        'velocity': velocity.to(device),
    }

def inference(input_df, model, num_historical_steps=20, output_dim=2):
    '''
    Args:
        input_df: columns=[FRAME, TRACK_ID, OBJECT_TYPE (vehicle, pedestrian, obstacle), X, Y, VELOCITY_X, VELOCITY_Y, YAW]
        model: pretrained model

    Returns:
        out_df: columns=['FRAME', 'TRACK_ID', 'X', 'Y']
    '''
    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Preprocessing
    data = {}
    data['agent'] = get_agent_features(input_df, num_historical_steps)
    #print("before:", data)
    data = HeteroData(data)
    data = data.to(device)
    model = model.to(device)
    #print("after:", data)
    # print(data['agent']['num_nodes'])
    # print(data['agent']['valid_mask'])
    # print(data['agent']['predict_mask'])
    # print(data['agent']['id'])
    # print(data['agent']['type'])
    # print(data['agent']['position'])
    # print(data['agent']['heading'])
    # print(data['agent']['velocity'])

    # inference
    pred = model(data)
    if model.output_head:
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                    pred['loc_refine_head'],
                                    pred['scale_refine_pos'][..., :output_dim],
                                    pred['conc_refine_head']], dim=-1)
    else:
        traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                    pred['scale_refine_pos'][..., :output_dim]], dim=-1)

    eval_mask = data['agent']['type'] < 3

    origin_eval = data['agent']['position'][eval_mask, num_historical_steps - 1]
    theta_eval = data['agent']['heading'][eval_mask, num_historical_steps - 1]
    cos, sin = theta_eval.cos(), theta_eval.sin()
    rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
    traj_eval = traj_eval.detach().cpu().numpy() #.cpu().numpy()
    traj_pred = traj_eval.squeeze() # shape: [n, 30, 2]

    ##################################################################3
    #agent_ids = list(compress(list(chain(*data['agent']['id'])), eval_mask)) # shape: n
    agent_ids = data['agent']['id']

    # To dataframe
    num_agents = len(agent_ids)
    num_future_frame = traj_pred.shape[1]
    out_np_arr = np.zeros((num_agents*num_future_frame, 4)) # shape: [n*30, 4]
    start_frame = list(input_df['FRAME'].unique())[-1] + 1
    frame_list = np.arange(start_frame, start_frame+30)
    for idx, traj in enumerate(traj_pred):
        indices = list(range(idx, num_future_frame*num_agents, num_agents))
        # print("indices:", indices)
        # print("frame_list:", frame_list)
        # print("out_np_arr[indices]:", out_np_arr[indices].shape, out_np_arr[indices])
        # out_np_arr[indices][0] = frame_list
        # out_np_arr[indices][1] = agent_ids[idx]
        # out_np_arr[indices][2:] = traj
        out_np_arr[indices, 0] = frame_list
        out_np_arr[indices, 1] = agent_ids[idx]
        out_np_arr[indices, 2:] = traj


    out_df = pd.DataFrame(data=out_np_arr, columns=['FRAME', 'TRACK_ID', 'X', 'Y'])
    out_df['FRAME'] = out_df['FRAME'].astype("int")
    out_df['TRACK_ID'] = out_df['TRACK_ID'].astype("int").astype("str")
    out_df = pd.concat([input_df, out_df])
    out_df['X'] = out_df['X'].astype("float")
    out_df['Y'] = out_df['Y'].astype("float")
    out_df = out_df.drop(columns=['OBJECT_TYPE', 'VELOCITY_X', 'VELOCITY_Y', 'YAW'])
    out_df = out_df.reset_index(drop=True)
    #print("out_df:", out_df)
    return out_df

def QCNet_inference(vehicle_list, specific_frame, variant_ego_id, pedestrian_id_list, vehicle_id_list, obstacle_dict):

    vehicle_length = 4.7
    vehicle_width = 2
    pedestrian_length = 0.8
    pedestrian_width = 0.8
    # vehicle_length = 10
    # vehicle_width = 10
    # pedestrian_length = 8
    # pedestrian_width = 8
    agent_area = [[vehicle_length, vehicle_width],
                [pedestrian_length, pedestrian_width]]


    #config
    future_len = 30
    #print(vehicle_list)

    ckpt_path = './models/weights/QCNet/epoch38.ckpt'
    model = {
        'QCNet': QCNet,
    }['QCNet'].load_from_checkpoint(checkpoint_path=ckpt_path)
    # print(pd.concat(vehicle_list))
    inference_df = pd.concat(vehicle_list)
    
    #print("before ids:", len(list(set(inference_df['TRACK_ID'].values))), list(set(inference_df['TRACK_ID'].values)))
    
    #print("obstacle_dict:", obstacle_dict)
    for track_id, rest_df in inference_df.groupby('TRACK_ID'):
        if track_id in vehicle_id_list:
            inference_df.loc[inference_df.OBJECT_TYPE == 'ACTOR', 'OBJECT_TYPE'] = 'vehicle'
        elif track_id in pedestrian_id_list:
            inference_df.loc[inference_df.OBJECT_TYPE == 'ACTOR', 'OBJECT_TYPE'] = 'pedestrian'
    inference_df.loc[inference_df.OBJECT_TYPE == 'ACTOR', 'OBJECT_TYPE'] = 'vehicle'
    temp_df = inference(inference_df, model)
    temp_df['TRACK_ID'] = temp_df['TRACK_ID'].astype("int")
    
    #print("ids:", len(list(set(temp_df['TRACK_ID'].values))), list(set(temp_df['TRACK_ID'].values)))
    
    #print("temp_df:", temp_df)
    vehicle_list = []
    for track_id, remain_df in temp_df.groupby('TRACK_ID'):
        vehicle_list.append(remain_df)
        # if int(track_id) == 97712:
        #     print(track_id, remain_df)

    risky_vehicle_list = []
    ego_prediction = np.zeros((future_len, 2))
    #print("vehicle_list:", vehicle_list)
    for n in range(len(vehicle_list)):
        vl = vehicle_list[n].to_numpy()
        now_id = int(vl[0][1])
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
    for val_vehicle_num in range(len(vehicle_list)):
        #ego, agent, other = 0, 0, 0
        vl = vehicle_list[val_vehicle_num].to_numpy()
        # vl : frame, id, x, y
        #print("vl:", vl)
        now_id = vl[0][1]
        if str(int(now_id)) in pedestrian_id_list:
            agent_type = 1
        elif str(int(now_id)) in vehicle_id_list:
            agent_type = 0
        elif int(now_id) == int(variant_ego_id):
            agent_type = 0
        #print(str(int(now_id)), type(obstacle_id_list[0]), obstacle_id_list)
        for pred_t in range(future_len - 1):
            if int(now_id) == int(variant_ego_id):
                continue
            real_pred_x = vl[pred_t + 20][2]
            real_pred_x_next = vl[pred_t + 21][2]
            real_pred_y = vl[pred_t + 20][3]
            real_pred_y_next = vl[pred_t + 21][3]
            #print(now_id, obstacle_id_list)
            
            #if str(int(now_id)) in obstacle_id_list:
            if int(now_id) in list(obstacle_dict):
                # temp = obstacle_collision(vehicle_length, vehicle_width, 2, 2, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                #                                  pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                
                # if vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.trafficcone01':
                #     temp = obstacle_collision(vehicle_length, vehicle_width, 0.85, 0.85, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                #                                 pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                # elif vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.streetbarrier':
                #     temp = obstacle_collision(vehicle_length, vehicle_width, 1.25, 0.375, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                #                                 pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                # elif vehicle_list[val_vehicle_num].OBJECT_TYPE[0] == 'static.prop.trafficwarning':
                #     temp = obstacle_collision(vehicle_length, vehicle_width, 3, 2.33, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                #                                 pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                if obstacle_dict[now_id] == 'static.prop.trafficcone01':
                    temp = obstacle_collision(vehicle_length, vehicle_width, 0.85, 0.85, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                elif obstacle_dict[now_id] == 'static.prop.streetbarrier':
                    temp = obstacle_collision(vehicle_length, vehicle_width, 1.25, 0.375, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)
                elif obstacle_dict[now_id] == 'static.prop.trafficwarning':
                    temp = obstacle_collision(vehicle_length, vehicle_width, 3, 2.33, ego_prediction[pred_t][0], ego_prediction[pred_t][1], ego_prediction[
                                                pred_t + 1][0], ego_prediction[pred_t + 1][1], vl[0][2], vl[0][3], 0.0, vehicle_length, vehicle_width, specific_frame, pred_t, now_id)

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
                    # risky_vehicle_list.append(
                    #     [specific_frame, int(vl[0][1])])
                    risky_vehicle_list.append(
                        vl[0][1])


    # file_d = {}
    # for frame, id in risky_vehicle_list:
    #     if frame in file_d:
    #         if str(id) in file_d[frame]:
    #             continue
    #         else:
    #             file_d[frame].append(str(id))
    #     else:
    #         file_d[frame] = [str(id)]
    # if specific_frame in file_d:
    #     risky_id = file_d[specific_frame]
    # else:
    #     risky_id = []
    risky_vehicle_list = list(set(risky_vehicle_list))

    return risky_vehicle_list

if __name__== '__main__':
    input_file = ''
    input_df = pd.read_csv(input_file)
    ckpt_path = 'lightning_logs/version_14/checkpoints/epoch\=38-step\=402948.ckpt'
    model = {
        'QCNet': QCNet,
    }['QCNet'].load_from_checkpoint(checkpoint_path=ckpt_path)
    inference(input_df, model)