import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import json
import argparse


def main(args):
    folder = os.path.join(args.input_path, args.data_type)
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
            sav_path = os.path.join(
                args.output_path, args.data_type, scenario_name, 'variant_scenario', weather_type)
            traj_path = sav_path + '/trajectory_frame'
            if not os.path.exists(traj_path):
                os.makedirs(traj_path)
            id_source = folder + scenario_name + '/variant_scenario/' + \
                weather_type + '/dynamic_description.json'
            id_destination = sav_path + '/dynamic_description.json'
            shutil.copyfile(id_source, id_destination)
            topo_source = folder + scenario_name + '/variant_scenario/' + \
                weather_type + '/' + scenario_name + '.npy'
            topo_destination = sav_path + '/' + scenario_name + '.npy'
            shutil.copyfile(topo_source, topo_destination)
            bbox_source = '../mnt/Final_Dataset/dataset/' + args.data_type + '/' + \
                scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/'
            bbox_path = sav_path + '/bbox/front'
            shutil.copytree(bbox_source, bbox_path)
            ego_list = []
            actor_list = []
            actor_id_list = []
            if args.data_type == 'interactive' or args.data_type == 'collision':
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
                    traj_df.to_csv(traj_path + '/' +
                                   scenario_name + '.csv', index=False)

            if args.data_type == 'non-interactive':
                traj_df = pd.read_csv(folder + scenario_name + '/variant_scenario/' +
                                      weather_type + '/trajectory_frame/' + scenario_name + '.csv')
                with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/dynamic_description.json') as f:
                    data = json.load(f)
                    variant_ego_id = str(data['player'])
                    traj_df.loc[traj_df.TRACK_ID ==
                                variant_ego_id, 'OBJECT_TYPE'] = 'EGO'
                    traj_df.to_csv(traj_path + '/' +
                                   scenario_name + '.csv', index=False)

            if has_obstacle and args.data_type == 'obstacle':
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
                        if scenario_name.split('_')[0] != '5' and scenario_name.split('_')[0] != '10':
                            continue

                        with open(folder + scenario_name + '/variant_scenario/' + weather_type + '/obstacle_info.json') as f:
                            obj_data = json.load(f)
                            obj_d = obj_data['interactive_obstacle_list']
                        right_id_obj = pd.DataFrame()
                        f = open(
                            sav_path + '/GT.txt', 'w')
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
                        final_df = pd.concat([traj_df, right_id_obj])
                        final_df.reset_index(drop=True)
                        final_df.to_csv(traj_path + '/' +
                                        scenario_name + '.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_type', type=str, default='interactive')
    args = parser.parse_args()

    main(args)
