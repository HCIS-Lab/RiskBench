import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import sys
import json
import argparse


def main(future_len, val_or_test, data_path, method):
    method = 'kalman_filter'
    specific_type = 'non-interactive'
    future_len = 30
    val_or_test = 'test'
    T_num = 1
    all_type = ['interactive', 'non-interactive', 'obstacle', 'collision']
    for data_type in all_type:
        folder = data_path + '/' + data_type + '/'
        final_d = {}
        for scenario_name in sorted(os.listdir(folder)):
            weather_filename = folder + scenario_name + '/'
            for weather_type in sorted(os.listdir(weather_filename)):
                if weather_type == 'variant_scenario':
                    continue
                all_txt_list = []
                txt_filename = folder + scenario_name + '/' + weather_type + '/'
                print(method, scenario_name, weather_type)
                for filename in os.listdir(txt_filename):
                    if filename == 'prediction':
                        continue
                    if filename == 'prediction_test':
                        continue
                    elif filename == 'collsion_description':
                        continue
                    all_txt_list.append(
                        int(filename.split(".")[0].split("-")[1]))
                max_time = np.array(all_txt_list)
                file_t = np.max(max_time)
                for filename in os.listdir(folder + scenario_name + '/variant_scenario/' + weather_type + '/bbox/front/'):
                    all_txt_list.append(
                        int(filename.split(".")[0]))
                bbox_time_list = np.array(all_txt_list)
                bbox_first_frame = np.min(bbox_time_list)

                if scenario_name == '5_t3-9_0_m_u_l_0' and (weather_type == 'MidRainyNight_high_' or weather_type == 'MidRainyNoon_low_'):
                    bbox_first_frame = 64
                each_file = folder + scenario_name + '/' + weather_type + \
                    '/collsion_description/' + scenario_name + '.json'
                if os.path.isfile(each_file):
                    with open(each_file) as f:
                        file_d = json.load(f)
                        d = {}
                        for i in range(bbox_first_frame, file_t):
                            temp_d = {}
                            key = str(i)
                            value = file_d.get(key)
                            if value == None:
                                temp_d[str(False)] = {}
                                d[key] = temp_d
                            else:
                                temp_d[str(True)] = value
                                d[key] = temp_d
                        final_key = scenario_name + '_' + weather_type
                        final_d[final_key] = d
                else:
                    d = {}
                    for i in range(bbox_first_frame, file_t):
                        temp_d = {}
                        temp_d[str(False)] = {}
                        d[str(i)] = temp_d
                    final_key = scenario_name + '_' + weather_type
                    final_d[final_key] = d
        file = open(method + '_final_output/' + method +
                    '_' + str(float(future_len / 20)) + 's_' + data_type + '_' + val_or_test + '_T=' + str(T_num) + '.json', "w")
        json.dump(final_d, file, indent=4)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--future_length', default='30',
                        type=str, required=True)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--val_or_test', type=str, default='test')
    parser.add_argument('--method', type=str, default='social_gan')
    args = parser.parse_args()
    main(args.future_length, args.val_or_test, args.data_path, args.method)
