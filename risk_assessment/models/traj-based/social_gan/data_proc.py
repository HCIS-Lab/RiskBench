import os
import csv
import json
import argparse


def data_proc(args):
    data_path = args.input_path
    for scenario in os.listdir(data_path):
        scenario_path = os.path.join(data_path, scenario)
        scenario_v_path = os.path.join(scenario_path, 'variant_scenario')
        for variant in os.listdir(scenario_v_path):
            variant_traj_path_list = [os.path.join(scenario_v_path, variant)]
            variant_traj_path_list.append('trajectory_frame')
            variant_traj_path_list.append('{}.csv'.format(scenario))
            variant_traj_path = os.path.join(*variant_traj_path_list)
            print('Processing scenario: {}, variant: {} now'.format(
                scenario, variant))
            traj_txt_list = []
            with open(args.dynamic_description_path) as f:
                data = json.load(f)
                variant_ego_id = str(data['player'])
            with open(variant_traj_path, newline='') as traj_csv:
                rows = csv.reader(traj_csv)

                ego_info = []
                for row in rows:
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

            path_list = args.output_path
            txt_name = scenario + '-with-' + variant + '.txt'
            path_list.append(txt_name)
            txt_path = os.path.join(*path_list)
            f = open(txt_path, 'w')
            for ego_traj in ego_info:
                ego_traj_str = ego_traj[0] + '\t' + ego_traj[1] + \
                    '\t' + ego_traj[2] + '\t' + ego_traj[3] + '\n'
                f.write(ego_traj_str)
            for traj in traj_txt_list:
                traj_str = traj[0] + '\t' + traj[1] + \
                    '\t' + traj[2] + '\t' + traj[3] + '\n'
                # print(traj_str)
                f.write(traj_str)
            f.close()


def main(args):
    data_proc(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dynamic_description_path', type=str)
    args = parser.parse_args()

    main(args)
