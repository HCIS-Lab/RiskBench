import os
import json
import numpy as np
from datetime import datetime

__all__ = ['parse_args', 'read_data',
           'create_ROI_result', 'create_ckpt_result']


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)

    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def create_ROI_result(args):

    ckpt = args.ckpt_path.split('/')[-2]
    epoch = int(args.ckpt_path.split('/')[-1].split('-')[1].split('.')[0])
    args.roi_path = f"{args.roi_root}/{ckpt}_{epoch}_{args.data_type}.json"

    with open(args.roi_path, "w") as f:
        json.dump({}, f, indent=4)

    return args


def create_ckpt_result(args):

    copy_args = vars(args).copy()
    log = {"args": copy_args}
    print(log)

    year, month, day, hour, minute, second = get_current_time()
    formated_time = f"{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}"
    args.result_path = os.path.join(args.result_root, f"{formated_time}.json")
    args.ckpt_folder = os.path.join(args.ckpt_root, formated_time)
    args.log_folder = os.path.join(args.log_root, formated_time)

    with open(args.result_path, "w") as f:
        json.dump([log], f, indent=4)

    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    if not os.path.isdir(args.log_folder):
        os.makedirs(args.log_folder)

    return args


def read_data(args, valiation=False):

    town = ["10", "A6", "B3"] if not valiation else ["5_"]

    args.train_session_set = []
    args.test_session_set = []
    args.tracking_list = []

    if args.data_type == "all":
        data_types = ['interactive', 'non-interactive',
                      'obstacle', "collision"][:3]
    else:
        data_types = [args.data_type]

    skip_list = json.load(open("./datasets/skip_scenario.json"))

    for data_type in data_types:
        _type_path = os.path.join(args.data_root, data_type)

        for basic in sorted(os.listdir(_type_path)):
            basic_path = os.path.join(_type_path, basic, 'variant_scenario')

            for variant in os.listdir(basic_path):
                if [data_type, basic, variant] in skip_list:
                    continue
                variant_path = os.path.join(basic_path, variant)

                if basic[:2] in town:
                    tracking_results = np.load(
                        os.path.join(variant_path, "tracking.npy"))
                    args.tracking_list.append(tracking_results)
                    args.test_session_set.append((basic, variant, data_type))

                elif not basic[:2] in ["10", "A6", "B3", "5_"]:
                    args.train_session_set.append((basic, variant, data_type))

    if args.test_sample > 0:
        args.train_session_set = args.train_session_set[:args.test_sample]
        args.test_session_set = args.test_session_set[:args.test_sample]
        args.tracking_list = args.tracking_list[:args.test_sample]

    return args


def parse_args(parser):

    parser.add_argument(
        '--data_root', default=f"/PATH/TO/RiskBench_Dataset", type=str)
    parser.add_argument(
        '--behavior_root', default=f"./datasets/behavior", type=str)
    parser.add_argument(
        '--state_root', default=f"./datasets/state", type=str)
    parser.add_argument(
        '--result_root', default=f'./results', type=str)
    parser.add_argument(
        '--ckpt_root', default=f'./checkpoints', type=str)
    parser.add_argument(
        '--log_root', default=f'./logs', type=str)
    parser.add_argument(
        '--roi_root', default=f'./ROI', type=str)
    parser.add_argument('--class_index', default=['go', 'stop'], type=list)
    parser.add_argument('--test_sample', default=0, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--num_box', default=25, type=int)
    parser.add_argument('--img_size', default=(256, 640), type=tuple)
    parser.add_argument('--img_resize', default=(256, 640), type=tuple)

    args = parser.parse_args()
    if len(args.phases) != 1:
        args = create_ckpt_result(args)

    return args
