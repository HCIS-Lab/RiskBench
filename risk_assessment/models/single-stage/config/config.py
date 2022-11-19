import os.path as osp
import os
from collections import OrderedDict
import socket
import getpass
machine_name = socket.gethostname()
username = getpass.getuser()

__all__ = ['parse_args']


def parse_args(parser):
    parser.add_argument(
        '--data_root', default=osp.expanduser(''), type=str)
    parser.add_argument(
        '--save_path', default=osp.expanduser('/home/cli/exp_trn'), type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)

    args = parser.parse_args()

    args.test_session_set = []
    args.train_session_set = []
    args.class_index = ['go', 'stop']
    args.cause = 'all'

    if args.cause == "all":
        types = ['interactive', 'non-interactive', 'obstacle']
    else:
        types = [args.cause]

    for cause in types:
        data_root = f'/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/{cause}'
        for basic_scene in os.listdir(data_root):
            basic_scene_path = osp.join(
                data_root, basic_scene, 'variant_scenario')
            
            if not os.path.isdir(basic_scene_path):
                print(basic_scene_path)
                continue

            for var_scene in os.listdir(basic_scene_path):
                var_scene_path = osp.join(basic_scene_path, var_scene)

                if basic_scene[:2] == '5_':     # "5_" or "10"
                    args.test_session_set.append([var_scene_path, cause])
                elif basic_scene[:2] != '10':
                    args.train_session_set.append([var_scene_path, cause])


    # print(len(args.train_session_set))
    # print(len(args.test_session_set))
  
    return args
