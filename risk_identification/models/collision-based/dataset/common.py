import numpy as np
import os
import torch
from tqdm import tqdm

from .RiskBenchDataset import RiskBench_dataset

def get_dataset(root,dataset_setting, mode='train',scenario_type=['collision','interactive','non-interactive'],s_id_list=None):
    datasets = []
    for s_type in os.listdir(root):
        if s_type not in scenario_type:
            continue
        for s_id in os.listdir(os.path.join(root,s_type)):
            if s_id_list is not None:
                if s_id not in s_id_list:
                    continue
            map = parse_scenario_id(s_type,s_id)['map']
            if mode == 'val':
                if not map == '5':
                    continue
            elif mode == 'test':
                if not (map == '10' or map == 'A6' or map =='B3'):
                    continue
            dataset = RiskBench_dataset(root,s_type,s_id,**dataset_setting)
            if len(dataset)!=0:
                datasets.append(dataset)
    datasets = torch.utils.data.ConcatDataset(datasets)
    return datasets

def parse_scenario_id(s_type,s_id):
    s_id = s_id.split('_')
    if s_type == 'obstacle':
        return {'map':s_id[0],'ego_intention':s_id[3]}
    else:
        return {'map':s_id[0],'ego_intention':s_id[5],'interactor_intention':s_id[4]}

def iterate_dataset(root,types=['collision','interactive','obstacle','non-interactive']):
    out = {}
    for s_type in os.listdir(root):
        if s_type not in types:
            continue
        print(s_type)
        for s_id in tqdm(os.listdir(os.path.join(root,s_type)), position=0, leave=False):
            intention = parse_scenario_id(s_type,s_id)['ego_intention']
            num = len(os.listdir(os.path.join(root,s_type,s_id,'variant_scenario')))
            if intention not in out:
                out[intention] = num
            else:
                out[intention] += num
            # for variant in tqdm(os.listdir(os.path.join(root,s_type,s_id,'variant_scenario')), position=1, leave=False):
            #     variant_path = os.path.join(root,s_type,s_id,'variant_scenario',variant)
            #     yield variant_path
    return out
    