import os
import json
import numpy as np
import shutil
from distutils.dir_util import copy_tree

skip_list = []
MIN_AREA = 100
save_tracklet = True

def build_tracking(_type=None):
    """
        tracking_results Kx10 ndarray:
        [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
        e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
    """

    data_root = f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/{_type}"

    cnt = 0
    max_cnt = 0
    max_file = None
    
    for basic in os.listdir(data_root):

        basic_path = os.path.join(data_root, basic, 'variant_scenario')

        for variant in os.listdir(basic_path):

            variant_path = os.path.join(basic_path, variant)
            bbox_path = os.path.join(variant_path, 'bbox.json')
            bbox_json = json.load(open(bbox_path))

            tracking_results = []
            ego_id = json.load(open(variant_path+"/actor_attribute.json"))["ego_id"]

            for frame in bbox_json:
                bbox_cnt = 0
                
                for actor_id in bbox_json[frame]:

                    if ego_id % 65536 == int(actor_id) % 65536:
                        continue

                    bbox = bbox_json[frame][actor_id]
                    w = bbox[2]-bbox[0]
                    h = bbox[3]-bbox[1]
                    if w*h < MIN_AREA:
                        continue
                    tracking_results.append(
                                        [int(frame), int(actor_id)%65536, bbox[0], bbox[1], w, h, 1, -1, -1, -1])
                    bbox_cnt += 1
                
                
                if bbox_cnt > max_cnt:
                    max_cnt = bbox_cnt
                    max_file = variant_path

            if len(tracking_results) == 0:
                skip_list.append([_type, basic, variant])
                # print(_type, basic, variant)

            if save_tracklet:
                np.save(os.path.join(variant_path, 'tracking.npy'),
                        np.array(tracking_results))

            cnt += len(bbox_json)

            # print("=========================================")

    print(f"{_type} max_cnt: {max_cnt}, {max_file}")
    print(f"Total box: {cnt}")


if __name__ == '__main__':
    data_type = ['interactive', 'non-interactive', 'obstacle', 'collision'][:]

    for _type in data_type:
        build_tracking(_type)

    # with open("skip_scenario.json", "w") as f:
    #     json.dump(skip_list, f, indent=4)