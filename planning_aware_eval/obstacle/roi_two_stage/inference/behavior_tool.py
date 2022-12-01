import os.path as osp
import os
import json
import numpy as np
import shutil
from distutils.dir_util import copy_tree


def build_tracking(start_frame, data_path='test_data', save_tracklet = True):

    # use only 5 frame 
    # start_frame-5 ~ start_frame
    
    bbox_path = osp.join(data_path, 'bbox/front')
    rgb_path = osp.join(data_path, 'rgb/front')

    tracking_results = []
    # bbox_list = os.listdir(bbox_path)
    bbox_list = []
    for index in range(4,-1,-1):
        save_id = start_frame - index
        bbox_list.append(f'{save_id:08d}.json')

    # print(bbox_list)

    for bbox_file in sorted(bbox_list):
        # print(bbox_file)
        id_json = osp.join(bbox_path, bbox_file)

        if osp.isfile(id_json):
            frame_id = int(bbox_file.split('.')[0])

            if osp.isfile(osp.join(rgb_path, f'{frame_id:08d}.png')):
                json_file = open(id_json)
                data = json.load(json_file)
                json_file.close()

                for info in data:
                    id = info['actor_id']
                    bbox = info['box']
                    cls = info['class']
                    if cls in (4, 10, 20):
                        w = bbox[2]-bbox[0]
                        h = bbox[3]-bbox[1]
                        tracking_results.append(
                            [frame_id, id, bbox[0], bbox[1], w, h, 1, -1, -1, -1])
                    else:
                        print(id_json)
                        print(cls)

    if save_tracklet:
        np.save(osp.join(data_path, 'tracking.npy'),
                np.array(tracking_results))

    print("Saved tracking.npy")

if __name__ == '__main__':
    build_tracking()

    pass
