import numpy as np
import os
import cv2
import imageio
import json

attr_cls = {"4-Way": 0, "T-intersection-A": 0, "T-intersection-B": 0, "T-intersection-C": 0, "Straight": 0, "Roundabout": 0,
            "Low": 1, "Mid": 1, "High": 1,
            "Clear": 1, "Cloudy": 1, "Wet": 1, "Rain": 1, "Noon": 1, "Sunset": 1, "Night": 1,
            "Car": 2, "Truck": 2, "Bike": 2, "Motor": 2, "Pedestrian": 2, "4-Wheeler": 2, "2-Wheeler": 2,
            "Obstacle": 3, "Illegal Parking": 3}

attr_to_token = {"4-Way": "i", "T-intersection-A": "t1-", "T-intersection-B": "t2-", "T-intersection-C": "t3-",
                 "Straight": "s", "Roundabout": "r",
                 "Low": "low", "Mid": "mid", "High": "high",
                 "Clear": "Clear", "Cloudy": "Cloudy", "Wet": "Wet", "Rain": "Rain", "Noon": "Noon", "Sunset": "Sunset", "Night": "Night",
                 "Car": "c", "Truck": "t", "Bike": "b", "Motor": "m", "Pedestrian": "p", "4-Wheeler": "4", "2-Wheeler": "2",
                 "Obstacle": ["0", "1", "2"], "Illegal Parking": ["3"]}


def filter_roi_scenario(data_type, method, attr_list="", model_root="./model"):

    model_path = os.path.join(model_root, method, f"{data_type}.json")
    roi_result = json.load(open(model_path))
    
    if attr_list == "All":
        return roi_result

    # filter scenario
    new_roi_result = {}

    for attr in attr_list:

        if attr_cls[attr] == 3 and data_type != "obstacle":
            continue

        copy_roi_result = roi_result.copy()
        all_scnarios = list(copy_roi_result.keys())

        for scenario_weather in all_scnarios:

            is_del = False

            if attr_cls[attr] == 0:
                topo = scenario_weather.split('_')[1]
                if not attr_to_token[attr] in topo:
                    is_del = True

            elif attr_cls[attr] == 1:
                if not attr_to_token[attr] in scenario_weather:
                    is_del = True
            elif attr_cls[attr] == 2:
                actor_type = scenario_weather.split('_')[3]
                if attr_to_token[attr] == "4":
                    if not actor_type in ["c", "t"]:
                        is_del = True
                elif attr_to_token[attr] == "2":
                    if not actor_type in ["m", "b"]:
                        is_del = True
                else:
                    if attr_to_token[attr] != actor_type:
                        is_del = True

            elif attr_cls[attr] == 3:
                obstacle_type = scenario_weather.split('_')[2]
                if not obstacle_type in attr_to_token[attr]:
                    is_del = True

            if is_del:
                del copy_roi_result[scenario_weather]

        new_roi_result.update(copy_roi_result)

    return new_roi_result


def read_metadata(data_type, metadata_root="./metadata"):

    if data_type in ["interactive", "obstacle"]:
        bahavior_path = os.path.join(
            metadata_root, "behavior", f"{data_type}.json")
        behavior = json.load(open(bahavior_path))
    else:
        behavior = None

    if data_type in ["interactive", "obstacle", "collision"]:
        gt_risk_path = os.path.join(
            metadata_root, "GT_risk", f"{data_type}.json")
        gt_risk = json.load(open(gt_risk_path))
    else:
        gt_risk = None

    critical_dict_path = os.path.join(
        metadata_root, "GT_critical_point", f"{data_type}.json")
    critical_dict = json.load(open(critical_dict_path))

    return behavior, gt_risk, critical_dict


def get_scenario_info(scenario, data_type, roi_result, behavior_dict, gt_risk_dict):

    basic = '_'.join(scenario.split('_')[:-3])
    variant = '_'.join(scenario.split('_')[-3:])

    if data_type in ["interactive", "obstacle"]:
        behavior = behavior_dict[basic][variant]
    else:
        behavior = (-1, 999)

    if data_type in ["interactive", "obstacle", "collision"]:
        risky_id = int(gt_risk_dict[basic+"_"+variant][0])
    else:
        risky_id = None

    roi = roi_result[basic+"_"+variant]

    return basic, variant, roi, behavior, risky_id


def mask_risk_object(rgb_path, seg_path, bbox, roi_dict, risky_id, mask_color=np.array([0, 0, 255], dtype="uint8")):
    """
        actor_id = channel[1] + 256*channel[2]
        class = channel[0]
    """

    rgb_frame = cv2.imread(rgb_path)
    seg_frame = cv2.imread(seg_path)
    img_shape = rgb_frame.shape      # (256, 640, 3)

    # height, width, channel = rgb_frame.shape

    for actor_id in bbox:

        x1, y1, x2, y2 = bbox[actor_id]

        if str(int(actor_id) % 65536) in roi_dict:
            src_id = str(int(actor_id) % 65536)
        elif str((int(actor_id) % 65536 + 65536)) in roi_dict:
            src_id = str((int(actor_id) % 65536 + 65536))
        else:
            continue

        # print(src_id, roi_dict[src_id], bbox[actor_id])
        is_risky = roi_dict[src_id]
        if is_risky:
            rgb_frame = cv2.rectangle(
                rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not (risky_id is None) and int(actor_id) % 65536 == int(risky_id) % 65536:

            for y in range(y1, y2):
                for x in range(x1, x2):

                    object_color = seg_frame[y, x][::-1]
                    instance_id = object_color[1] + 256*object_color[2]
                    # instance_class = object_color[0]

                    if instance_id % 65536 == int(risky_id) % 65536:
                        rgb_frame[y, x] = cv2.addWeighted(
                            rgb_frame[y, x], 0.2, mask_color, 0.8, 0).squeeze()
                        # rgb_frame[y, x] = color

    return rgb_frame[:, :, ::-1]


def make_video(gif_save_path, variant_path, roi, behavior, risky_id, FPS=10):

    rgb_folder = os.path.join(variant_path, f"rgb/front")
    seg_folder = os.path.join(variant_path, f"instance_segmentation/front")

    bbox_path = os.path.join(variant_path, f"bbox.json")
    bbox_info = json.load(open(bbox_path))

    frame_list = []
    start, end = behavior
    # N = len(sorted(os.listdir(rgb_folder)))

    for i, frame_id in enumerate(sorted(os.listdir(rgb_folder)), 1):

        frame = int(frame_id.split('.')[0])
        if not start <= frame <= end:
            continue

        rgb_path = os.path.join(rgb_folder, frame_id)
        seg_path = os.path.join(seg_folder, f"{frame:08d}.png")
        bbox = bbox_info[f"{frame:08d}"]

        if not f"{frame}" in roi:
            continue
        roi_dict = roi[f"{frame}"]

        roi_frame = mask_risk_object(rgb_path, seg_path, bbox, roi_dict, risky_id)
        frame_list.append(roi_frame)

    if len(frame_list) != 0:
        imageio.mimsave(gif_save_path, frame_list,
                        format='GIF', duration=1000/FPS)
