import os
import torch
import torch.utils.data as data
import numpy as np
import PIL.Image as Image
import random
import json

__all__ = [
    'GCNDataLayer',
]


class GCNDataLayer(data.Dataset):
    def __init__(self, data_root, behavior_root, state_root, scenario, time_steps, camera_transforms, num_box,
                 raw_img_size=(256, 640), img_resize=(256, 640), data_augmentation=True, phase="train"):

        self.data_root = data_root
        self.time_steps = time_steps
        self.camera_transforms = camera_transforms
        self.raw_img_size = raw_img_size
        self.img_resize = img_resize
        self.data_augmentation = data_augmentation
        self.phase = phase
        self.num_box = num_box
        self.scale_w = img_resize[1]/raw_img_size[1]
        self.scale_h = img_resize[0]/raw_img_size[0]

        self.behavior_dict = {}
        self.state_dict = {}
        self.cnt_labels = np.zeros(2, dtype=np.int32)
        self.inputs = []

        self.load_behavior(behavior_root)
        self.load_state(state_root, scenario)

        for (basic, variant, data_type) in scenario:

            variant_path = os.path.join(
                data_root, data_type, basic, "variant_scenario", variant)

            # get positive and negative behavior from behavior_dict
            frames, labels = self.get_behavior(
                basic, variant, variant_path, data_type)

            for frame_no, label in list(zip(frames, labels))[::5]:
                # for frame_no, label in zip(frames_no, labels):
                self.inputs.append([variant_path, frame_no-self.time_steps+1,
                                    frame_no+1, np.array(label), data_type])
                self.cnt_labels[label] += 1

        """
            train   Label 'go'   (negative):   38387
            train   Label 'stop' (positive):   27032
            test    Label 'go'   (negative):   11072
            test    Label 'stop' (positive):    4148
        """
        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")

    def load_behavior(self, behavior_root):

        for _type in ["interactive", "obstacle"]:
            behavior_path = os.path.join(
                behavior_root, f"{_type}.json")
            behavior = json.load(open(behavior_path))
            self.behavior_dict[_type] = behavior

    def load_state(self, state_root, scenario):
        
        for (basic, variant, data_type) in scenario:
            state_path = os.path.join(state_root, data_type, basic+"_"+variant+".json")
            state = json.load(open(state_path))   
            self.state_dict[basic+"_"+variant] = state

    def get_behavior(self, basic, variant, variant_path, data_type, start_frame=1):

        N = len(os.listdir(variant_path+'/rgb/front'))
        first_frame_id = start_frame + self.time_steps - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(first_frame_id, last_frame_id+1))
        labels = np.zeros(N, dtype=np.int32)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = min(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start-first_frame_id: end-first_frame_id] = 1

        return frames, labels

    def normalize_box(self, trackers):
        """
            return normalized_trackers TxNx4 ndarray:
            [BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_BOTRIGHT_X, BBOX_BOTRIGHT_Y]
        """

        normalized_trackers = trackers.copy()
        normalized_trackers[:, :,
                            0] = normalized_trackers[:, :, 0] * self.scale_w
        normalized_trackers[:, :,
                            2] = normalized_trackers[:, :, 2] * self.scale_w
        normalized_trackers[:, :,
                            1] = normalized_trackers[:, :, 1] * self.scale_h
        normalized_trackers[:, :,
                            3] = normalized_trackers[:, :, 3] * self.scale_h

        return normalized_trackers

    def process_tracking(self, variant_path, start, end):
        """
            tracking_results Kx10 ndarray:
            [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
            e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
        """

        INTENTIONS = {'r': 1, 'sl': 2, 'f': 3, 'gi': 4, 'l': 5, 'gr': 6, 'u': 7, 'sr': 8,'er': 9}

        def parse_scenario_id(variant_path):
            
            variant_path_token = variant_path.split('/')
            
            basic = variant_path_token[-3]
            variant = variant_path_token[-1]

            basic_token = basic.split('_')

            if "obstacle" in variant_path:
                return basic_token[3], basic, variant
            else:
                return basic_token[5], basic, variant

        ego_intention, basic, variant = parse_scenario_id(variant_path)

        tracking_results = np.load(
            os.path.join(variant_path, 'tracking.npy'))
        assert len(tracking_results) > 0, f"{variant_path} No tracklet"

        height, width = self.raw_img_size

        t_array = tracking_results[:, 0]
        tracking_index = tracking_results[np.where(t_array == end-1)[0], 1]

        trackers = np.zeros([self.time_steps, self.num_box, 4]).astype(np.float32) # TxNx4
        intentions = np.zeros(10).astype(np.float32)   # 10
        states = np.zeros([self.time_steps, self.num_box+1, 2]).astype(np.float32)   # Tx(N+1)x2

        intentions[INTENTIONS[ego_intention]] = 1

        for t in range(start, end):
            current_tracking = tracking_results[np.where(t_array == t)[0]]

            for i, object_id in enumerate(tracking_index):
                current_actor_id_idx = np.where(
                    current_tracking[:, 1] == object_id)[0]

                if len(current_actor_id_idx) != 0:
                    # x1, y1, x2, y2
                    bbox = current_tracking[current_actor_id_idx, 2:6]
                    bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
                    bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, width)
                    bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
                    bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, height)
                    trackers[t-start, i, :] = bbox

                    try:
                        states[t-start, i+1] = self.state_dict[basic+"_"+variant][str(t)][str(object_id)]
                    except:
                        states[t-start, i+1] = 0


        trackers = self.normalize_box(trackers)

        return trackers, tracking_index, intentions, states

    def __getitem__(self, index):

        variant_path, start, end, label, data_type = self.inputs[index]

        camera_inputs = []

        for idx in range(start, end):
            camera_name = f"{int(idx):08d}.jpg"
            camera_path = os.path.join(variant_path, "rgb/front", camera_name)
            img = self.camera_transforms(
                Image.open(camera_path).convert('RGB'))
            camera_inputs.append(img)

        camera_inputs = torch.stack(camera_inputs)

        trackers, tracking_id, intention_inputs, state_inputs = self.process_tracking(
            variant_path, start, end)

        num_object = len(tracking_id)
        obstacle_idx = []
        obstacle_idx.append(random.randint(0, num_object)+1)

        # add data augmentation
        mask = torch.ones(
            (self.time_steps, 3, self.img_resize[0], self.img_resize[1]))

        if self.data_augmentation and data_type != "obstacle" and label == 0:

            for t in range(self.time_steps):
                for i in obstacle_idx:

                    y1, x1, y2, x2 = trackers[t, i, :]
                    mask[t, :, int(y1):int(y2), int(x1):int(x2)] = 0

        trackers = torch.from_numpy(trackers)
        intention_inputs = torch.from_numpy(intention_inputs)
        state_inputs = torch.from_numpy(state_inputs)
        label = torch.from_numpy(label.astype(np.int64))

        return camera_inputs, trackers, mask, label, intention_inputs, state_inputs

    def __len__(self):
        return len(self.inputs)
