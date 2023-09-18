import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import cv2
import math
import pdb
import re
import tqdm
import pandas as pd

# colormap
colors = [(0, 0, 0), (0.87, 0.87, 0.87),
          (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)


class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.
    The building class is merged into the background class
    0:background 1:street 2:sidewalk, 3:building 4: vegetation ---> 0:background 1:street 2:sidewalk, 3: vegetation
    """

    def __init__(self, train, name, weather_name, num_time, vehicle, f_frame, now_frame, pred_len):

        # useless
        self.dim_clip = 180  # dim_clip*2 is the dimension of scene (pixel)
        self.is_train = train
        self.angle_presents = []
        self.scene = []
        self.scene_crop = []

        self.video_track = []     # '0001'
        self.number_vec = []      # '4' ?
        self.index = []           # '50' ?
        self.pasts = []           # [len_past, 2]
        self.presents = []        # position in complete scene
        self.futures = []         # [len_future, 2]
        num_total = 20 + 30  # 50

        num_total = 50  # 50

        DATA_DIR = 'data_carla_risk_all/'

        #print("now:", train, self.is_train)

        # Preload data
        for folder in os.listdir(DATA_DIR):
            #print("folder:", folder)
            if train == True and not re.search(r'train', folder):
                continue
            elif train == False and not re.search(r'test', folder):
                continue
            #print(f"folder: {folder}")
            if train == True:
                f = 'train'
            elif train == False:
                f = 'test'
            #traj_df = pd.read_csv(DATA_DIR + f + '/non-interactive/' + name + '/variant_scenario/' + weather_name + '/trajectory/' + name + '_all.csv')
            #traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
            #vehicle_list = []
            # for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
                #print("remain_df:", remain_df)
            #    if obj_type == 'AGENT':
            #        vehicle_list.append(remain_df)
            #    elif obj_type == 'vehicle':
            #        vehicle_list.append(remain_df)

            #print(train, name, weather_name, num_time, vehicle, f_frame)

            #points = np.vstack((vehicle_list[num_vec]['X'], vehicle_list[num_vec]['Y'])).T
            points = np.vstack((vehicle['X'], vehicle['Y'])).T

            temp_past = points[num_time:num_time + 20].copy()
            origin = temp_past[-1]
            if points.shape[0] <= 80:
                print("num_time:", num_time, points.shape)

            temp_future = points[num_time + 20:num_time + 20 + pred_len].copy()
            '''
            if (num_time + 50) > points.shape[0]:
                # if points[num_time + 50][0] != None:
                temp_future = points[num_time + 20:num_time + 50].copy()
            else:
                temp_future = np.zeros((30, 2))
            '''
            id = vehicle['TRACK_ID'].values[0]
            object_type = vehicle['OBJECT_TYPE'].values[0]
            #print("past:", temp_past, len(temp_past[0]))

            '''
            # filter out noise for non-moving vehicles
            if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                temp_past = np.zeros((20, 2))
            else:
                temp_past = temp_past - origin
            if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                temp_future = np.zeros((30, 2))
            else:
                temp_future = temp_future - origin
            '''
            #print("temp_future:", temp_future, "origin:", origin)
            temp_past = temp_past - origin
            temp_future = temp_future - origin
            #print("temp_future:", temp_future)

            ################
            '''
            unit_y_axis = torch.Tensor([0, -1])
            vector = temp_past[-5]
            if int(vector[0]) == 0:
                angle = 0
            elif vector[0] > 0.0:
                angle = np.rad2deg(self.angle_vectors(vector, unit_y_axis))
            else:
                angle = -np.rad2deg(self.angle_vectors(vector, unit_y_axis))
            matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
            matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)
            past_rot = cv2.transform(temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
            future_rot = cv2.transform(temp_future.reshape(-1, 1, 2), matRot_track).squeeze()

            self.pasts.append(past_rot)
            self.futures.append(future_rot)
            '''
            '''
            if num_time == 0:
                print(num_vec, "past:", temp_past, "future:", temp_future)
            if num_time == 10:
                print(num_vec, "past:", temp_past, "future:", temp_future)
            if num_time == 20:
                print(num_vec, "past:", temp_past, "future:", temp_future)
            if num_time == 30:
                print(num_vec, "past:", temp_past, "future:", temp_future)
            '''
            #print(c, count, temp_past)

            self.pasts.append(temp_past)
            self.futures.append(temp_future)
            self.presents.append(origin)
            self.index.append(num_time + 19)
        #print("self:", self.pasts, self.futures, self.presents)
        self.index = np.array(self.index)
        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.presents = torch.FloatTensor(self.presents)
        self.file_name = name
        self.weather_type = weather_name
        self.train_val = 'train' if train else 'val'
        self.track_id = id
        self.type = object_type
        self.f_frame = f_frame
        self.now_frame = now_frame
        #self.num_vehicle = len

    def save_dataset(self, folder_save):
        for num_vehicle in range(len(self.pasts)):
            for i in range(self.pasts[num_vehicle].shape[0]):
                #video = self.video_track[i]
                #vehicle = self.vehicles[i]
                #number = self.number_vec[i]
                past = self.pasts[num_vehicle][i]
                future = self.futures[num_vehicle][i]
                #scene_track = self.scene_crop[i]

                saving_list = ['only_tracks']
                for sav in saving_list:
                    folder_save_type = folder_save + sav + '/'
                    # if not os.path.exists(folder_save_type):
                    #    os.makedirs(folder_save_type)
                    video_path = folder_save_type + '/'
                    if not os.path.exists(video_path + 'track'):
                        os.makedirs(video_path + 'track')
                    vehicle_path = video_path + '/track/'
                    if sav == 'only_tracks':
                        self.draw_track(
                            past, future, index_tracklet=self.index[num_vehicle][i], path=vehicle_path)
                # if sav == 'only_scenes':
                #    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                # if sav == 'tracks_on_scene':
                #    self.draw_track_in_scene(past, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self, past, future, index_tracklet, path):
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1],
                     c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        # print semantic map
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_track_in_scene(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        plt.plot(story[:, 0] * 2 + self.dim_clip, story[:, 1] *
                 2 + self.dim_clip, c='blue', marker='o', markersize=1)
        plt.plot(future[:, 0] * 2 + self.dim_clip, future[:, 1]
                 * 2 + self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    @staticmethod
    def get_desire_track_files(train):
        """ Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained from the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        """
        if train:
            desire_ids = [5, 9, 11, 13, 14, 17, 27,
                          28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files, desire_ids

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_vectors(self, v1, v2):
        """ Returns angle between two vectors.  """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(angle):
            return 0.0
        else:
            return angle

    def __len__(self):
        #print("111:", self.past.shape[0])
        #print("222:", self.past[0].shape[0])
        return self.pasts.shape[0]
        # return

    def __getitem__(self, idx):
        # print("111:", self.pasts.shape[0]) #2
        # print("222:", self.pasts[0].shape[0]) #45
        #print("self.pasts[0]:", self.pasts[0])
        #print("num:", self.num_vehicle)
        #print("l:", len(self.pasts))
        # for i in range(len(self.pasts)):
        #    return self.pasts[i][idx], self.futures[i][idx], self.presents[i][idx]
        # return 0, 0, 0
        return self.pasts[idx], self.futures[idx], self.presents[idx]
