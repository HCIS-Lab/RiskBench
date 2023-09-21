#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

import glob
import os
import sys

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')
    sys.path.append('rss/')  # rss

except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from six.moves import cPickle as pickle # for performance
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc
import torch
from util.read_input import *
from util.get_and_control_trafficlight import *
from util.random_actors import spawn_actor_nearby
import carla
from util.controller import VehiclePIDController
import argparse
import logging
import math
import random
import cv2
import json
import re
import matplotlib
matplotlib.use('Agg')
from PIL import Image, ImageDraw
import pandas as pd
import pygame
import numpy as np
from util.KeyboardControl import KeyboardControl
from util.hud import HUD
from util.sensors import CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor, RadarSensor, CameraManager
from util.data_collection import Data_Collection
from torchvision.ops.boxes import masks_to_boxes
from torchvision import transforms
        
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def write_json(filename, index, seed):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        y = {str(index): seed}
        file_data.update(y)
        file.seek(0)
        json.dump(file_data, file, indent=4)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, client_bp, hud, args, seeds):
        self.world = carla_world
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.abandon_scenario = False
        self.finish = False
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        self.args = args
        
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = client_bp
        self._gamma = args.gamma
        self.ego_data = {}
        self.save_mode = not args.no_save
        self.inference_mode = args.inference
        
        
        self.restart(self.args, seeds)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self, args, seeds):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.

        seed_1 = seeds[1]
        random.seed(seed_1)
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)

        if blueprint.has_attribute('color'):

            seed_2 = seeds[2]
            random.seed(seed_2)

            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):

            seed_3 = seeds[2]
            random.seed(seed_3)

            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # coor=carla.Location(location[0],location[1],location[2]+2.0)
            # self.player.set_location(coor)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player, self.ego_data)
        self.imu_sensor = IMUSensor(self.player, self.ego_data)
        self.camera_manager = CameraManager(
            self.player, self.hud, self._gamma, self.save_mode, self.inference_mode)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.background = True
        self.camera_manager.save_mode = self.save_mode

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def record_speed_control_transform(self, frame):
        v = self.player.get_velocity()
        c = self.player.get_control()
        t = self.player.get_transform()
        if frame not in self.ego_data:
            self.ego_data[frame] = {}
        self.ego_data[frame]['speed'] = {'constant': math.sqrt(v.x**2 + v.y**2 + v.z**2),
                                         'x': v.x, 'y': v.y, 'z': v.z}
        self.ego_data[frame]['control'] = {'throttle': c.throttle, 'steer': c.steer,
                                           'brake': c.brake, 'hand_brake': c.hand_brake,
                                           'manual_gear_shift': c.manual_gear_shift,
                                           'gear': c.gear}
        self.ego_data[frame]['transform'] = {'x': t.location.x, 'y': t.location.y, 'z': t.location.z,
                                             'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll}

    def save_ego_data(self, path):
        self.imu_sensor.toggle_recording_IMU()
        self.gnss_sensor.toggle_recording_Gnss()
        with open(os.path.join(path, 'ego_data.json'), 'w') as f:
            json.dump(self.ego_data, f, indent=4)
        self.ego_data = {}

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        
        if self.inference_mode:
           # inference_transfuser model
            sensors = [
                    self.camera_manager.sensor_top,
                    self.camera_manager.sensor_rgb_front,
                    self.camera_manager.sensor_ss_front,
                    self.collision_sensor.sensor,
                    self.lane_invasion_sensor.sensor,
                    self.gnss_sensor.sensor,
                    self.imu_sensor.sensor
                ]
        else:
            if self.save_mode:
                sensors = [
                    self.camera_manager.sensor_top,
                    self.camera_manager.sensor_ss_top,
                    self.camera_manager.sensor_rgb_front,
                    self.camera_manager.sensor_ss_front,
                    self.camera_manager.sensor_depth_front,
                    self.camera_manager.sensor_lidar,
                    self.collision_sensor.sensor,
                    self.lane_invasion_sensor.sensor,
                    self.gnss_sensor.sensor,
                    self.imu_sensor.sensor
                ]
            else:
                sensors = [
                    self.camera_manager.sensor_top,
                    self.collision_sensor.sensor,
                    self.lane_invasion_sensor.sensor,
                    self.gnss_sensor.sensor,
                    self.imu_sensor.sensor
                ]

        for i, sensor in enumerate(sensors):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        if self.player is not None:
            self.player.destroy()

def set_bp(blueprint):
    blueprint = random.choice(blueprint)
    blueprint.set_attribute('role_name', 'tp')
    if blueprint.has_attribute('color'):
        color = random.choice(
            blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(
            blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')

    return blueprint

def check_actor_list(world):
    # inf id min and max 
    def check_row( actors, filter_str, min_id, max_id):
        filter_actors = actors.filter(filter_str)
        for actor in filter_actors:
            if actor.id < min_id:
                min_id = actor.id
            if actor.id > max_id:
                max_id = actor.id
        return min_id, max_id

    filter_ = ['walker.*', 'vehicle.*', 'static.prop.streetbarrier*',
               'static.prop.trafficcone*', 'static.prop.trafficwarning*']
    id_ = [4, 10, 20, 20, 20]
    actors = world.world.get_actors()
    min_id = int(1e7)
    max_id = int(0)
    for filter_str, class_id in zip(filter_, id_):
        min_id, max_id = check_row( actors, filter_str, min_id, max_id)        
    return min_id, max_id

def generate_obstacle(world, bp, src_path, ego_transform, obstacle_GT_location):
    """
        stored_path : data_collection/{scenario_type}/{scenario_id}/{weather}+'_'+{random_actors}+'_'    
    """


    obstacle_list = json.load(open(src_path))
    obstacle_info = {}

    min_dis = float('Inf')
    nearest_obstacle = -1

    obstacle_static_id_list = []
    ill_parking_id_list = []

    # obstacle_GT_location
    for obstacle_attr in obstacle_list:

        """
            obstacle_attr = {"obstacle_type": actor.type_id,
                            "basic_id": actor.id,
                            "location": new_trans.location.__dict__,
                            "rotation": new_trans.rotation.__dict__}
        """

        obstacle_name = obstacle_attr["obstacle_type"]
    
        location = obstacle_attr["location"]
        rotation = obstacle_attr["rotation"]

        x = float(location["x"])
        y = float(location["y"])
        z = float(location["z"])

        # only spawn Ground truth obstacle
        min_distance  = 1000
        for loc in obstacle_GT_location:
            gt_x = float(loc[0])
            gt_y = float(loc[1])
            distance = math.sqrt((x-gt_x)**2 + (y-gt_y)**2)
            if distance < min_distance:
                min_distance = distance
        if min_distance > 1.5:
            continue

        pitch = float(rotation["pitch"])
        yaw = float(rotation["yaw"])
        roll = float(rotation["roll"])

        obstacle_loc = carla.Location(x, y, z)
        obstacle_rot = carla.Rotation(pitch, yaw, roll)
        obstacle_trans = carla.Transform(obstacle_loc, obstacle_rot)

        obstacle_actor = world.spawn_actor(
            bp.filter(obstacle_name)[0], obstacle_trans)

        dis = ego_transform.location.distance(obstacle_loc)
        if dis < min_dis:
            nearest_obstacle = obstacle_actor.id
            min_dis = dis

        obstacle_info[obstacle_actor.id] = obstacle_attr

        if "static" in obstacle_name:
            obstacle_static_id_list.append(obstacle_actor.id)
        else:

            ill_parking_id_list.append(obstacle_actor.id)

    return nearest_obstacle, obstacle_info, obstacle_static_id_list, ill_parking_id_list

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        from collections import deque
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
        
class Inference():
    def __init__(self, args, variant_path, weather) -> None:
        from models.LBC.map_model import MapModel

        self.scenario_type = args.scenario_type
        self.scenario_id = args.scenario_id
        self.weather = weather
        self.actor = args.random_actors
        self.seed = args.random_seed
        self.map = args.map
        self.args = args
        # init model 
        self.model = None
        self.rgb_front = None
        self.ss_front = None
        self.compass = None
        self.gt_interactor = -1

        self.gt_obstacle_id_list = []
        self.gt_obstacle_id_nearest = -1

        self.birdview_producer = BirdViewProducer(
                self.args.map, 
                PixelDimensions(width=256, height=256), 
                pixels_per_meter=5)

        # load LBC model 
        if self.scenario_type =="interactive":
            self.net = MapModel.load_from_checkpoint("./models/weights/LBC/interactive.ckpt")
        elif self.scenario_type =="obstacle":
            self.net = MapModel.load_from_checkpoint("./models/weights/LBC/obstacle.ckpt")

        self.net.cuda()
        self.net.eval()
        self.variant_path = variant_path

        target = self.load_dict(os.path.join(variant_path, "target_point.pkl"))["target_point"]
        
        x=float(target[0])
        y=float(target[1])
        self.v = np.array([x, y])
        self.topdown_debug_list = []
        self.ego_speed_controller = PIDController(K_P=1, K_I=0, K_D=0.0)
        self.counter = 0 # use counter to deal with agent stuck porblem
        self.min_distance = 1000 # caculate the min distance with gt interactor  
        self.avg_distance = 0
        self.counter_avg_distance = 0

        self.obestacle_id_list = []
        self.ill_parking_id_list = []

        self.mode = args.mode 

        self.rgb_list_bc_method = []
        self.bbox_list_bc_method = []

        self.bbox_list_DSA = []
        self.bbox_id_list_DSA = []

        self.collision_flag = False

        self.Mean_filter_list = []
        self.front_rgb_out = cv2.VideoWriter(f'./{args.scenario_id}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,  (640, 256)) 


        if self.mode == "Kalman_Filter":
            from models.KalmanFilter import kf_inference
            
            self.kf_inference = kf_inference
            self.df_list = []
            self.df_start_frame = 0
        elif self.mode == "MANTRA":
            self.df_list = []
            self.df_start_frame = 0
            from models.mantra.mantra import mantra_inference
            self.mantra_inference = mantra_inference
        elif self.mode == "Social-GAN":
            self.df_list = []
            self.df_start_frame = 0
            from models.sgan.social_gan import socal_gan_inference
            from attrdict import AttrDict
            from models.sgan.models import TrajectoryGenerator

            self.socal_gan_inference = socal_gan_inference

            checkpoint = torch.load("./models/weights/sgan/gan_test_with_model_all.pt")
            args_sg = AttrDict(checkpoint['args'])
            self.generator = TrajectoryGenerator(
                obs_len=args_sg.obs_len,
                pred_len=30 ,#new_args.pred_len,
                embedding_dim=args_sg.embedding_dim,
                encoder_h_dim=args_sg.encoder_h_dim_g,
                decoder_h_dim=args_sg.decoder_h_dim_g,
                mlp_dim=args_sg.mlp_dim,
                num_layers=args_sg.num_layers,
                noise_dim=args_sg.noise_dim,
                noise_type=args_sg.noise_type,
                noise_mix_type=args_sg.noise_mix_type,
                pooling_type=args_sg.pooling_type,
                pool_every_timestep=args_sg.pool_every_timestep,
                dropout=args_sg.dropout,
                bottleneck_dim=args_sg.bottleneck_dim,
                neighborhood_size=args_sg.neighborhood_size,
                grid_size=args_sg.grid_size,
                batch_norm=args_sg.batch_norm)
            self.generator.load_state_dict(checkpoint['g_state'])
            self.generator.cuda()
            self.generator.train()
            self._args = AttrDict(checkpoint['args'])
            self._args.dataset_name = "interactive" 
            self._args.skip = 1
            self._args.pred_len = 30 

            # def socal_gan_inference(vehicle_list, specific_frame, variant_ego_id, pedestrian_id_list, vehicle_id_list , obstacle_id_list, _args, generator):
        elif self.mode == "QCNet":
            self.df_list = []
            self.df_start_frame = 0

            from models.QCNet.QCNet import QCNet_inference
            self.QCNet_inference = QCNet_inference
        elif self.mode == "BP" or self.mode ==  "BCP" or self.mode == "BCP_smoothing" or self.mode == "BP_smoothing":
            from models.two_stage.inference import testing
            from models.two_stage.models import GCN as Model
            
            self.BC_model = Model()
            checkpoint="./models/weights/two_stage/weight.pth"
            state_dict = torch.load(checkpoint)
            state_dict_copy = {}
            for key in state_dict.keys():
                state_dict_copy[key[7:]] = state_dict[key]
            self.BC_model.load_state_dict(state_dict_copy)
            self.BC_model = self.BC_model.to('cuda')
            self.BC_model.train(False)

            self.BC_testing = testing
        elif self.mode == "DSA" or self.mode =="DSA_smoothing": 
            # ckpt = {'DSA_no_intention':'8_27_3_46','DSA_intention':'9_1_2_2','RRL_no_intention':'8_29_21_34','RRL_intention':'8_29_18_21'}
            # model_names = [['DSA_no_intention','DSA_intention'],['RRL_no_intention','RRL_intention']]

            from models.dsa.DSA_RRL import Baseline_SA
            from models.dsa.backbone import Riskbench_backbone

            intention  = False
            supervised = False
            model_path = f"./models/weights/dsa/8_27_3_46/best_model.pt"
            object_num = 20
            n_frame = 40
            device = torch.device('cuda')

            backbone = Riskbench_backbone(8,object_num,intention=intention)
            self.dsa_model = Baseline_SA(backbone,n_frame,object_num,intention=intention,supervised=supervised, state= False)
            # 
            # self.dsa_model = Baseline_SA(backbone,n_frame,object_num,intention=intention,supervised=supervised,state=state)

            self.dsa_model.load_state_dict(torch.load(model_path,map_location=device))
            self.dsa_model.cuda()
            self.dsa_model.eval()
        elif self.mode == "RRL"  or self.mode == "RRL_smoothing" :
            from models.dsa.DSA_RRL import Baseline_SA
            from models.dsa.backbone import Riskbench_backbone

            intention  = False
            supervised = True
            model_path = f"./models/weights/dsa/9_12_0_56/best_model.pt"
            object_num = 20
            n_frame = 40
            device = torch.device('cuda')
            backbone = Riskbench_backbone(8,object_num,intention=intention)
            self.dsa_model = Baseline_SA(backbone,n_frame,object_num,intention=intention,supervised=supervised, state= False)
            self.dsa_model.load_state_dict(torch.load(model_path,map_location=device))
            self.dsa_model.cuda()
            self.dsa_model.eval()


        # read static bbox 
        self.static_vehicle_bbox_list = []
        with open(f"./util/static_bbox/static_{self.args.map}.json") as f:
            data = json.load(f)
            len_of_bbox = data["num_of_id"]
            for index in range(len_of_bbox):
                pos_0 = data[str(index)]["cord_bounding_box"]["cord_0"]
                pos_1 = data[str(index)]["cord_bounding_box"]["cord_4"]
                pos_2 = data[str(index)]["cord_bounding_box"]["cord_6"]
                pos_3 = data[str(index)]["cord_bounding_box"]["cord_2"]
                self.static_vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])

    def set_obstacle_ids(self, obstacle_id_list, ill_parking_id_list):
        self.obestacle_id_list = obstacle_id_list
        self.ill_parking_id_list = ill_parking_id_list

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di


    def set_end_position(self, x, y):

        self.end_position_x = x
        self.end_position_y = y
        
    def set_gt_obstacle_ids(self, obstacle_gt_location, world, obstacle_nearest_id):

        # get obstacle gt ids
        obstacle = world.world.get_actors().filter("*static.prop*")
        vehicles = world.world.get_actors().filter("*vehicle*")

        gt_obstacle_id_list = []
        for loc in obstacle_gt_location:
            x = float(loc[0])
            y = float(loc[1])

            min_distance = 100
            gt_id = -1

            for actor in obstacle:
                _id = actor.id 
                actor_loc = actor.get_location()
                x_ = actor_loc.x
                y_ = actor_loc.y
                distance = math.sqrt((x-x_)**2 + (y-y_)**2)

                if distance < min_distance:
                    min_distance = distance
                    gt_id = _id

            for actor in vehicles:
                _id = actor.id 
                actor_loc = actor.get_location()
                x_ = actor_loc.x
                y_ = actor_loc.y
                distance = math.sqrt((x-x_)**2 + (y-y_)**2)

                if distance < min_distance:
                    min_distance = distance
                    gt_id = _id
                
            if gt_id ==-1:
                print("error")

            gt_obstacle_id_list.append(gt_id)

        self.gt_obstacle_id_list = gt_obstacle_id_list
        self.gt_obstacle_id_nearest = obstacle_nearest_id

        return gt_obstacle_id_list




    def collect_actor_data(self, world, frame):

        all_id_list = []

        if self.mode == "Kalman_Filter" or self.mode == "MANTRA" or self.mode == "Social-GAN" or self.mode == "QCNet":
            if self.df_start_frame == 0:
                self.df_start_frame = frame

        vehicles_id_list = []
        bike_blueprint = ["vehicle.bh.crossbike","vehicle.diamondback.century","vehicle.gazelle.omafiets"]
        motor_blueprint = ["vehicle.harley-davidson.low_rider","vehicle.kawasaki.ninja","vehicle.yamaha.yzf","vehicle.vespa.zx125"]
        
        def get_xyz(method, rotation=False):

            if rotation:
                roll = method.roll
                pitch = method.pitch
                yaw = method.yaw
                return {"pitch": pitch, "yaw": yaw, "roll": roll}

            else:
                x = method.x
                y = method.y
                z = method.z

                # return x, y, z
                return {"x": x, "y": y, "z": z}

        ego_loc = world.player.get_location()
        data = {}

        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

        
            _id = actor.id
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            transform = actor.get_transform().rotation
            rotation = get_xyz(transform, True)


            cord_bounding_box = {}
            bbox = actor.bounding_box
            if actor.type_id in motor_blueprint:
                bbox.extent.x = 1.177870
                bbox.extent.y = 0.381839
                bbox.extent.z = 0.75
                bbox.location = carla.Location(0, 0, bbox.extent.z)
            elif actor.type_id in bike_blueprint:
                bbox.extent.x = 0.821422
                bbox.extent.y = 0.186258
                bbox.extent.z = 0.9
                bbox.location = carla.Location(0, 0, bbox.extent.z)
                
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)


            if distance < 35:
                if _id != self.ego_id:
                    all_id_list.append(_id)



            vehicles_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            
            angular_velocity = get_xyz(actor.get_angular_velocity())

            v = actor.get_velocity()
            velocity = get_xyz(v)

            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            vehicle_control = actor.get_control()
            control = {
                "throttle": vehicle_control.throttle,
                "steer": vehicle_control.steer,
                "brake": vehicle_control.brake,
                "hand_brake": vehicle_control.hand_brake,
                "reverse": vehicle_control.reverse,
                "manual_gear_shift": vehicle_control.manual_gear_shift,
                "gear": vehicle_control.gear
            }

            data[_id] = {}
            data[_id]["location"] = location
            data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["speed"] = speed
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control
            if _id == self.ego_id:
                data[_id]["compass"] = self.compass
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "vehicle"


            if self.mode == "Kalman_Filter" or self.mode == "MANTRA" or self.mode == "Social-GAN" or self.mode == "QCNet":
                if _id == self.ego_id:
                    self.df_list.append([frame, _id, 'EGO', str(actor_loc.x), str(actor_loc.y), v.x , v.y, transform.yaw])
                elif _id == self.gt_interactor:
                    self.df_list.append([frame, _id, 'ACTOR', str(actor_loc.x), str(actor_loc.y), v.x , v.y, transform.yaw])
                else:
                    self.df_list.append([frame, _id, 'vehicle', str(actor_loc.x), str(actor_loc.y), v.x , v.y, transform.yaw])

        pedestrian_id_list = []

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            transform = actor.get_transform().rotation
            rotation = get_xyz(transform, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            if distance < 35:
                
                all_id_list.append(_id)

            
            pedestrian_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            v = actor.get_velocity()
            velocity = get_xyz(v)
            angular_velocity = get_xyz(actor.get_angular_velocity())

            walker_control = actor.get_control()
            control = {"direction": get_xyz(walker_control.direction),
                       "speed": walker_control.speed, "jump": walker_control.jump}

            data[_id] = {}
            data[_id]["location"] = location
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control

            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = 'pedestrian'

            if self.mode == "Kalman_Filter" or self.mode == "MANTRA" or self.mode == "Social-GAN" or self.mode == "QCNet":
                if _id == self.gt_interactor:
                    self.df_list.append([frame, _id, 'ACTOR', str(actor_loc.x), str(actor_loc.y), v.x , v.y, control["direction"]["y"]])
                else:
                    self.df_list.append([frame, _id, 'pedestrian', str(actor_loc.x), str(actor_loc.y), v.x , v.y, control["direction"]["y"] ])

        obstacle_id_list = []

        obstacle = world.world.get_actors().filter("*static.prop*")

        data["obstacle"]= {}
        for actor in obstacle:

            _id = actor.id
            type_id = actor.type_id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            transform = actor.get_transform().rotation
            rotation = get_xyz(transform, True)
            distance = ego_loc.distance(actor_loc)


            if distance < 35:
                all_id_list.append(_id)

            bbox = actor.bounding_box

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            #if distance < 50:
            obstacle_id_list.append(_id)

            data["obstacle"][_id] = {}
            data["obstacle"][_id]["distance"] = distance
            data["obstacle"][_id]["type_id"] = type_id
            # 
            data["obstacle"][_id]["type"] = "obstacle"
            data["obstacle"][_id]["cord_bounding_box"] = cord_bounding_box

            if self.mode == "Kalman_Filter" or self.mode == "MANTRA" or self.mode == "Social-GAN" or self.mode == "QCNet":
                self.df_list.append([frame, _id, type_id, str(actor_loc.x), str(actor_loc.y), 0 , 0, transform.yaw])


        # data["traffic_light_ids"] = traffic_id_list

        data["obstacle_ids"] = obstacle_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list
        data["all_ids"] = all_id_list

        return data
    
    def set_scenario_type(self, sceanrio):
        self.scenario_type = sceanrio

    def set_ego_id(self, world):
        self.ego_id = world.player.id

    def set_gt_interactor(self, id):
        self.gt_interactor = id


    def get_ids(self, mask, area_threshold=400):
        """
            Args:
                mask: instance image
        """
        obstacle_boxes,obstacle_ids = [], []
        h,w = mask.shape[1:]
        # print("h, w", h, w)
        mask_2 = torch.zeros((2,h,w), device="cuda:0")
        mask_2[0] = mask[0]
        mask_2[1] = mask[1]+mask[2]*256

        if self.scenario_type =="obstacle":
            condition = mask[0]== 21 # Obstacle
            obstacle_ids = torch.unique(mask_2[1,condition])
            masks = mask_2[1] == obstacle_ids[:, None, None]
            masks = masks*condition
            area_condition = masks.long().sum((1,2))>=area_threshold
            masks = masks[area_condition]
            
            obstacle_ids = obstacle_ids[area_condition].type(torch.int).cpu().numpy()
            obstacle_boxes = masks_to_boxes(masks).type(torch.int16).cpu().numpy()
            assert len(obstacle_ids) == len(obstacle_boxes)

        condition = mask_2[0]== 14 # Car
        condition += mask_2[0]== 15 # Truck
        condition += mask_2[0]== 16 # Bus
        condition += mask_2[0]== 12 # Pedestrian
        condition += mask[0]== 18 # Motorcycle
        condition += mask[0]== 19 # Bicycle
        obj_ids = torch.unique(mask_2[1,condition])
        masks = mask_2[1] == obj_ids[:, None, None]
        masks = masks*condition
        area_condition = masks.long().sum((1,2))>=area_threshold
        masks = masks[area_condition]
        
        obj_ids = obj_ids[area_condition].type(torch.int).cpu().numpy()
        boxes = masks_to_boxes(masks).type(torch.int16).cpu().numpy()
        assert len(obj_ids) == len(boxes)

        return boxes, obj_ids, obstacle_boxes, obstacle_ids
    
    
    def set_autopilot(self, world):
        ## automomatic 
        from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
        from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
        self.agent = BehaviorAgent(world.player, behavior="cautious")
        destination = carla.Location( float(self.v[0]), float(self.v[1]), 0.6)
        self.agent.set_destination(destination)

    def run_inference(self, frame, world, pre_get_data = False):
        
        
        while True:
            if world.camera_manager.ss_front.frame == frame:
                self.ss_front = world.camera_manager.ss_front
                break

        while True:
            if world.camera_manager.rgb_front.frame == frame:
                self.rgb_front = world.camera_manager.rgb_front
                break

        # ins_front_array = torch.from_numpy(ins_front_array.copy())[:,:,:3].type(torch.int).permute((2,0,1))
        #produce_bbx(ins_front_array, actor_list_and_position, frame)


        instance = np.frombuffer(self.ss_front.raw_data, dtype=np.dtype("uint8"))
        instance = np.reshape(instance, (self.ss_front.height, self.ss_front.width, 4))
        instance = instance[:, :, :3]
        instance_torch = torch.flip(torch.from_numpy(instance.copy()).type(torch.int).permute(2,0,1),[0])
        # print(instance_torch.shape) # torch.Size([3, 256, 640])

        instance_torch = instance_torch.to("cuda:0")
        boxes, obj_ids, obstacle_boxes, obstacle_ids = self.get_ids(instance_torch)



        rgb = np.frombuffer(self.rgb_front.raw_data, dtype=np.dtype("uint8"))
        rgb = np.reshape(rgb, (self.rgb_front.height, self.rgb_front.width, 4))
        rgb = rgb[:, :, :3]

        
        self.front_rgb_out.write(rgb)

        camera_transforms = transforms.Compose([
        # transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])



        bbox_list = np.zeros((20,4))
        bbox_id_list = np.zeros((20))

        counter = 0


        bbox_dict = {}


        for i in range(len(obstacle_ids)):
            if counter >= 20:
                break
            id = obstacle_ids[i]
            bbox = obstacle_boxes[i]
            bbox_dict[id] = bbox

            bbox_list[counter] = bbox
            bbox_id_list[counter] = id
            counter += 1

        for i in range(len(obj_ids)):
            id = obj_ids[i]
            bbox = boxes[i]
            # remove ego car 
            if id == (self.ego_id%65536):
                continue
            bbox_dict[id] = bbox

            bbox_list[counter] = bbox
            bbox_id_list[counter] = id
            counter += 1


        if self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing"  :

            # print(len(self.rgb_list_bc_method))
            if len(self.rgb_list_bc_method) < 6:
                self.rgb_list_bc_method.append(camera_transforms(rgb)) 
                self.bbox_list_bc_method.append(bbox_dict)

                self.bbox_list_DSA.append(np.array(bbox_list).astype(np.float32))
                
                self.bbox_id_list_DSA.append(bbox_id_list)

            else:
                # pop first one
                self.rgb_list_bc_method.pop(0)
                self.bbox_list_bc_method.pop(0)
                self.bbox_list_DSA.pop(0)
                self.bbox_id_list_DSA.pop(0)

        if pre_get_data:
            return

        actor_dict = self.collect_actor_data(world, frame)
        ego_id = self.ego_id 
        ego_pos = Loc(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_yaw = actor_dict[ego_id]["rotation"]["yaw"]

        # if scneario is non-interactive, obstacle --> interactor_id is -1 
        interactor_id = self.gt_interactor

        all_ids_list = actor_dict["all_ids"]
        
        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []

        # for risk vis 
        risk_bbox_list=[]
        other_bbox_list=[]

        # interactive id 
        vehicle_id_list = list(actor_dict["vehicles_ids"])
        # pedestrian id list 
        pedestrian_id_list = list(actor_dict["pedestrian_ids"])
        # obstacle id list 
        obstacle_id_list = list(actor_dict["obstacle_ids"])
        
        obstacle_dict = {}
        for id in obstacle_id_list:
            obstacle_dict[id] = actor_dict["obstacle"][id]["type_id"]

        # trajectory based method 
        if self.mode == "Kalman_Filter" or self.mode == "MANTRA" or self.mode == "Social-GAN" or self.mode == "QCNet":
            vehicle_list = []
            traj_df = pd.DataFrame(self.df_list, columns=['FRAME', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'VELOCITY_X', 'VELOCITY_Y', 'YAW'])

            for _, remain_df in traj_df.groupby('TRACK_ID'): # actor id 
                filter = (remain_df.FRAME > ( int(frame) - 20))

                remain_df = remain_df[filter].reset_index(drop=True)
                #print("remain_df: ", remain_df)
                remain_df = remain_df.reset_index(drop=True)

                now_df = remain_df[remain_df.FRAME == int(frame)]
                actor_pos_x = float(now_df["X"].values[0])
                actor_pos_y = float(now_df["Y"].values[0])
                # actor_pos_x = float( remain_df.loc[19, 'X'])
                # actor_pos_y = float( remain_df.loc[ int(frame), 'Y'])
                dist_x = actor_pos_x - actor_dict[ego_id]["location"]["x"]
                dist_y = actor_pos_y - actor_dict[ego_id]["location"]["y"]
                remain_df["X"] = remain_df["X"].astype(float)
                remain_df["Y"] = remain_df["Y"].astype(float)
                #print(actor_pos_x, actor_pos_y, "dist_x", dist_x, "dist_y", dist_y)
                if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                    vehicle_list.append(remain_df)
            if self.mode == "Kalman_Filter":
                risky_ids = self.kf_inference(vehicle_list, frame, ego_id, pedestrian_id_list, vehicle_id_list, obstacle_id_list)
                risky_ids = risky_ids[:1]
            if self.mode == "MANTRA":
                risky_ids = self.mantra_inference(vehicle_list, frame, ego_id, pedestrian_id_list, vehicle_id_list, obstacle_dict)
                risky_ids = risky_ids[:1]
            if self.mode == "Social-GAN":
                risky_ids = self.socal_gan_inference(vehicle_list, frame, ego_id, pedestrian_id_list, vehicle_id_list, obstacle_dict, self._args, self.generator)
                risky_ids = risky_ids[:1]
            if self.mode == "QCNet":
                risky_ids = self.QCNet_inference(vehicle_list, frame, ego_id, pedestrian_id_list, vehicle_id_list, obstacle_dict)
                risky_ids = risky_ids[:1]
        # vision based methods
        elif self.mode == "DSA" or self.mode == "RRL" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" :

            if self.mode == "DSA":
                threshold = 0.9
            else:
                threshold = 0.8
            #  input 
            imgs_input = torch.stack(self.rgb_list_bc_method) # 1 5 3 H W
            imgs_input = imgs_input.unsqueeze(0).cuda()

            bbox_input = torch.from_numpy(np.array(self.bbox_list_DSA).astype(np.float32))
            
            bbox_input = bbox_input.unsqueeze(0).cuda()

            bbox_id_input = self.bbox_id_list_DSA

            risky_ids = []
            tmp_dict = {}

            with torch.no_grad():
                _, all_alphas, _ = self.dsa_model(imgs_input, bbox_input)
                
                all_alphas = all_alphas[0]
                
                n_frame = -1
                max_score = 0
                for score, id in zip(all_alphas[n_frame], bbox_id_input[n_frame]):

                    
                    score, id = score.cpu().numpy(), id
                    if id == -1:
                        break
                    score = round(float(score),2)

                    tmp_dict[int(id)] = score

                    if score > max_score:
                        risky_ids = [int(id)]
                        max_score = score
                    if max_score  <  threshold:
                        risky_ids = []
                    # if round(float(score),2) > threshold: # 0.8
                        # risky_ids.append(int(id))

            # mean filter 

            if self.mode ==  "RRL_smoothing" or self.mode == "DSA_smoothing":
                if len(self.Mean_filter_list) < 5:
                    self.Mean_filter_list.append(tmp_dict)
                else:
                    self.Mean_filter_list.pop(0) # pop first one 
                    self.Mean_filter_list.append(tmp_dict)
                    # take the avg 
                    # get all ids
                    mean_filter_id_list = []
                    for i in range(5):
                        mean_filter_id_list += list(self.Mean_filter_list[i].keys())
                    
                    # take the avg 
                    result_dict = {}
                    for mean_filter_id in mean_filter_id_list:
                        counter = 0
                        score = 0
                        for i in range(5):
                            if mean_filter_id in self.Mean_filter_list[i].keys():
                                counter+=1
                                score+=self.Mean_filter_list[i][mean_filter_id]
                        avg_score = float(score/counter)

                        # threshold 
                        if avg_score > 0.25:
                            result_dict[mean_filter_id] = avg_score
                    

                    if len(result_dict) != 0:
                        # final find the max
                        # max_id = [key for key, value in result_dict.items() if value == max(result_dict.values())]
                        # risky_ids = max_id
                        # two_result = max_id
                        max_score = 0
                        for key in  self.Mean_filter_list[-1].keys():
                            if key in result_dict.keys():
                                value = result_dict[key]
                                if value > max_score:
                                    value = max_score
                                    risky_ids = [key]


                    else:
                        # two_result = []
                        risky_ids = []
                        

        elif self.mode == "BP" or self.mode == "BCP" or self.mode == "BCP_smoothing" or self.mode == "BP_smoothing":
            tracking_results = []
            for i in range(5):
                for actor_id in self.bbox_list_bc_method[i]:

                    bbox = self.bbox_list_bc_method[i][actor_id]
                    w = bbox[2]-bbox[0]
                    h = bbox[3]-bbox[1]

                    if w*h < 100: #MIN_AREA:
                        continue

                    tracking_results.append([int(i), int(actor_id), bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1])
            tracking = np.array(tracking_results)

            two_result = []
            single_result = []
    
            if tracking.shape[0] != 0:
                t_array = tracking[:, 0]

                tracking_id = tracking[np.where(t_array == 4)[0], 1]

                trackers = np.zeros([5, 25, 4])

                for t in range(5):
                    current_tracking = tracking[np.where(t_array == t)[0]]

                    for i, object_id in enumerate(tracking_id):
                        current_actor_id_idx = np.where(
                            current_tracking[:, 1] == object_id)[0]

                        if len(current_actor_id_idx) != 0:
                            # x1, y1, x2, y2
                            bbox = current_tracking[current_actor_id_idx, 2:6]
                            trackers[t, i, :] = bbox

                # return trackers, tracking_id
                with torch.no_grad():
                    """
                        single_result, two_result: Dictionary
                        e.g.
                        {
                            object_id1 (str): True
                            object_id2 (str): False
                            object_id3 (str): True
                            ...
                        }
                    """
                    single_result, two_result, two_score_dict, single_score_dict = self.BC_testing(self.BC_model, self.rgb_list_bc_method, trackers, tracking_id)

                    if self.mode ==  "BP_smoothing" : #or  :
                        if len(self.Mean_filter_list) < 5:
                            self.Mean_filter_list.append(single_score_dict)
                        else:
                            self.Mean_filter_list.pop(0) # pop first one 
                            self.Mean_filter_list.append(single_score_dict)
                            # take the avg 
                            # get all ids
                            mean_filter_id_list = []
                            for i in range(5):
                                mean_filter_id_list += list(self.Mean_filter_list[i].keys())
                            
                            # take the avg 
                            result_dict = {}
                            for mean_filter_id in mean_filter_id_list:
                                counter = 0
                                score = 0
                                for i in range(5):
                                    if mean_filter_id in self.Mean_filter_list[i].keys():
                                        counter+=1
                                        score+=self.Mean_filter_list[i][mean_filter_id]
                                avg_score = float(score/counter)

                                # threshold 
                                if avg_score > 0.18:
                                    result_dict[mean_filter_id] = avg_score

                            if len(result_dict) != 0:
                                # final find the max
                                # max_id = [key for key, value in result_dict.items() if value == max(result_dict.values())]
                                # two_result = max_id

                                max_score = 0
                                
                                for key in  self.Mean_filter_list[-1].keys():
                                    if key in result_dict.keys():
                                        value = result_dict[key]
                                        if value > max_score:
                                            value = max_score
                                            single_result = [key]
                            else:
                                single_result = []

                if self.mode ==  "BCP_smoothing" : #or  :
                    if len(self.Mean_filter_list) < 5:
                        self.Mean_filter_list.append(two_score_dict)
                    else:
                        self.Mean_filter_list.pop(0) # pop first one 
                        self.Mean_filter_list.append(two_score_dict)
                        # take the avg 
                        # get all ids
                        mean_filter_id_list = []
                        for i in range(5):
                            mean_filter_id_list += list(self.Mean_filter_list[i].keys())
                        
                        # take the avg 
                        result_dict = {}
                        for mean_filter_id in mean_filter_id_list:
                            counter = 0
                            score = 0
                            for i in range(5):
                                if mean_filter_id in self.Mean_filter_list[i].keys():
                                    counter+=1
                                    score+=self.Mean_filter_list[i][mean_filter_id]
                            avg_score = float(score/counter)

                            # threshold 
                            if avg_score > 0.18:
                                result_dict[mean_filter_id] = avg_score

                        if len(result_dict) != 0:
                            # final find the max
                            # max_id = [key for key, value in result_dict.items() if value == max(result_dict.values())]
                            # two_result = max_id

                            max_score = 0
                            
                            for key in  self.Mean_filter_list[-1].keys():
                                if key in result_dict.keys():
                                    value = result_dict[key]
                                    if value > max_score:
                                        value = max_score
                                        two_result = [key]
                        else:
                            two_result = []
                            

            if self.mode == "BCP" or  self.mode == "BCP_smoothing":
                risky_ids = two_result
            else:
                risky_ids = single_result

        # rule based methods
        elif self.mode == "Random":
            # ids = list(obstacle_ids) + list(obj_ids)

            # for id in ids:
            #     if id == (self.ego_id % 65536):
            #         ids.remove(id)
            # all_ids_list

            if len(all_ids_list) != 0 :
                risky_ids = [random.choice(all_ids_list)]
                # for id in ids:
                #     if random.random() > 0.5:
                #         risky_ids.append(id)

            else:
                risky_ids = []
                
        elif self.mode == "Range":
            ids = list(obstacle_ids) + list(obj_ids)
            if len(ids) == 0 :
                risky_ids = []
            else:
                # risky_ids = []
                # find nearest object 
                min_distance = 1000
                min_id = -1

                for id in self.obestacle_id_list:
                    id_tmp = id % 65536 
                    if id_tmp in ids:
                        distance = actor_dict["obstacle"][id]["distance"]

                        if distance < min_distance:
                            min_distance = distance
                            min_id = id

                for id in vehicle_id_list:
                    if id == self.ego_id:
                        continue
                    id_tmp = id % 65536 
                    if id_tmp in ids:
                        distance = actor_dict[id]["distance"]
                        if distance < min_distance:
                            min_distance = distance
                            min_id = id
                for id in pedestrian_id_list:
                    id_tmp = id % 65536
                    if id_tmp in ids:
                        distance = actor_dict[id]["distance"]
                        if distance < min_distance:
                            min_distance = distance
                            min_id = id

                if min_distance > 10 :#15:
                    risky_ids = []
                else:
                    risky_ids = [min_id]

        else:
            risky_ids = []
        # Get bbox for lbc Input 

        print("***************************************************")
        
        tmp = []
        for id in risky_ids:
            tmp.append(int(id))
        risky_ids = tmp
        print("          risky id: ", risky_ids)    
        print("Ground obstacle id: ", self.gt_obstacle_id_list)
        print("     Interactor id: ", self.gt_interactor)


        if self.args.obstacle_region:
            if not (self.mode == "Ground_Truth" or self.mode == "Full_Observation"):     
                for id in self.gt_obstacle_id_list:
                    if self.mode == "Range" or  self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing":
                        id = id % 65536
                    if id in risky_ids:

                        for gt_id in self.gt_obstacle_id_list:
                            try:
                                pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                                pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                                pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                                pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]

                                obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                    Loc(x=pos_1[0], y=pos_1[1]), 
                                    Loc(x=pos_2[0], y=pos_2[1]), 
                                    Loc(x=pos_3[0], y=pos_3[1]), 
                                    ])
                            except:
                                pass
            else:
                for gt_id in self.gt_obstacle_id_list:
                    try:
                    
                        pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                        pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                        pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                        pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]

                        obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                            Loc(x=pos_1[0], y=pos_1[1]), 
                            Loc(x=pos_2[0], y=pos_2[1]), 
                            Loc(x=pos_3[0], y=pos_3[1]), 
                            ])
                        print("append gt obstacle id ")
                    except:
                        pass
        else:
            for gt_id in self.gt_obstacle_id_list:
                try:
                
                    pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                    pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                    pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                    pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]

                    obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                        Loc(x=pos_1[0], y=pos_1[1]), 
                        Loc(x=pos_2[0], y=pos_2[1]), 
                        Loc(x=pos_3[0], y=pos_3[1]), 
                        ])
                except:
                    continue


                if self.mode == "Ground_Truth" or self.mode == "Full_Observation":
                    if id in self.gt_obstacle_id_list :
                        try:
                            pos_0 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_0"]
                            pos_1 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_4"]
                            pos_2 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_6"]
                            pos_3 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_2"]

                            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])
                        except:
                            pass
                    else: 
                        if self.mode == "Ground_Truth":
                            continue
                        try:
                            pos_0 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_0"]
                            pos_1 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_4"]
                            pos_2 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_6"]
                            pos_3 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_2"]

                            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])
                        except:
                            pass
                else:
                    # risky_ids
                    try:
                        pos_0 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_0"]
                        pos_1 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_4"]
                        pos_2 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_6"]
                        pos_3 = actor_dict["obstacle"][id]["cord_bounding_box"]["cord_2"]

                        if self.mode == "Range" or  self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" :
                            id = id % 65536
                        if id in risky_ids:
                            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])
                    except:
                        pass
        
        for id in vehicle_id_list:
            # Draw ego car 
            if int(id) == int(ego_id):
                pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
                pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
                pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
                pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

                agent_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
                continue
                

            
            if self.mode == "Ground_Truth" or self.mode == "Full_Observation":   
                pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
                pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
                pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
                pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

                if int(id) == int(interactor_id):
                    vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
                        
                elif id in self.ill_parking_id_list:
                    
                    if id in self.gt_obstacle_id_list:
                        obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
                    else:
                        if self.mode == "Ground_Truth":
                            continue
                        obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
                else:
                    if self.mode == "Ground_Truth":
                        continue
                    vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
            else:

                pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
                pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
                pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
                pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

                if self.mode == "Range" or  self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" :
                    id = id % 65536

                
                if id in risky_ids:
                    if self.scenario_type == "obstacle":
                        obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                Loc(x=pos_1[0], y=pos_1[1]), 
                                Loc(x=pos_2[0], y=pos_2[1]), 
                                Loc(x=pos_3[0], y=pos_3[1]), 
                                ])
                    else:
                        vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                Loc(x=pos_1[0], y=pos_1[1]), 
                                Loc(x=pos_2[0], y=pos_2[1]), 
                                Loc(x=pos_3[0], y=pos_3[1]), 
                                ])

    
        for id in pedestrian_id_list:
            if self.mode == "Ground_Truth" or self.mode == "Full_Observation":
                pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
                pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
                pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
                pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

                if int(id) == int(interactor_id):
                    pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
                        
                else:
                    if self.mode == "Ground_Truth":
                        continue
                    pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
            else:
                pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
                pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
                pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
                pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

                if self.mode == "Range" or  self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing":
                    id = id % 65536

                if id in risky_ids:
                    # mod 65536
                    
                    pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                    


        ### for vis ############################################

        for id in pedestrian_id_list:
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            if self.mode == "Full_Observation" or self.mode =="AUTO":
                risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
            elif self.mode == "Ground_Truth":
                if int(id) == int(interactor_id):
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])  
                else:
                    other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
            else:
                # other method 
                if self.mode == "Range" or  self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" :
                    id = id % 65536

                if id in risky_ids:
                    # mod 65536
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                else:
                    other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                                 
        for id in vehicle_id_list:
            if int(id) == int(ego_id):
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

            if self.mode == "Full_Observation":
                risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
                
            elif self.mode == "Ground_Truth":
                if int(id) == int(interactor_id):
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])  
                elif id in self.gt_obstacle_id_list:
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ]) 
                else:
                    other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
            else:
                # other method 
                if self.mode == "Range" or self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" :
                    id = id % 65536

                if id in risky_ids:
                    # mod 65536
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                else:
                    other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])

                      
        if self.args.obstacle_region:


            if self.mode == "Full_Observation" or self.mode == "Ground_Truth":

                for gt_id in self.gt_obstacle_id_list:

                    try:
                        pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                        pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                        pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                        pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]
                    except:
                        continue
                    
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
            else:
                in_risky_id_flag = False
                
                for gt_id in self.gt_obstacle_id_list:
                    if self.mode == "Range" or self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing" : 
                        id = gt_id % 65536
                    if id in risky_ids:
                        in_risky_id_flag = True
                
                
                for gt_id in self.gt_obstacle_id_list:

                    try:
                        pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                        pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                        pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                        pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]
                    except:
                        continue
                    if in_risky_id_flag:
                        risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])

                    else:
                        other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
        else:
            for gt_id in self.gt_obstacle_id_list:
                try:
                    pos_0 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_0"]
                    pos_1 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_4"]
                    pos_2 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_6"]
                    pos_3 = actor_dict["obstacle"][gt_id]["cord_bounding_box"]["cord_2"]
                except:
                    continue
                if self.mode == "Full_Observation":
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                elif self.mode == "Ground_Truth":
                    risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ]) 
                else:
                    if self.mode == "Range" or self.mode == "DSA" or self.mode == "RRL" or self.mode == "BP" or self.mode == "BCP" or  self.mode == "BCP_smoothing" or self.mode == "BP_smoothing" or self.mode == "RRL_smoothing" or self.mode == "DSA_smoothing": 
                        id = gt_id % 65536

                    if not self.args.obstacle_region:

                        if id in risky_ids:
                            risk_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])
                        else:
                            other_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                                        ])
          


        if self.mode == "Full_Observation" or self.mode == "AUTO":
            risk_bbox_list += self.static_vehicle_bbox_list
        else:
            other_bbox_list += self.static_vehicle_bbox_list 


        # static vehicle bbox 
        # self.static_vehicle_bbox_list
        if self.mode == "Full_Observation" or self.mode == "AUTO":
            vehicle_bbox_list = vehicle_bbox_list + self.static_vehicle_bbox_list

        birdview: BirdView = self.birdview_producer.produce(ego_pos, yaw=ego_yaw,
                                                       agent_bbox_list=agent_bbox_list, 
                                                       vehicle_bbox_list=vehicle_bbox_list,
                                                       pedestrians_bbox_list=pedestrian_bbox_list,
                                                       obstacle_bbox_list=obstacle_bbox_list)
    


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        topdown = BirdViewProducer.as_ss(birdview)

        # N_CLASSES

        topdown = torch.LongTensor(topdown)
        topdown = torch.nn.functional.one_hot(topdown, 7).permute(2, 0, 1).float()

        topdown = topdown.reshape([1, 7, 256, 256])
        topdown = topdown.to(device)



        u = np.float32([ego_pos.x, ego_pos.y])
                  
        # yaw = theta - 450
        theta = ego_yaw + 450
        theta = math.radians(theta)
        if np.isnan(theta):
             theta = 0.0
        
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        target = R.T.dot(self.v - u)
        target *= 5
        target += [128, 128] 
        
        target = np.clip(target, 0, 256)
        target_xy = torch.FloatTensor(target)
        target_xy = target_xy.reshape((1, 2))
        target_xy = target_xy.to(device)
        
        
        if self.mode == "AUTO":
            
            control = self.agent.run_step()
            control.manual_gear_shift = False
            # world.player.apply_control(control)
            
        else:

            with torch.no_grad():
                points_pred = self.net.forward(topdown, target_xy)
                control = self.net.controller(points_pred).cpu().data.numpy()[0]

            steer = control[0] 
            desired_speed = control[1] 
            
            speed = world.player.get_velocity()
            speed = ((speed.x)**2 + (speed.y)**2+(speed.z)**2)**(0.5)

            brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1


            delta = np.clip(desired_speed - speed, 0.0, 0.5)
            throttle = self.ego_speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75) 
            throttle = throttle if not brake else 0.0

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle) 
            control.brake = float(brake)
        
        
    


        # vis the result 
        # draw target point on BEV map 

        birdview: BirdView = self.birdview_producer.produce(ego_pos, yaw=ego_yaw,
                                                agent_bbox_list=agent_bbox_list, 
                                                vehicle_bbox_list=[],
                                                pedestrians_bbox_list=[],
                                                obstacle_bbox_list=[],
                                                risk_bbox_list=risk_bbox_list,
                                                other_bbox_list=other_bbox_list,
                                                risk_vis=True)

        topview_rgb = BirdViewProducer.as_rgb(birdview)
        _topdown = Image.fromarray(topview_rgb)

        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse((target[0]-2, target[1]-2, target[0]+2, target[1]+2), (0, 0, 255))


        if self.mode != "AUTO":
            for x, y in points_pred.cpu().data.numpy()[0]:
                x = (x + 1) / 2 * 256
                y = (y + 1) / 2 * 256

                _draw.ellipse((x-2, y-2, x+2, y+2), (35,80,127))#(255, 0, 0))

        _topdown = cv2.cvtColor(np.asarray(_topdown), cv2.COLOR_RGB2BGR)


        self.topdown_debug_list.append(_topdown)

        if self.scenario_type == "interactive" or self.scenario_type == "collision":
           
            interactor_location = world.world.get_actor(self.gt_interactor).get_location()

            distance = math.sqrt((ego_pos.x - interactor_location.x)**2 + (ego_pos.y - interactor_location.y)**2)

            self.avg_distance += distance
            self.counter_avg_distance +=1

            if distance < self.min_distance:
                self.min_distance = distance

        elif self.scenario_type == "obstacle":

            #for id in self.gt_obstacle_id_list:
            id = self.gt_obstacle_id_nearest
            interactor_location = world.world.get_actor(id).get_location()
            distance = math.sqrt((ego_pos.x - interactor_location.x)**2 + (ego_pos.y - interactor_location.y)**2)
            
            self.avg_distance += distance
            self.counter_avg_distance +=1
            
            if distance < self.min_distance:
                self.min_distance = distance

        # draw waypoints
        distance = math.sqrt((ego_pos.x - self.end_position_x)**2 + (ego_pos.y - self.end_position_y)**2)
        isReach = False
        # print(distance)
        
        if distance < 1.0:
            isReach = True

        if self.counter > 120:
            isReach = True

        if isReach:
            self.save_video()

        self.counter+=1
        return control, isReach
    
    def save_video(self):

        self.front_rgb_out.release()
        
        path = self.variant_path.split("data_collection/")[1].replace("/", "#")

        if self.args.obstacle_region:
        
            if not os.path.exists(f"./{self.scenario_type}_region_results/{self.mode}"):
                os.makedirs(f"./{self.scenario_type}_region_results/{self.mode}")
            out = cv2.VideoWriter(f'./{self.scenario_type}_region_results/{self.mode}/{path}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,  (256, 256)) 
            for img in self.topdown_debug_list:
                out.write(img)
            out.release()
        else:
            if not os.path.exists(f"./{self.scenario_type}_results/{self.mode}"):
                os.makedirs(f"./{self.scenario_type}_results/{self.mode}")
            out = cv2.VideoWriter(f'./{self.scenario_type}_results/{self.mode}/{path}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,  (256, 256)) 
            for img in self.topdown_debug_list:
                out.write(img)
            out.release()

        self.avg_distance  = float(self.avg_distance/self.counter_avg_distance)
        

        with open("./result.txt", "a") as f:
            f.write(
                f"{self.scenario_type}#{self.scenario_id}#{self.map}#{self.weather}#{self.actor}#{self.seed}#{self.min_distance}#{self.avg_distance}#{self.collision_flag}\n")

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    path = os.path.join('data_collection',
                        args.scenario_type, args.scenario_id)

    filter_dict = {}
    try:
        for root, _, files in os.walk(path + '/filter/'):
            for name in files:
                f = open(path + '/filter/' + name, 'r')
                bp = f.readlines()[0]
                name = name.strip('.txt')
                f.close()
                if args.scenario_type != "obstacle" or name == "player":
                    filter_dict[name] = bp
        # print(filter_dict)
    except:
        print("檔案夾不存在。")

    # generate random seed
    # during replay, we use seed to generate the same behavior
    random_seed_int = int(args.random_seed)
    if random_seed_int == 0:
        random_seed_int = random.randint(0, 100000)
    random.seed(random_seed_int)
    seeds = []
    for _ in range(12):
        seeds.append(random.randint(1565169134, 2665169134))

    # load files for scenario reproducing
    transform_dict = {}
    velocity_dict = {}
    ped_control_dict = {}
    for actor_id, filter in filter_dict.items():
        transform_dict[actor_id] = read_transform(
            os.path.join(path, 'transform', actor_id + '.npy'))
        velocity_dict[actor_id] = read_velocity(
            os.path.join(path, 'velocity', actor_id + '.npy'))
        if 'pedestrian' in filter:
            ped_control_dict[actor_id] = read_ped_control(
                os.path.join(path, 'ped_control', actor_id + '.npy'))
    abandon_scenario = False
    scenario_name = None

    vehicles_list = []
    all_id = []

    collection_flag = False
    detect_start = True
    detect_end = False
    # read start position and end position
    if (args.inference and not args.generate_random_seed) or not args.no_save:
        with open(f"{path}/start_end_point.json") as f:
            data = json.load(f)
            start_position_x = float(data["start_x"])
            start_position_y = float(data["start_y"])
            end_position_x = float(data["end_x"])
            end_position_y = float(data["end_y"])






    client = carla.Client(args.host, args.port)
    client.reload_world()
    client.set_timeout(10.0)
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0, 0, 0))
    pygame.display.flip()

    hud = HUD(args.width, args.height, client.get_world(), args)

    weather = args.weather

    exec("args.weather = carla.WeatherParameters.%s" % args.weather)
    stored_path = os.path.join('data_collection', args.scenario_type, args.scenario_id,
                               'variant_scenario', weather + "_" + args.random_actors + "_")

    if args.test:
        out = cv2.VideoWriter(f'data_collection/{args.scenario_type}/{args.scenario_id}/{args.scenario_id}.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), 20,  (640, 360))
    else:
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        out = cv2.VideoWriter(stored_path+"/"+str(args.scenario_id)+".mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), 20,  (640, 360))

    # pass seeds to the world
    world = World(client.load_world(args.map),
                  filter_dict['player'], hud, args, seeds)

    client.get_world().set_weather(args.weather)
    # client.get_world().set_weather(getattr(carla.WeatherParameters, args.weather))
    # sync mode
    settings = world.world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True  # Enables synchronous mode
    world.world.apply_settings(settings)

    # other setting
    controller = KeyboardControl(world, args.autopilot)
    blueprint_library = client.get_world().get_blueprint_library()

    lights = []
    actors = world.world.get_actors().filter('traffic.traffic_light*')
    for l in actors:
        lights.append(l)
    light_dict, light_transform_dict = read_traffic_lights(path, lights)
    clock = pygame.time.Clock()

    agents_dict = {}
    controller_dict = {}
    actor_transform_index = {}
    finish = {}

    # init position for player
    ego_transform = transform_dict['player'][0]
    ego_transform.location.z += 3
    world.player.set_transform(ego_transform)
    agents_dict['player'] = world.player

    # generate obstacles and calculate the distance between ego-car and nearest obstacle
    min_dis = float('Inf')
    nearest_obstacle = -1
    if args.scenario_type == 'obstacle':
        obstacle_GT_location = np.array([])
        if args.map =="Town10HD" or args.map == "A6":    
            obstacle_GT_location = np.load(f"{stored_path}/obstacle_location.npy")


        # GT_obstacle_location 
        nearest_obstacle_id, obstacle_info, obstacle_static_id_list, ill_parking_id_list = generate_obstacle(client.get_world(), blueprint_library,
                                                            path+"/obstacle/obstacle_list.json", ego_transform, obstacle_GT_location)

    # set controller
    for actor_id, bp in filter_dict.items():

        if actor_id != 'player':
            transform_spawn = transform_dict[actor_id][0]

            while True:
                try:
                    agents_dict[actor_id] = client.get_world().spawn_actor(
                        set_bp(blueprint_library.filter(
                            filter_dict[actor_id])),
                        transform_spawn)
                    break
                except Exception:
                    transform_spawn.location.z += 1.5

            # set other actor id for checking collision object's identity
            world.collision_sensor.other_actor_id = agents_dict[actor_id].id

            # 

        if 'vehicle' in bp:
            controller_dict[actor_id] = VehiclePIDController(agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                             max_throttle=1.0, max_brake=1.0, max_steering=1.0)
            try:
                agents_dict[actor_id].set_light_state(
                    carla.VehicleLightState.LowBeam)
            except:
                print('vehicle has no low beam light')

        actor_transform_index[actor_id] = 1
        finish[actor_id] = False

    if args.scenario_type == "obstacle" and not args.no_save and not args.test:
        with open(os.path.join(stored_path, "obstacle_info.json"), "w")as f:
            json.dump(obstacle_info, f, indent=4)

    # root = os.path.join('data_collection', args.scenario_type, args.scenario_id)
    scenario_name = str(weather) + '_'

    if args.random_actors != 'none':
        if args.random_actors == 'pedestrian':  # only pedestrian
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=30, v_ratio=0.0,
                                                                   pedestrian=40, transform_dict=transform_dict)
        elif args.random_actors == 'low':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.3,
                                                                   pedestrian=20, transform_dict=transform_dict)
        elif args.random_actors == 'mid':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.6,
                                                                   pedestrian=45, transform_dict=transform_dict)
        elif args.random_actors == 'high':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.8,
                                                                   pedestrian=70, transform_dict=transform_dict)

    scenario_name = scenario_name + args.random_actors + '_'

    # write actor list
    # min_id, max_id = check_actor_list(world)
    # if max_id-min_id >= 65535:
    #     print('Actor id error. Abandom.')
    #     abandon_scenario = True
    #     raise

    iter_tick = 0
    iter_start = 25
    iter_toggle = 50

    if not args.no_save :
        data_collection = Data_Collection()
        data_collection.set_scenario_type(args.scenario_type)
        data_collection.set_ego_id(world)
        data_collection.set_attribute(
            args.scenario_type, args.scenario_id, weather, args.random_actors, args.random_seed, args.map)
        
    if args.inference:
    
        inference = Inference(args, stored_path, weather)
        inference.set_end_position(end_position_x, end_position_y)

        if args.scenario_type == 'obstacle':
            inference.set_obstacle_ids(obstacle_static_id_list, ill_parking_id_list)

            # nearest_obstacle


        gt_interactor_id = -1
        if args.scenario_type == "interactive" or args.scenario_type == "collision":
            keys = list(agents_dict.keys())
            keys.remove('player')
            gt_interactor_id = int(agents_dict[keys[0]].id)
            inference.set_gt_interactor(gt_interactor_id )
            world.collision_sensor.other_actor_id = gt_interactor_id

        

        inference.set_scenario_type(args.scenario_type)
        inference.set_ego_id(world)
        
        if args.mode == "AUTO":
            inference.set_autopilot(world)

        # only testing set has GT obstacle location
        if args.scenario_type == "obstacle":
            if args.map =="Town10HD" or args.map == "A6":
                
                obstacle_GT_location = np.load(f"{stored_path}/obstacle_location.npy")
                inference.set_gt_obstacle_ids(obstacle_GT_location, world, nearest_obstacle_id)

                world.collision_sensor.other_actor_id = nearest_obstacle_id
                world.collision_sensor.other_actor_ids = (obstacle_static_id_list+ ill_parking_id_list)

                


    if args.scenario_type:
        collision_detect_end = False
        collision_counter = 0

    while (1):
        clock.tick_busy_loop(40)
        frame = world.world.tick()

        hud.frame = frame
        iter_tick += 1
        if iter_tick == iter_start + 1:
            ref_light = get_next_traffic_light(
                world.player, world.world, light_transform_dict)
            annotate = annotate_trafficlight_in_group(
                ref_light, lights, world.world)

        elif iter_tick > iter_start:

            if not args.no_save and not args.inference:
                if args.scenario_type == "interactive" or args.scenario_type == "collision":
                    keys = list(agents_dict.keys())
                    keys.remove('player')
                    gt_interactor_id = int(agents_dict[keys[0]].id)
                    data_collection.set_gt_interactor(gt_interactor_id)

            # iterate actors
            for actor_id, _ in filter_dict.items():

                # apply recorded location and velocity on the controller
                actors = world.world.get_actors()
                # reproduce traffic light state
                if actor_id == 'player' and ref_light:
                    set_light_state(
                        lights, light_dict, actor_transform_index[actor_id], annotate)

                if actor_transform_index[actor_id] < len(transform_dict[actor_id]):
                    x = transform_dict[actor_id][actor_transform_index[actor_id]].location.x
                    y = transform_dict[actor_id][actor_transform_index[actor_id]].location.y

                    if 'vehicle' in filter_dict[actor_id]:

                        if not detect_start:
                        
                            if args.inference and actor_id == 'player':
                                # Not to apply control for ego vehicle ( player )
                                continue

                        target_speed = (
                            velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6
                        waypoint = transform_dict[actor_id][actor_transform_index[actor_id]]

                        agents_dict[actor_id].apply_control(
                            controller_dict[actor_id].run_step(target_speed, waypoint))
                        # agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                        #     (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6, transform_dict[actor_id][actor_transform_index[actor_id]]))

                        v = agents_dict[actor_id].get_velocity()
                        v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                        # to avoid the actor slowing down for the dense location around
                        if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                            actor_transform_index[actor_id] += 2
                        elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                            actor_transform_index[actor_id] += 6
                        else:
                            actor_transform_index[actor_id] += 1

                    elif 'pedestrian' in filter_dict[actor_id]:
                        agents_dict[actor_id].apply_control(
                            ped_control_dict[actor_id][actor_transform_index[actor_id]])
                        actor_transform_index[actor_id] += 1
                else:
                    finish[actor_id] = True

            if args.inference:
                if detect_start:
                    
                    if args.mode == "BP" or  args.mode == "BCP" or  args.mode ==  "DSA" or  args.mode ==  "RRL" or args.mode =="BCP_smoothing" or args.mode == "BP_smoothing" or args.mode == "RRL_smoothing" or args.mode == "DSA_smoothing":
                        inference.run_inference(frame, world, True)
                    else:
                        inference.collect_actor_data(world, frame)
                
                
                if not detect_start:
                    if args.inference:
                        control, isReach = inference.run_inference(frame, world)
                        world.player.apply_control(control)

                        if isReach:
                            break
                    else:
                        if args.mode =="AUTO":    
                            inference.agent.run_step()
                        
                else:
                    if args.mode =="AUTO":    
                        inference.agent.run_step()

            if not False in finish.values():
                break

            if controller.parse_events(client, world, clock) == 1:
                return


# 
            if world.collision_sensor.true_collision and args.scenario_type != 'collision':
                print('True_collision, abandon scenario')
                inference.collision_flag = True
                abandon_scenario = True


            if world.collision_sensor.collision and args.scenario_type != 'collision':
                print('unintentional collision')

                if not args.inference:
                    abandon_scenario = True


            
            elif world.collision_sensor.wrong_collision:
                print('collided with wrong object')
                if not args.inference:  
                    abandon_scenario = True
            if abandon_scenario:
                if args.inference:
                    inference.save_video()

                world.abandon_scenario = True
                break

            elif iter_tick > iter_toggle:

                if not args.no_save and (not abandon_scenario) and collection_flag and detect_start == False and not args.inference:
                    # collect data in sensor's list
                    data_collection.collect_sensor(frame, world)

                view = pygame.surfarray.array3d(display)
                #  convert from (width, height, channel) to (height, width, channel)
                view = view.transpose([1, 0, 2])
                #  convert from rgb to bgr
                image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                out.write(image)

                ego_loc = world.player.get_location()
                x = ego_loc.x
                y = ego_loc.y

                if not args.test and not args.generate_random_seed:
                    if detect_start:
                        distacne = math.sqrt(
                            (x - start_position_x)**2 + (y - start_position_y)**2)
                        if distacne < 1.0:
                            detect_start = False
                            collection_flag = True
                            detect_end = True
                            if not args.no_save and not args.inference:
                                data_collection.set_start_frame(frame)

                    if not args.inference or not args.no_save:
                        # check end point
                        if detect_end:
                            if args.scenario_type != "collision":
                                
                                distacne = math.sqrt(
                                    (x - end_position_x)**2 + (y - end_position_y)**2)
                                if distacne < 1.0:
                                    collection_flag = False
                            else:
                                # detect collision 
                                if world.collision_sensor.collision and not world.collision_sensor.wrong_collision and not collision_detect_end:
                                    collision_detect_end = True 
                                    collision_counter = 0
                                
                                    if not args.no_save and not args.inference:
                                        data_collection.save_collision_frame(frame,  world.collision_sensor.collision_actor_id, world.collision_sensor.collision_actor_type, stored_path)
                                if collision_detect_end:
                                    collision_counter+=1

                                if collision_counter > 10:
                                    collection_flag = False

                if detect_end and not collection_flag:
                    print('stop scenario ')
                    break

            # cehck end position
        world.tick(clock)
        world.render(display)
        pygame.display.flip()

    if args.no_save and args.generate_random_seed and (not abandon_scenario) and not args.test:
        # save random_seed
        with open(f'{stored_path}/seed.txt', 'w') as f:
            f.write(str(random_seed_int))

    if not args.no_save and not abandon_scenario and not args.test and not args.inference:
        data_collection.set_end_frame(frame)
        # save for only one time
        data_collection.collect_actor_attr(world)
        data_collection.collect_static_actor_data(world)
        data_collection.save_data(stored_path)

    # to save a top view video
    out.release()
    print('Closing...')

    print('destroying vehicles')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    # stop walker controllers (list is [controller, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()

    print('destroying walkers')
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

    if world is not None:
        world.destroy()

    pygame.quit()

    return

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--scenario_id',
        type=str,
        required=True,
        help='name of the scenario')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x360',
        help='window resolution (default: 640x360)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')

    argparser.add_argument(
        '--weather',
        default='ClearNoon',
        type=str,
        choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
                 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset',
                 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'MidRainyNight', 'HardRainNight', 'SoftRainNight'],
        help='weather name')
    argparser.add_argument(
        '--map',
        default='Town03',
        type=str,
        required=True,
        help='map name')
    argparser.add_argument(
        '--random_actors',
        type=str,
        default='none',
        choices=['none', 'pedestrian', 'low', 'mid', 'high'],
        help='enable roaming actors')

    argparser.add_argument(
        '--scenario_type',
        type=str,
        choices=['interactive', 'collision', 'obstacle', 'non-interactive'],
        required=True,
        help='enable roaming actors')

    argparser.add_argument(
        '--test',
        action='store_true',
        help='test the Scenario')

    # no_save flag ( Only use few camera sensor)
    argparser.add_argument(
        '--no_save',
        # default=False,
        action='store_true',
        help='run scenarios only')
    
    argparser.add_argument(
        '--inference',
        # default=False,
        action='store_true',
        help='run end to end model ( inference mode )')
    
    argparser.add_argument(
        '--generate_random_seed',
        # default=False,
        action='store_true',
        help='run scenarios only')

    argparser.add_argument(
        '--random_seed',
        default=0,
        type=int,
        help='use random_seed to replay the same behavior ')
    argparser.add_argument(
        '--mode',
        type=str,
        choices=['Full_Observation', 'Ground_Truth', 'Random', 'Range', 
                'Kalman_Filter', 'Social-GAN', 'MANTRA', 'QCNet',
                'DSA', 'RRL', 'BP',
                'BCP', 'AUTO',
                 'BCP_smoothing', 'RRL_smoothing', 'DSA_smoothing', 'BP_smoothing' ],
        help='enable roaming actors')
    

    argparser.add_argument(
        '--obstacle_region',
        # default=False,
        action='store_true',
        help='run scenarios only')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)


    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
