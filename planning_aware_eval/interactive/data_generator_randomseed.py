
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.




from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

from roi_two_stage.demo import roi_two_stage_inference
from single_stage.demo import single_stage

import glob
import os
import sys
import random
import csv
import json
from turtle import back
from PIL import Image, ImageDraw
import pandas as pd


try:
    #
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    #
    # sys.path.append('../carla/agents/navigation')
    # sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')
    sys.path.append('rss/') # rss
    #sys.path.append('LBC/') # LBC

except IndexError:
    pass



## automomatic 
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from utils.heatmap import ToHeatmap

import carla
from carla import VehicleLightState as vls

from carla import ColorConverter as cc
from carla import Transform, Location, Rotation
from controller import VehiclePIDController
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import threading
from multiprocessing import Process

import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from random_actors import spawn_actor_nearby
from read_input import *
from get_and_control_trafficlight import *
# rss
from rss_sensor_benchmark import RssSensor # pylint: disable=relative-import
from rss_visualization import RssUnstructuredSceneVisualizer, RssBoundingBoxVisualizer, RssStateVisualizer # pylint: disable=relative-import
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e
    from pygame.locals import K_o
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# LBC
import torch

from LBC.map_model import MapModel

from LBC.converter import Converter
import LBC.common as common
# import pathlib
# import uuid
import copy
from collections import deque
# from controller import VehiclePIDController
import cv2
# LBC


import gdown
#



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


def write_json(filename, index, seed ):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        y = {str(index):seed}
        file_data.update(y)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, client_bp, hud, args, store_path, seeds):
        self.world = carla_world
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.abandon_scenario = False
        self.finish = False
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        self.store_path = store_path
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
        
        # # rss
        # self.dim = (args.width, args.height)
        # self.rss_sensor = None
        # self.rss_unstructured_scene_visualizer = None
        # self.rss_bounding_box_visualizer = None
        # # rss end

    def restart(self, args, seeds):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.


        if int(args.random_seed) == -1 :

            P = self.store_path.split("/")

            with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
        
                data = json.load(outfile)
                seed_1 = int(data["1"])
            
        else:
            seed_1 = seeds[1]
        random.seed(seed_1)


        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):


            if int(args.random_seed) == -1 :
                P = self.store_path.split("/")
                with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
            
                    data = json.load(outfile)
                    seed_2 = int(data["2"])

            else:
                seed_2 = seeds[2]

            random.seed(seed_2)
                    
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):
            if int(args.random_seed) == -1 :
                P = self.store_path.split("/")
                
                with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                    data = json.load(outfile)
                    seed_3 = int(data["3"])

            else:
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
            random.seed(seed_2)
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
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self.save_mode)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.background = True
        self.camera_manager.save_mode = self.save_mode

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        # rss
        if self.args.save_rss:
            self.rss_unstructured_scene_visualizer = RssUnstructuredSceneVisualizer(self.player, self.world, self.hud.dim)
            self.rss_bounding_box_visualizer = RssBoundingBoxVisualizer(self.hud.dim, self.world, self.camera_manager.sensor_top)
            self.rss_sensor = RssSensor(self.player, self.world,
                                        self.rss_unstructured_scene_visualizer, self.rss_bounding_box_visualizer, self.hud.rss_state_visualizer)
        # rss end

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
        if self.save_mode:
            sensors = [
                self.camera_manager.sensor_lbc_img,
                self.camera_manager.sensor_top,
                self.camera_manager.sensor_front,
                self.camera_manager.sensor_left,
                self.camera_manager.sensor_right,
                self.camera_manager.sensor_back,
                self.camera_manager.sensor_back_left,
                self.camera_manager.sensor_back_right,
                self.camera_manager.sensor_lidar,
                self.camera_manager.sensor_dvs,
                self.camera_manager.sensor_flow,
                self.camera_manager.sensor_lbc_ins,
                self.camera_manager.ins_top,

                self.camera_manager.ins_front,
                self.camera_manager.ins_back,
                self.camera_manager.ins_right,
                self.camera_manager.ins_left,
                self.camera_manager.ins_back_right,
                self.camera_manager.ins_back_left,

                self.camera_manager.depth_front,
                self.camera_manager.depth_right,
                self.camera_manager.depth_left,
                self.camera_manager.depth_back,
                self.camera_manager.depth_back_right,
                self.camera_manager.depth_back_left,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor]

            self.camera_manager.sensor_front = None


        else:
            sensors = [
                self.camera_manager.sensor_front,
                self.camera_manager.sensor_lbc_img,
                self.camera_manager.sensor_lbc_ins,
                self.camera_manager.sensor_top,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor,
                self.camera_manager.ins_front,
            ]

        if self.args.save_rss and self.save_mode:
            # rss
            if self.rss_sensor:
                self.rss_sensor.destroy()
            if self.rss_unstructured_scene_visualizer:
                self.rss_unstructured_scene_visualizer.destroy()


        for i, sensor in enumerate(sensors):
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass


        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = None
            self.control_list = None
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        r = 2
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return 1
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_e:
                    xx = int(input("x: "))
                    yy = int(input("y: "))
                    zz = int(input("z: "))
                    new_location = carla.Location(xx, yy, zz)
                    world.player.set_location(new_location)
                elif event.key == K_o:
                    xyz = [float(s) for s in input(
                        'Enter coordinate: x , y , z  : ').split()]
                    new_location = carla.Location(xyz[0], xyz[1], xyz[2])
                    world.player.set_location(new_location)
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    scenario_name = None
                    world.camera_manager.recording = not world.camera_manager.recording
                    # world.lidar_sensor.recording= not  world.lidar_sensor.recording
                    # if not  world.lidar_sensor.recording:
                    if not world.camera_manager.recording:
                        scenario_name = input("scenario id: ")
                    world.camera_manager.toggle_recording(scenario_name)

                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")

                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        "Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec",
                                       world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker
                if event.key == K_r:
                    r = 3

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(
                    pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(
                        carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(
                    pygame.key.get_pressed(), clock.get_time(), world)
            # world.player.apply_control(self._control)
            return 0

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, world, args):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._world = world
        self.args = args
        self.frame = 0
        if self.args.save_rss: # rss
            self._world = world 
            self.rss_state_visualizer = RssStateVisualizer(self.dim, self._font_mono, self._world)

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()

        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            'Frame:   %s' % self.frame,
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            #'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            #'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            #'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            #'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
            if self.args.save_rss:
                self.rss_state_visualizer.render(display, v_offset) # rss
        self._notifications.render(display)
        # self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.other_actor_id = 0 # init as 0 for static object
        self.wrong_collision = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))
        self.collision = False

    # def get_collision_history(self):
    #     history = collections.defaultdict(int)
    #     for frame, intensity in self.history:
    #         history[frame] += intensity
    #     return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        # impulse = event.normal_impulse
        # intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # dict: {data1, data2}
        # data = frame: {timestamp, other_actor's id, intensity}
        self.history.append({'frame': event.frame, 'actor_id': event.other_actor.id})
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        self.collision = True
        if event.other_actor.id != self.other_actor_id:
            self.wrong_collision = True
        

    def save_history(self, path):
        if self.collision:
            # for i, collision in enumerate(self.history):
            #     self.history[i] = list(self.history[i])
            # history = np.asarray(self.history)
            # if len(history) != 0:
            #     np.save('%s/collision_history' % (path), history)
            with open(os.path.join(path, 'collision_history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor, ego_data):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        self.recording = False
        self.ego_dict = ego_data
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        if self.recording:
            gnss = {'lat': event.latitude, 'lon': event.longitude}
            gnss_transform = {'x': event.transform.location.x, 'y': event.transform.location.y, 'z': event.transform.location.z,
                              'pitch': event.transform.rotation.pitch, 'yaw': event.transform.rotation.yaw, 'roll': event.transform.rotation.roll}

            if not event.frame in self.ego_dict:
                self.ego_dict[event.frame] = {}
            self.ego_dict[event.frame]['gnss'] = gnss
            self.ego_dict[event.frame]['gnss_transform'] = gnss_transform

    def toggle_recording_Gnss(self):
        self.recording = not self.recording

# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor, ego_data):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        self.frame = None # LBC
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.recording = False
        # self.imu_save = []
        self.ego_dict = ego_data
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
        self.frame = sensor_data.frame # LBC
        if self.recording:
            imu = {'accelerometer_x': self.accelerometer[0], 'accelerometer_y': self.accelerometer[1],
                   'accelerometer_z': self.accelerometer[2], 'gyroscope_x': self.gyroscope[0],
                   'gyroscope_y': self.gyroscope[1], 'gyroscope_z': self.gyroscope[2],
                   'compass': self.compass}
            # self.imu_save.append([sensor_data.frame,
            #                     self.accelerometer[0], self.accelerometer[1], self.accelerometer[2],
            #                     self.gyroscope[0], self.gyroscope[1], self.gyroscope[2],
            #                     self.compass])
            if not sensor_data.frame in self.ego_dict:
                self.ego_dict[sensor_data.frame] = {}
            self.ego_dict[sensor_data.frame]['imu'] = imu
            self.ego_dict[sensor_data.frame]['timestamp'] = sensor_data.timestamp
    def toggle_recording_IMU(self):
        self.recording = not self.recording
    #     if not self.recording:
    #         t_top = threading.Thread(target = self.save_IMU, args=(self.imu_save, path))
    #         t_top.start()
    #         self.imu_save = []

    # def save_IMU(self, save_list, path):
    #     np_imu = np.asarray(save_list)
    #     np.save('%s/imu' % (path), np_imu)
# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / \
                self.velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, save_mode):
        self.sensor_top = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        # LBC
        self.bev_map = None
        self.bev_map_frame = None
        self.actor_id_array = None
        self.ins_front_array = None
        self.front_image = None

        self.lbc_img = []
        self.top_img = []
        self.front_img = []
        self.left_img = []
        self.right_img = []
        self.back_img = []
        self.back_left_img = []
        self.back_right_img = []

        self.lidar = []
        self.flow = []
        self.dvs = []

        self.lbc_ins = []
        self.top_ins = []
        self.front_ins = []
        self.left_ins = []
        self.right_ins = []
        self.back_ins = []
        self.back_left_ins = []
        self.back_right_ins = []

        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.back_depth = []
        self.back_left_depth = []
        self.back_right_depth = []

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                # front view
                (carla.Transform(carla.Location(x=+0.8*bound_x,
                 y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                # front-left view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-55)), Attachment.Rigid),
                # front-right view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=55)), Attachment.Rigid),
                # back view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=180)), Attachment.Rigid),
                # back-left view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=235)), Attachment.Rigid),
                # back-right view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-235)), Attachment.Rigid),
                # top view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=23*bound_z), carla.Rotation(pitch=18.0)), Attachment.SpringArm),
                # LBC top view
                (carla.Transform(carla.Location(x=0, y=0,
                 z=100.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm)
            ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-5.5, z=2.5),
                 carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                 Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-8.0, z=6.0),
                 carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '85', 'rotation_frequency': '25'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', None, 'Optical Flow', {}],
            ['sensor.other.lane_invasion', None, 'Lane lane_invasion', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette,
                'Camera Instance Segmentation (CityScapes Palette)', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        self.bev_bp = bp_library.find('sensor.camera.rgb')
        self.bev_bp.set_attribute('image_size_x', str(512))
        self.bev_bp.set_attribute('image_size_y', str(512))
        self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))


        self.bev_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(512))
        self.bev_seg_bp.set_attribute('image_size_y', str(512))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        for item in self.sensors:
            
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor_top is not None:
                self.sensor_top.destroy()
                self.surface = None

            # rgb sensor
            self.sensor_lbc_img = self._parent.get_world().spawn_actor(
                self.bev_bp,
                self._camera_transforms[7][0],
                attach_to=self._parent)


            self.sensor_top = self._parent.get_world().spawn_actor(
                self.sensors[0][-1],
                self._camera_transforms[6][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[6][1])

            self.sensor_front = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

            self.ins_front = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
            if self.save_mode:
                self.sensor_front = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.sensor_left = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[1][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[1][1])
                self.sensor_right = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[2][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[2][1])
                self.sensor_back = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[3][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[3][1])
                self.sensor_back_left = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[4][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[4][1])
                self.sensor_back_right = self._parent.get_world().spawn_actor(
                    self.sensors[0][-1],
                    self._camera_transforms[5][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[5][1])

                # lidar sensor
                self.sensor_lidar = self._parent.get_world().spawn_actor(
                    self.sensors[6][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_dvs = self._parent.get_world().spawn_actor(
                    self.sensors[7][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                # optical flow
                self.sensor_flow = self._parent.get_world().spawn_actor(
                    self.sensors[8][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_lbc_ins = self._parent.get_world().spawn_actor(
                    self.bev_seg_bp,
                    self._camera_transforms[7][0],
                    attach_to=self._parent)

                self.ins_top = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[6][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[6][1])

                self.ins_front = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.ins_left = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[1][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[1][1])
                self.ins_right = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[2][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[2][1])
                self.ins_back = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[3][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[3][1])
                self.ins_back_left = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[4][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[4][1])
                self.ins_back_right = self._parent.get_world().spawn_actor(
                    self.sensors[10][-1],
                    self._camera_transforms[5][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[5][1])


                # depth estimation sensor
                self.depth_front = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.depth_left = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[1][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[1][1])
                self.depth_right = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[2][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[2][1])
                self.depth_back = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[3][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[3][1])
                self.depth_back_left = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[4][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[4][1])
                self.depth_back_right = self._parent.get_world().spawn_actor(
                    self.sensors[2][-1],
                    self._camera_transforms[5][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[5][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor_lbc_img.listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'lbc_img'))
            self.sensor_top.listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'top'))

            if self.save_mode:
                self.sensor_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'front'))
                self.sensor_right.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'right'))
                self.sensor_left.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'left'))
                self.sensor_back.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'back'))
                self.sensor_back_right.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'back_right'))
                self.sensor_back_left.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'back_left'))

                self.sensor_lidar.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'lidar'))
                self.sensor_dvs.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'dvs'))
                self.sensor_flow.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'flow'))


                self.sensor_lbc_ins.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'lbc_ins'))
                self.ins_top.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_top'))

                self.ins_front.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_front'))
                self.ins_right.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_right'))
                self.ins_left.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_left'))
                self.ins_back.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_back'))
                self.ins_back_right.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_back_right'))
                self.ins_back_left.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_back_left'))

                self.depth_front.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_front'))
                self.depth_right.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_right'))
                self.depth_left.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_left'))
                self.depth_back.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_back'))
                self.depth_back_right.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_back_right'))
                self.depth_back_left.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'depth_back_left'))

            if not self.save_mode: # for ego LBC
                self.sensor_lbc_ins = self._parent.get_world().spawn_actor(
                        self.bev_seg_bp,
                        self._camera_transforms[7][0],
                        attach_to=self._parent)
                self.sensor_lbc_ins.listen(lambda image: CameraManager._parse_image(
                        weak_self, image, 'lbc_ins_inf'))
                print('LBC sensor spawend!')

                self.sensor_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'front'))
                print('LBC sensor spawend!')
                self.ins_front.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'ins_front'))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self, path):
        self.recording = not self.recording
        if not self.recording:
            t_lbc_img = Process(
                target=self.save_img, args=(self.lbc_img, 0, path, 'lbc_img'))
            t_top = Process(
                target=self.save_img, args=(self.top_img, 0, path, 'top'))
            t_front = Process(target=self.save_img, args=(
                self.front_img, 0, path, 'front'))
            t_right = Process(target=self.save_img, args=(
                self.right_img, 0, path, 'right'))
            t_left = Process(
                target=self.save_img, args=(self.left_img, 0, path, 'left'))
            t_back = Process(
                target=self.save_img, args=(self.back_img, 0, path, 'back'))
            t_back_right = Process(target=self.save_img, args=(
                self.back_right_img, 0, path, 'back_right'))
            t_back_left = Process(target=self.save_img, args=(
                self.back_left_img, 0, path, 'back_left'))

            t_lidar = Process(
                target=self.save_img, args=(self.lidar, 6, path, 'lidar'))
            t_dvs = Process(
                target=self.save_img, args=(self.dvs, 7, path, 'dvs'))
            t_flow = Process(
                target=self.save_img, args=(self.flow, 8, path, 'flow'))

            t_lbc_ins = Process(
                target=self.save_img, args=(self.lbc_ins, 10, path, 'lbc_ins'))
            t_ins_top = Process(
                target=self.save_img, args=(self.top_ins, 10, path, 'ins_top'))

            t_ins_front = Process(
                target=self.save_img, args=(self.front_ins, 10, path, 'ins_front'))
            t_ins_right = Process(
                target=self.save_img, args=(self.right_ins, 10, path, 'ins_right'))
            t_ins_left = Process(
                target=self.save_img, args=(self.left_ins, 10, path, 'ins_left'))
            t_ins_back = Process(
                target=self.save_img, args=(self.back_ins, 10, path, 'ins_back'))
            t_ins_back_right = Process(
                target=self.save_img, args=(self.back_right_ins, 10, path, 'ins_back_right'))
            t_ins_back_left = Process(
                target=self.save_img, args=(self.back_left_ins, 10, path, 'ins_back_left'))

            t_depth_front = Process(target=self.save_img, args=(
                self.front_depth, 1, path, 'depth_front'))
            t_depth_right = Process(target=self.save_img, args=(
                self.right_depth, 1, path, 'depth_right'))
            t_depth_left = Process(target=self.save_img, args=(
                self.left_depth, 1, path, 'depth_left'))
            t_depth_back = Process(target=self.save_img, args=(
                self.back_depth, 1, path, 'depth_back'))
            t_depth_back_right = Process(target=self.save_img, args=(
                self.back_right_depth, 1, path, 'depth_back_right'))
            t_depth_back_left = Process(target=self.save_img, args=(
                self.back_left_depth, 1, path, 'depth_back_left'))

            start_time = time.time()
            t_lbc_img.start()
            t_top.start()
            t_front.start()
            t_left.start()
            t_right.start()
            t_back.start()
            t_back_left.start()
            t_back_right.start()
            t_lidar.start()
            t_dvs.start()
            t_flow.start()
            t_lbc_ins.start()
            t_ins_top.start()
            t_ins_front.start()
            t_ins_right.start()
            t_ins_left.start()
            t_ins_back.start()
            t_ins_back_right.start()
            t_ins_back_left.start()

            t_depth_front.start()
            t_depth_right.start()
            t_depth_left.start()
            t_depth_back.start()
            t_depth_back_right.start()
            t_depth_back_left.start()

            self.top_img = []
            self.lbc_img = []
            self.lbc_ins = []

            self.front_img = []
            self.right_img = []
            self.left_img = []
            self.back_img = []
            self.back_right_img = []
            self.back_left_img = []

            self.lidar = []
            self.dvs = []
            self.flow = []


            self.front_depth = []
            self.right_depth = []
            self.left_depth = []
            self.back_depth = []
            self.back_right_depth = []
            self.back_left_depth = []

            t_depth_front.join()

            self.top_img = []
            self.top_ins = []

            t_lbc_img.join()
            t_top.join()
            t_front.join()
            t_left.join()
            t_right.join()
            t_back.join()
            t_back_left.join()
            t_back_right.join()

            t_lidar.join()
            t_dvs.join()
            t_flow.join()
            
            t_lbc_ins.join()
            t_ins_top.join()


            t_ins_front.join()
            t_ins_right.join()
            t_ins_left.join()
            t_ins_back.join()
            t_ins_back_right.join()
            t_ins_back_left.join()

            t_depth_front.join()
            t_depth_right.join()
            t_depth_left.join()
            t_depth_back.join()
            t_depth_back_right.join()
            t_depth_back_left.join()
            end_time = time.time()
            print('sensor data save done in %s' % (end_time-start_time))
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def save_img(self, img_list, sensor, path, view='top'):
        modality = self.sensors[sensor][0].split('.')[-1]
        for img in img_list:
            if img.frame % 1 == 0:
                if 'seg' in view:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, img.frame), cc.CityScapesPalette)
                elif 'depth' in view:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, img.frame), cc.LogarithmicDepth)
                elif 'dvs' in view:
                    dvs_events = np.frombuffer(img.raw_data, dtype=np.dtype([
                        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))
                    dvs_img = np.zeros(
                        (img.height, img.width, 3), dtype=np.uint8)
                    # Blue is positive, red is negative
                    dvs_img[dvs_events[:]['y'], dvs_events[:]
                            ['x'], dvs_events[:]['pol'] * 2] = 255
                    # img = img.to_image()
                    stored_path = os.path.join(path, modality, view)
                    if not os.path.exists(stored_path):
                        os.makedirs(stored_path)
                    np.save('%s/%08d' % (stored_path, img.frame), dvs_img)
                elif 'flow' in view:
                    frame = img.frame
                    img = img.get_color_coded_flow()
                    array = np.frombuffer(
                        img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    array = array[:, :, :3]
                    array = array[:, :, ::-1]
                    stored_path = os.path.join(path, modality, view)
                    if not os.path.exists(stored_path):
                        os.makedirs(stored_path)
                    np.save('%s/%08d' % (stored_path, frame), array)
                else:
                    img.save_to_disk('%s/%s/%s/%08d' %
                                     (path, modality, view, img.frame))
        print("%s %s save finished." % (self.sensors[sensor][2], view))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='top'):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))

        elif view == 'top':
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        elif view == 'front':
            self.front_image = image
        elif view == 'ins_front':
            self.ins_front_array = image
            ins_front_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            self.ins_front_array = np.reshape(ins_front_array, (image.height, image.width, 4))
            
        elif view == 'lbc_ins_inf':
            # print("lbc ins inf")
            def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
                result = list()

                transform = vehicle.get_transform()
                pos = transform.location
                theta = np.radians(90 + transform.rotation.yaw)
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)],
                    ])

                for light in lights:
                    delta = light.get_transform().location - pos

                    target = R.T.dot([delta.x, delta.y])
                    target *= pixels_per_meter
                    target += size // 2

                    if min(target) < 0 or max(target) >= size:
                        continue

                    trigger = light.trigger_volume
                    light.get_transform().transform(trigger.location)
                    dist = trigger.location.distance(vehicle.get_location())
                    a = np.sqrt(
                            trigger.extent.x ** 2 +
                            trigger.extent.y ** 2 +
                            trigger.extent.z ** 2)
                    b = np.sqrt(
                            vehicle.bounding_box.extent.x ** 2 +
                            vehicle.bounding_box.extent.y ** 2 +
                            vehicle.bounding_box.extent.z ** 2)

                    if dist > a + b:
                        continue

                    result.append(light)

                return result
            def draw_traffic_lights(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
                from PIL import Image, ImageDraw
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)

                transform = vehicle.get_transform()
                pos = transform.location
                theta = np.radians(90 + transform.rotation.yaw)
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)],
                    ])

                for light in lights:
                    delta = light.get_transform().location - pos

                    target = R.T.dot([delta.x, delta.y])
                    target *= pixels_per_meter
                    target += size // 2

                    if min(target) < 0 or max(target) >= size:
                        continue

                    trigger = light.trigger_volume
                    light.get_transform().transform(trigger.location)
                    dist = trigger.location.distance(vehicle.get_location())
                    a = np.sqrt(
                            trigger.extent.x ** 2 +
                            trigger.extent.y ** 2 +
                            trigger.extent.z ** 2)
                    b = np.sqrt(
                            vehicle.bounding_box.extent.x ** 2 +
                            vehicle.bounding_box.extent.y ** 2 +
                            vehicle.bounding_box.extent.z ** 2)

                    if dist > a + b:
                        continue

                    x, y = target
                    draw.ellipse(
                            (x-radius, y-radius, x+radius, y+radius),
                            23 + light.state.real)

                return np.array(image)
            '''
            Instance Variables: image
                fov (float - degrees): Horizontal field of view of the image.
                height (int): Image height in pixels.
                width (int): Image width in pixels.
                raw_data (bytes)
            '''

            # take out the raw data
            array_raw = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            # print("array.shape", array.shape) # array.shape (1048576,)
            array_raw = np.reshape(array_raw, (image.height, image.width, 4))
            actors = self._parent.get_world().get_actors()
            traffic_lights = get_nearby_lights(self._parent, actors.filter('*traffic_light*'))
            # np.save("./raw_topdown.npy", array_raw)

            b_array = array_raw[:, :, 0]
            g_array = array_raw[:, :, 1]
            self.actor_id_array = b_array*256 + g_array
            
            self.bev_map = copy.deepcopy(array_raw[:, :, 2])
            self.bev_map_frame = image.frame

            

            #print(b_array*256 + g_array)

        if self.recording and image.frame % 1 == 0:
            # print(view,image.frame)
            if view == 'top':
                self.top_img.append(image)
            elif view == 'lbc_img':
                self.lbc_img.append(image)
            elif view == 'front':
                self.front_img.append(image)
            elif view == 'left':
                self.left_img.append(image)
            elif view == 'right':
                self.right_img.append(image)
            elif view == 'back':
                self.back_img.append(image)
            elif view == 'back_left':
                self.back_left_img.append(image)
            elif view == 'back_right':
                self.back_right_img.append(image)
            elif view == 'lidar':
                self.lidar.append(image)
            elif view == 'dvs':
                self.dvs.append(image)
            elif view == 'flow':
                self.flow.append(image)
            elif view == 'lbc_ins':
                self.lbc_ins.append(image)
            elif view == 'ins_top':
                self.top_ins.append(image)
            elif view == 'seg_front':
                self.front_seg.append(image)
            # elif view == 'seg_right':
            #     self.right_seg.append(image)
            # elif view == 'seg_left':
            #     self.left_seg.append(image)
            # elif view == 'seg_back':
            #     self.back_seg.append(image)
            # elif view == 'seg_back_right':
            #     self.back_right_seg.append(image)
            # elif view == 'seg_back_left':
            #     self.back_left_seg.append(image)
            elif view == 'ins_front':
                self.front_ins.append(image)
            elif view == 'ins_right':
                self.right_ins.append(image)
            elif view == 'ins_left':
                self.left_ins.append(image)
            elif view == 'ins_back':
                self.back_ins.append(image)
            elif view == 'ins_back_right':
                self.back_right_ins.append(image)
            elif view == 'ins_back_left':
                self.back_left_ins.append(image)

            elif view == 'depth_front':
                self.front_depth.append(image)
            elif view == 'depth_right':
                self.right_depth.append(image)
            elif view == 'depth_left':
                self.left_depth.append(image)
            elif view == 'depth_back':
                self.back_depth.append(image)
            elif view == 'depth_back_right':
                self.back_right_depth.append(image)
            elif view == 'depth_back_left':
                self.back_left_depth.append(image)


def record_control(control, control_list):
    np_control = np.zeros(7)
    np_control[0] = control.throttle
    np_control[1] = control.steer
    np_control[2] = control.brake
    np_control[3] = control.hand_brake
    np_control[4] = control.reverse
    np_control[5] = control.manual_gear_shift
    np_control[6] = control.gear

    control_list.append(np_control)


def control_with_trasform_controller(controller, transform):
    control_signal = controller.run_step(10, transform)
    return control_signal

def collect_trajectory(get_world, agent, scenario_id, period_end, stored_path, clock):
    if not os.path.exists(stored_path + '/trajectory/'):
        os.mkdir(stored_path + '/trajectory/')
    filepath = stored_path + '/trajectory/' + str(scenario_id) + '.csv'
    is_exist = os.path.isfile(filepath)
    f = open(filepath, 'w')
    w = csv.writer(f)

    filepath_all = stored_path + '/trajectory/' + str(scenario_id) + '_all.csv'
    is_exist_all = os.path.isfile(filepath_all)
    f_all = open(filepath_all, 'w')
    w_all = csv.writer(f_all)

    if not is_exist:
        w.writerow(['TIMESTAMP', 'TRACK_ID',
                   'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])

    if not is_exist_all:
        w_all.writerow(['TIMESTAMP', 'TRACK_ID',
                   'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])

    actors = get_world.world.get_actors()
    town_map = get_world.world.get_map()
    agent_id = agent.id
    period_start = 0
    fps = clock.get_fps()
    record_time = 0
    time_start = time.time()
    try:
        while True:
            time_end = time.time()
            if get_world.abandon_scenario:
                print('Abandom, killing thread.')
                return
            elif get_world.finish:
                print("trajectory collection finish")
                return 
            # 25: the landing iter
            # 0.1s = 0.05000000074505806s * 2
            if (time_end - time_start) > 2/fps:
                period_start += 0.1 * fps
                record_time += 0.1
                time_start = time.time()
                for actor in actors:
                    if agent_id == actor.id:
                        agent = actor
                    if agent.get_location().x == 0 and agent.get_location().y == 0:
                        return True
                    if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
                        x = actor.get_location().x
                        y = actor.get_location().y
                        id = actor.id
                        if x == agent.get_location().x and y == agent.get_location().y:
                            w.writerow(
                                [record_time - 0.1, id, 'AGENT', str(x), str(y), town_map.name.split('/')[2]])
                            w_all.writerow(
                                [record_time - 0.1, id, 'AGENT', str(x), str(y), town_map.name.split('/')[2]])
                        else:
                            if actor.type_id[0:7] == 'vehicle':
                                w_all.writerow(
                                    [record_time - 0.1, id, 'vehicle', str(x), str(y), town_map.name.split('/')[2]])
                            if ((x - agent.get_location().x)**2 + (y - agent.get_location().y)**2) < 75**2:
                                if actor.type_id[0:7] == 'vehicle':
                                    w.writerow(
                                        [record_time - 0.1, id, 'vehicle', str(x), str(y), town_map.name.split('/')[2]])
                                elif actor.type_id[0:6] == 'walker':
                                    w.writerow(
                                        [record_time - 0.1, id, 'walker', str(x), str(y), town_map.name.split('/')[2]])
    except:
        print("trajectory collection error")


def collect_topology(get_world, agent, scenario_id, t, root, stored_path, clock):
    town_map = get_world.world.get_map()
    if not os.path.exists(stored_path + '/topology/'):
        os.mkdir(stored_path + '/topology/')
    #with open(root + '/scenario_description.json') as f:
    #    data = json.load(f)
    time_start = time.time()
    fps = clock.get_fps()
    try:
        while True:
            if get_world.abandon_scenario:
                print('Abandom, killing thread.')
                return
            time_end = time.time()
            if (time_end - time_start) > t * fps:         # t may need change
                waypoint = town_map.get_waypoint(agent.get_location())
                waypoint_list = town_map.generate_waypoints(2.0)
                nearby_waypoint = []
                roads = []
                all = []
                for wp in waypoint_list:
                    dist_x = int(wp.transform.location.x) - \
                        int(waypoint.transform.location.x)
                    dist_y = int(wp.transform.location.y) - \
                        int(waypoint.transform.location.y)
                    if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                        nearby_waypoint.append(wp)
                        roads.append((wp.road_id, wp.lane_id))
                for wp in nearby_waypoint:
                    for id in roads:
                        if wp.road_id == id[0] and wp.lane_id == id[1]:
                            all.append(((wp.road_id, wp.lane_id), wp))
                            break
                all = sorted(all, key=lambda s: s[0][1])
                temp_d = {}
                d = {}
                for (i, j), wp in all:
                    if (i, j) in temp_d:
                        temp_d[(i, j)] += 1
                    else:
                        temp_d[(i, j)] = 1
                for (i, j) in temp_d:
                    if temp_d[(i, j)] != 1:
                        d[(i, j)] = temp_d[(i, j)]
                rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
                lane_feature_ls = []
                for i, j in d:
                    halluc_lane_1, halluc_lane_2 = np.empty(
                        (0, 3*2)), np.empty((0, 3*2))
                    center_lane = np.empty((0, 3*2))
                    is_traffic_control = False
                    is_junction = False
                    turn_direction = None
                    for k in range(len(all)-1):
                        if (i, j) == all[k][0] and (i, j) == all[k+1][0]:
                            # may change & need traffic light
                            if all[k][1].get_landmarks(50, False):
                                is_traffic_control = True
                            if all[k][1].is_junction:
                                is_junction = True
                            # -= norm center
                            before = [all[k][1].transform.location.x,
                                      all[k][1].transform.location.y]
                            after = [all[k+1][1].transform.location.x,
                                     all[k+1][1].transform.location.y]
                            # transform.rotation.yaw can not be overwritten
                            before_yaw = all[k][1].transform.rotation.yaw
                            after_yaw = all[k+1][1].transform.rotation.yaw
                            if (before_yaw < -360.0):
                                before_yaw = before_yaw + 360.0
                            if (after_yaw < -360.0):
                                after_yaw = after_yaw + 360.0
                            if (after_yaw > before_yaw):
                                turn_direction = "right" # right
                            elif (after_yaw < before_yaw):
                                turn_direction = "left" # left
                            distance = []
                            for t in range(len(before)):
                                distance.append(after[t] - before[t])
                            np_distance = np.array(distance)
                            norm = np.linalg.norm(np_distance)
                            e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
                            lane_1 = np.hstack((before + e1 * all[k][1].lane_width/2, all[k][1].transform.location.z,
                                               after + e1 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            lane_2 = np.hstack((before + e2 * all[k][1].lane_width/2, all[k][1].transform.location.z,
                                               after + e2 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            lane_c = np.hstack((before, all[k][1].transform.location.z,
                                               after, all[k+1][1].transform.location.z))
                            halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
                            halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
                            center_lane = np.vstack((center_lane, lane_c))
                    #if data['traffic_light']:
                    #    is_junction = True
                    lane_feature_ls.append(
                        [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (i, j)])
                np.save(stored_path + '/topology/' + str(scenario_id),
                        np.array(lane_feature_ls))

                for features in lane_feature_ls:
                    xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
                        :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
                    plt.plot(xs, ys, '--', color='gray')
                    x_s, y_s = np.vstack((features[1][:, :2], features[1][-1, 3:5]))[
                        :, 0], np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 1]
                    plt.plot(x_s, y_s, '--', color='gray')

                plt.savefig(stored_path + '/topology/topology.png')
                break
        print("topology collection finished")
        return 
    except:
        print("topology collection error.")

def set_bp(blueprint, actor_id):
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


def save_description(world, args, stored_path, weather,agents_dict, nearest_obstacle):
    vehicles = world.world.get_actors().filter('vehicle.*')
    peds = world.world.get_actors().filter('walker.*')
    d = dict()
    d['num_actor'] = len(vehicles) + len(peds)
    d['num_vehicle'] = len(vehicles)
    d['weather'] = str(weather)
    # d['random_objects'] = args.random_objects
    d['random_actors'] = args.random_actors
    d['simulation_time'] = int(world.hud.simulation_time)
    d['nearest_obstacle'] = nearest_obstacle
    
    for key in agents_dict:
        d[key] = agents_dict[key].id

    with open('%s/dynamic_description.json' % (stored_path), 'w') as f:
        json.dump(d, f)


def write_actor_list(world,stored_path):

    def write_row(writer,actors,filter_str,class_id,min_id,max_id):
        filter_actors = actors.filter(filter_str)
        for actor in filter_actors:
            if actor.id < min_id:
                min_id = actor.id
            if actor.id > max_id:
                max_id = actor.id
            writer.writerow([actor.id,class_id,actor.type_id])
        return min_id,max_id
    
    filter_ = ['walker.*','vehicle.*','static.prop.streetbarrier*',
            'static.prop.trafficcone*','static.prop.trafficwarning*']
    id_ = [4,10,20,20,20]
    actors = world.world.get_actors()
    min_id = int(1e7)
    max_id = int(0)

    return min_id,max_id

def generate_obstacle(world, bp, path, ego_transform):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    min_dis = float('Inf')
    nearest_obstacle = -1
    
    if lines[0][:6] == 'static':
        for line in lines:
            obstacle_name = line.split('\t')[0]
            transform = line.split('\t')[1]
            # print(obstacle_name, " ", transform)
            # exec("obstacle_actor = world.spawn_actor(bp.filter(obstacle_name)[0], %s)" % transform)

            x = float(transform.split('x=')[1].split(',')[0])
            y = float(transform.split('y=')[1].split(',')[0])
            z = float(transform.split('z=')[1].split(')')[0])
            pitch = float(transform.split('pitch=')[1].split(',')[0])
            yaw = float(transform.split('yaw=')[1].split(',')[0])
            roll = float(transform.split('roll=')[1].split(')')[0])

            obstacle_loc = carla.Location(x, y, z)
            obstacle_rot = carla.Rotation(pitch, yaw, roll)
            obstacle_trans = carla.Transform(obstacle_loc, obstacle_rot)

            obstacle_actor = world.spawn_actor(bp.filter(obstacle_name)[0], obstacle_trans)

            dis = ego_transform.location.distance(obstacle_loc)
            if dis < min_dis:
                nearest_obstacle = obstacle_actor.id
                min_dis = dis

    return nearest_obstacle



def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(masks[index] != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def produce_bbx(mask, agents_dict, frame, front_image, threshold=60):

    # produce the bbox and save the image

    image = cv2.cvtColor(np.asarray(front_image),cv2.COLOR_RGB2BGR)  
    cv2.imwrite(f'./roi_two_stage/inference/test_data/rgb/front/{frame:08d}.png', image)
    # cv2.imwrite(f'./test.jpeg', image)

    # ID: b*256+g 
    # class : r 
    
    h,w = mask.shape[1:]
    mask_2 = torch.zeros(2,h,w)
    mask_2[0] = mask[2]
    mask_2[1] = mask[1] + mask[0]*256

    # ped,vehicle
    condition = mask_2[0]== 4
    condition += mask_2[0]== 10
    obj_ids = torch.unique(mask_2[1,condition])
    filter_exist_actor = []
    for obj_id in obj_ids:
        obj_id = int(obj_id.numpy())
        if obj_id in agents_dict.keys():
            filter_exist_actor.append(True)
        else:
            filter_exist_actor.append(False)

    obj_ids = obj_ids[filter_exist_actor]

    masks = mask_2[1] == obj_ids[:, None, None]
    masks = masks*condition
    
    area_condition = masks.long().sum((1,2))>=threshold

    masks = masks[area_condition]
    obj_ids = obj_ids[area_condition].type(torch.int).numpy()
    boxes = masks_to_boxes(masks).type(torch.int16).numpy()
    out_list = []
    for id,box in zip(obj_ids,boxes):
        out_list.append({'actor_id':int(id), 'class':int(agents_dict[id]), 'box':box.tolist()})

        # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
    with open(f'./roi_two_stage/inference/test_data/bbox/front/{frame:08d}.json', 'w') as f:
        json.dump(out_list, f)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    path = os.path.join('data_collection', args.scenario_type, args.scenario_id)

    if not os.path.exists("./test_result/out_video") :
            os.makedirs("./test_result/out_video")
    if not os.path.exists("./test_result/top_video") :
            os.makedirs("./test_result/top_video")
            
    out = cv2.VideoWriter(f"./test_result/out_video/{args.scenario_id}_{args.weather}_{args.random_actors}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (1280, 720) )

    out_topview = cv2.VideoWriter(f"./test_result/top_video/{args.scenario_id}_{args.weather}_{args.random_actors}_topview.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (256, 256) )

    output_name = f"./test_result/top_video/{args.scenario_id}_{args.weather}_{args.random_actors}_topview.mp4"



    random.seed(int(args.random_seed))
    seeds = []
    for _ in range(10):
        seeds.append(random.randint(1565169134, 2665169134))
    # print(seeds)



    filter_dict = {}
    try:
        for root, _, files in os.walk(path + '/filter/'):
            for name in files:
                f = open(path + '/filter/' + name, 'r')
                bp = f.readlines()[0]
                name = name.strip('.txt')
                f.close()
                filter_dict[name] = bp
        print(filter_dict)
    except:
        print("檔案夾不存在。")

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
    is_collision = False 





    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height, client.get_world(), args)

        weather = args.weather
        exec("args.weather = carla.WeatherParameters.%s" % args.weather)
        stored_path = os.path.join('data_collection', args.scenario_type, args.scenario_id, weather + "_" + args.random_actors + "_")
        print(stored_path)
        if not os.path.exists(stored_path) :
            os.makedirs(stored_path)
        world = World(client.load_world(args.map),
                      filter_dict['player'], hud, args, stored_path, seeds)
        client.get_world().set_weather(args.weather)

        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        world.world.apply_settings(settings)

        # LBC get frame ####################
        frame = None
        topdown = None
        N_CLASSES = len(common.COLOR)

        frame = world.world.tick()
        ####################################

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
            nearest_obstacle = generate_obstacle(client.get_world(), blueprint_library,
                              path+"/obstacle/obstacle_list.txt", ego_transform)


        # set controller
        for actor_id, bp in filter_dict.items():
            if actor_id != 'player':
                transform_spawn = transform_dict[actor_id][0]
                
                while True:
                    try:
                        agents_dict[actor_id] = client.get_world().spawn_actor(
                            set_bp(blueprint_library.filter(
                                filter_dict[actor_id]), actor_id),
                            transform_spawn)

                        if args.scenario_type == 'obstacle':
                            dis = ego_transform.location.distance(transform_spawn.location)
                            if dis < min_dis:
                                nearest_obstacle = actor_id
                                min_dis = dis

                        break
                    except Exception:
                        transform_spawn.location.z += 1.5

                # set other actor id for checking collision object's identity
                world.collision_sensor.other_actor_id = agents_dict[actor_id].id

            if 'vehicle' in bp:
                controller_dict[actor_id] = VehiclePIDController(agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                                    max_throttle=1.0, max_brake=1.0, max_steering=1.0)
                try:
                    agents_dict[actor_id].set_light_state(carla.VehicleLightState.LowBeam)
                except:
                    print('vehicle has no low beam light')
            actor_transform_index[actor_id] = 1
            finish[actor_id] = False

        # waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)

        root = os.path.join('data_collection', args.scenario_type, args.scenario_id)
        scenario_name = str(weather) + '_'

        vehicles_list = []
        all_id = None 
        if args.random_actors != 'none':
            if args.random_actors == 'pedestrian':  #only pedestrian
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, seeds, distance=100, v_ratio=0.0,
                                   pedestrian=40 , transform_dict=transform_dict)
            elif args.random_actors == 'low':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, seeds, distance=100, v_ratio=0.3,
                                   pedestrian=20 , transform_dict=transform_dict)
            elif args.random_actors == 'mid':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, seeds, distance=100, v_ratio=0.6,
                                   pedestrian=40, transform_dict=transform_dict)
            elif args.random_actors == 'high':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, seeds, distance=100, v_ratio=0.8,
                                   pedestrian=80, transform_dict=transform_dict)
        scenario_name = scenario_name + args.random_actors + '_'

        if not args.no_save:
            # recording traj
            id = []
            moment = []
            print(root)
            with open(os.path.join(root, 'timestamp.txt')) as f:
                for line in f.readlines():
                    s = line.split(',')
                    id.append(int(s[0]))
                    moment.append(s[1])
            period = float(moment[-1]) - float(moment[0])
            half_period = period / 2
        # dynamic scenario setting
        # stored_path = os.path.join(root, scenario_name)
        # print(stored_path)
        # if not os.path.exists(stored_path) and not args.no_save:
        #     os.makedirs(stored_path)
        if args.save_rss:
            print(world.rss_sensor)
            world.rss_sensor.stored_path = stored_path
        # write actor list
        min_id, max_id = write_actor_list(world,stored_path)
        if max_id-min_id>=65535:
            print('Actor id error. Abandom.')
            abandon_scenario = True
            raise 

        iter_tick = 0
        iter_start = 15
        iter_toggle = 50
        # # write trajector
        # if not os.path.exists( './trajectory_frame/'):
        #     os.mkdir('./trajectory_frame/')
        # filepath = stored_path + '/trajectory_frame/' + str(args.scenario_id) + '.csv'
        # is_exist = os.path.isfile('/trajectory_frame.csv')





        # ==============================================================================
        # -- LBC init. -----------------------------------------------------------------
        # ==============================================================================

        # get waypoints
        map = world.world.get_map()
        ego_index = 0
        def find_next_target(max_distance, wp_now):
            wp_last = None
            wp_next = wp_now.next(2)[0] # next waypoint in 2 meters
            # print(wp_next, len(wp_next))
            while wp_now.transform.location.distance(wp_next.transform.location) < max_distance:
                # print(wp_now.transform.location.distance(wp_next.transform.location))
                wp_last = wp_next
                wp_next = wp_next.next(2)[0]
                # if len(wp_next) > 1:
                #     wp_next = wp_next[0]
            # print(wp_now.transform.location.distance(wp_next.transform.location))
            return wp_last

        def find_next_target_index(max_distance, location_now, transform_list, index):
             if index == len(transform_list) - 1:
                 return index

             while location_now.distance(transform_list[index].location) < max_distance and index < len(transform_list):
                 index += 1
                 if index == len(transform_list):
                     break

             return index - 1
            
        min_distance = 7.5 
        max_distance = 25.0
        ego_transform_list = transform_dict['player']
        ego_loc_now = world.player.get_location()
        ego_index = find_next_target_index(max_distance, ego_loc_now, ego_transform_list, ego_index)

        DEBUG = False


        class PIDController(object):
            def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
                self._K_P = K_P
                self._K_I = K_I
                self._K_D = K_D

                self._window = deque([0 for _ in range(n)], maxlen=n)
                self._max = 1.0
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

                if DEBUG:
                    import cv2

                    canvas = np.ones((100, 100, 3), dtype=np.uint8)
                    w = int(canvas.shape[1] / len(self._window))
                    h = 99

                    for i in range(1, len(self._window)):
                        y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
                        y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

                        cv2.line(
                                canvas,
                                ((i-1) * w, int(y1 * h)),
                                ((i) * w, int(y2 * h)),
                                (255, 255, 255), 2)

                    canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

                    cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
                    cv2.waitKey(1)

                return self._K_P * error + self._K_I * integral + self._K_D * derivative

        ego_turn_controller = PIDController(K_P=1, K_I=0, K_D=0.0, n=40)
        ego_speed_controller = PIDController(K_P=1, K_I=0, K_D=0.0, n=40)
        

        ego_converter = Converter()
        
        if not os.path.exists("./model_weight/LBC/"):
            os.mkdir("./model_weight/LBC/")

        if not os.path.isfile('./model_weight/LBC/epoch=29.ckpt'):
            print("Download LBC (interactive) weight")
            url = "https://drive.google.com/u/4/uc?id=16UZLRNRocmjLBSO6TxcbfOI0zHEE5HGs&export=download"
            output = './model_weight/LBC/epoch=29.ckpt'
            gdown.download(url, output)

        net = MapModel.load_from_checkpoint('./model_weight/LBC/epoch=29.ckpt')
        net.cuda()
        net.eval()

        start_frame = int(args.start_frame) # 50 

        flag_for_store_actor_list = 1
        actor_list_and_position = {}

        P = stored_path.split("/")
               

        # =======================================

        
        df_list = []


        distance_counter = 0
        total_distance = 0.0
        avg_distance = 0.0
        min_distance_with_gt = 100000
        
        target_loc = ego_transform_list[-1].location

        if args.input_target:
            target_loc.x = float(args.target_x)
            target_loc.y = float(args.target_y)
            # print("input target point")
            
        target_xy = np.float32([target_loc.x, target_loc.y])




        # =======================================
        from attrdict import AttrDict
        from social_gan import social_gan_one_step
        from social_gan.sgan.models import TrajectoryGenerator
        
        if not os.path.exists("./social_gan/"):
            os.mkdir("./social_gan/")

        if not os.path.isfile('./social_gan/gan_test_with_model_all.pt'):
            print("Download Socal Gan weight")
            url = "https://drive.google.com/u/0/uc?id=1R6Vjac0HZBQi3PFfrfU-fzjUQYX7RIHm&export=download"
            output = './social_gan/gan_test_with_model_all.pt'
            gdown.download(url, output)

        checkpoint = torch.load("social_gan/gan_test_with_model_all.pt")
        args_sg = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args_sg.obs_len,
            pred_len=60 ,#new_args.pred_len,
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
        generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        _args = AttrDict(checkpoint['args'])
        _args.dataset_name = "interactive" 
        _args.skip = 1
        _args.pred_len = 60 

        while (1):
            if frame > start_frame+ 120 :
                abandon_scenario = True
                is_collision = False

            clock.tick_busy_loop(40)
            frame = world.world.tick()
            print('frame:', frame)
            hud.frame = frame
            iter_tick += 1
            if iter_tick == iter_start + 1:
                ref_light = get_next_traffic_light(
                    world.player, world.world, light_transform_dict)
                annotate = annotate_trafficlight_in_group(
                    ref_light, lights, world.world)
            
            elif iter_tick > iter_start:
                actors = world.world.get_actors()
                # print(actors) 
                #print(" =========================================================== ")
               #print(filter_dict["player"])



                ## get interactor id 
                keys = list( agents_dict.keys())
                keys.remove('player')

                ego_car_id = world.player.id
                gt_interactor_id = int(keys[0])




                for actor in actors:
                    if actor_transform_index['player'] < len(transform_dict[actor_id]):
                        x = actor.get_location().x
                        y = actor.get_location().y
                        id = actor.id
                        if "vehicle" in actor.type_id:
                            if id == ego_car_id:
                                #w.writerow( [frame, id, 'EGO', str(x), str(y), args.map])
                                df_list.append([frame, id, 'EGO', str(x), str(y), args.map])
                            elif id == gt_interactor_id:
                                # w.writerow( [frame, id, 'ACTOR', str(x), str(y), args.map])
                                df_list.append([frame, id, 'ACTOR', str(x), str(y), args.map])
                            else:
                                # w.writerow( [frame, id, 'vehicle', str(x), str(y), args.map])
                                df_list.append([frame, id, 'vehicle', str(x), str(y), args.map])
                                #print(f"write {frame}")

                        elif "walker" in actor.type_id:
                            if id == gt_interactor_id:
                                # w.writerow( [frame, id, 'ACTOR', str(x), str(y), args.map])
                                df_list.append(  [frame, id, 'ACTOR', str(x), str(y), args.map] )
                            else:
                                # w.writerow( [frame, id, 'pedestrian', str(x), str(y), args.map])
                                df_list.append(  [frame, id, 'pedestrian', str(x), str(y), args.map])

                                

                for actor in actors:
                    if actor_transform_index['player'] < len(transform_dict[actor_id]):
                        if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
                            x = actor.get_location().x
                            y = actor.get_location().y
                            id = actor.id
                            if actor.type_id[0:7] == 'vehicle':
                                # w.writerow(
                                #         [frame, id, 'vehicle', str(x), str(y), args.map])
                                if flag_for_store_actor_list == 1:
                                    actor_list_and_position.update({id: 10}) # 'vehicle'
                                
                            elif actor.type_id[0:6] == 'walker':
                                # w.writerow(
                                #         [frame, id, 'pedestrian', str(x), str(y), args.map])
                                if flag_for_store_actor_list == 1:
                                    actor_list_and_position.update({id: 4}) # pedestrian
                flag_for_store_actor_list = 0





                                
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
                            if actor_id != 'player':

                                target_speed = (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6
                                waypoint = transform_dict[actor_id][actor_transform_index[actor_id]]
                                agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(target_speed, waypoint))                            
                                # agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                                #     (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6, transform_dict[actor_id][actor_transform_index[actor_id]]))

                                v = agents_dict[actor_id].get_velocity()
                                v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                                # to avoid the actor slowing down for the dense location around
                                # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2 + v/20.0:
                                if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                                    actor_transform_index[actor_id] += 2
                                elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                                    actor_transform_index[actor_id] += 6
                                else:
                                    actor_transform_index[actor_id] += 1

                            elif actor_id == 'player' and frame < start_frame :

                                target_speed = (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6
                                waypoint = transform_dict[actor_id][actor_transform_index[actor_id]]
                                agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(target_speed, waypoint))                            
                                # agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                                #     (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6, transform_dict[actor_id][actor_transform_index[actor_id]]))

                                v = agents_dict[actor_id].get_velocity()
                                v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                                # to avoid the actor slowing down for the dense location around
                                # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2 + v/20.0:
                                if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                                    actor_transform_index[actor_id] += 2
                                elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                                    actor_transform_index[actor_id] += 6
                                else:
                                    actor_transform_index[actor_id] += 1


                            gt_actor_location = agents_dict[keys[0]].get_transform().location

             

                        elif 'pedestrian' in filter_dict[actor_id]:
                            agents_dict[actor_id].apply_control(
                                ped_control_dict[actor_id][actor_transform_index[actor_id]])
                            actor_transform_index[actor_id] += 1

                    else:
                        finish[actor_id] = True



                    
                    # ==============================================================================
                    # -- LBC inference and control -------------------------------------------------
                    # ==============================================================================
                    if actor_id == 'player':
                        # get input

                        if True:
                            ego_loc_now = world.player.get_location()
                            if ego_loc_now.distance(ego_transform_list[ego_index].location) < min_distance:
                                 ego_index = find_next_target_index(max_distance, ego_loc_now, ego_transform_list, ego_index)
                            
                            # wait for bev map
                            while True:    

                                if world.camera_manager.bev_map_frame == frame:
                                    topdown = world.camera_manager.bev_map
                                    actor_id_array = world.camera_manager.actor_id_array
                                    front_img_input = world.camera_manager.front_image
                                    ins_front_array = world.camera_manager.ins_front_array ## b, g, r
                                    break
                            while True:    
                                if world.imu_sensor.frame == frame:
                                    theta = world.imu_sensor.compass
                                    break

                            front_img_array = np.frombuffer(front_img_input.raw_data, dtype=np.dtype("uint8"))
                            front_img_array = np.reshape(front_img_array, (front_img_input.height, front_img_input.width, 4))

                            front_img_array = front_img_array[:, :, :3]
                            front_img_array = front_img_array[:, :, ::-1]

                            front_b_array = ins_front_array[:, :, 0]
                            front_g_array = ins_front_array[:, :, 1]
                            front_id_array = front_b_array*256 + front_g_array
                            front_class_array = ins_front_array[:, :, 2]

                            front_index_of_ped = np.argwhere(front_class_array==4 )
                            front_index_of_car = np.argwhere(front_class_array==10 )
                            
                            front_all_id = []
                            front_ped_id = []
                            front_vehicle_id = []
                            for i, j in front_index_of_car:
                                front_all_id.append(front_id_array[i][j])
                                front_vehicle_id.append(front_id_array[i][j])

                            for i, j in front_index_of_ped:
                                front_all_id.append(front_id_array[i][j])
                                front_ped_id.append(front_id_array[i][j])

                            front_all_id = np.unique(front_all_id)
                            front_ped_id = np.unique(front_ped_id)
                            front_vehicle_id = np.unique(front_vehicle_id)




                            ins_front_array = torch.from_numpy(ins_front_array.copy())[:,:,:3].type(torch.int).permute((2,0,1))
                            produce_bbx(ins_front_array, actor_list_and_position, frame, front_img_array)
                            # print(" ======= ")

                            theta = math.radians(theta)
                            if np.isnan(theta):
                                theta = 0

                            R = np.array([
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)],
                            ])

                            PIXELS_PER_WORLD = 5.5 # from coverter.py 
                            ego_location = world.player.get_location()
                            ego_xy = np.float32([ego_location.x, ego_location.y])

                            target_loc = ego_transform_list[-1].location

                            if args.input_target:
                                target_loc.x = float(args.target_x)
                                target_loc.y = float(args.target_y)
                                # print("input target point")
                                
                            target_xy = np.float32([target_loc.x, target_loc.y])
                            target_xy = R.T.dot(target_xy - ego_xy)
                            #target_xy_print = target_xy
                            target_xy *= PIXELS_PER_WORLD # 5.5
                            # target_xy_print = target_xy
                            target_xy += [128, 256]
                            target_xy = np.clip(target_xy, 0, 256)
                            target_xy = torch.FloatTensor(target_xy)
                            target_xy = target_xy.reshape([1, 2])

                            topdown = topdown[0: 256, 128:384]
                            topdown = np.array(topdown)
                            topdown = common.CONVERTER[topdown]


                            

                            a = np.array((ego_location.x, ego_location.y ))
                            b = np.array((target_loc.x, target_loc.y ))

                            dist = np.linalg.norm(a-b)

                            # print(dist)
                            # print("____________________________")
                            # print(f"player id:{agents_dict['player'].id}")
                            
                        
                            
                            unique_topdown =  copy.deepcopy(topdown)
                            # to_draw_topdown =  copy.deepcopy(topdown)

                            # print(unique_topdown)

                            actor_id_array = actor_id_array[0: 256, 128:384]
                            unique_topdown = np.array(unique_topdown)
                            
                            # print(len(unique_topdown))

                            from inference_test.inference import inference 
                            from inference_test.inference import SA_inference 


                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


                            index_of_ped = np.argwhere(unique_topdown==1 )
                            index_of_car = np.argwhere(unique_topdown==5)
                            mask_id = []
                            ped_id = []
                            vehicle_id = []
                            for i, j in index_of_car:
                                mask_id.append(actor_id_array[i][j])
                                vehicle_id.append(actor_id_array[i][j])
                            for i, j in index_of_ped:
                                mask_id.append(actor_id_array[i][j])
                                ped_id.append(actor_id_array[i][j])
                            
                            unique_actor = np.unique(mask_id)
                            unique_actor = np.delete(unique_actor, np.where(unique_actor == agents_dict['player'].id))

                            # all_id = np.unique(mask_id)
                            ped_id = np.unique(ped_id)
                            vehicle_id = np.unique(vehicle_id)
                            

                            remove_list = []


                            # if frame == start_frame:
                            #     agent = BehaviorAgent(world.player, behavior="cautious")
                            #     destination = carla.Location( float(target_loc.x), float(target_loc.y), 0.6)
                            #     agent.set_destination(destination)


                            ######################################## Methods 
                            if frame > start_frame:
                                if args.method == 1:

                                    ## method 1 
                                    # remove ground truth id
                                    
                                    keys = list( agents_dict.keys())
                                    keys.remove('player')
                                    gt_actor_id = agents_dict[keys[0]].id
                                    remove_list.append(gt_actor_id)

                                elif args.method == 2:

                                    # method 2 
                                    # random choose one actor 
                                    
                                    np.random.seed(int(time.time()))
                                    remove_list = np.random.choice(front_all_id, 1)
                                
                                elif args.method == 3:

                                    ## method 3  
                                    # nearest actor

                                    ego_location = world.player.get_location()
                                    ego_location.x
                                    ego_location.y


                                    nearest_distance = 100000
                                    nearest_actor = None
                                    actors = world.world.get_actors()
                                    for actor in actors:
                                        if actor_transform_index['player'] < len(transform_dict[actor_id]):
                                            if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
                                                x = actor.get_location().x
                                                y = actor.get_location().y
                                                id = actor.id

                                                if id in front_all_id:
                                                    distacne_ego_other = math.sqrt( (x-ego_location.x)**2 + (y-ego_location.y)**2)
                                                    if distacne_ego_other < nearest_distance:
                                                        nearest_distance  = distacne_ego_other
                                                        nearest_actor = id

                                    remove_list = [nearest_actor]

                                elif args.method == 4:
                                    ## method 4 
                                    #  RSS 


                                    # method 7 RSS 
                                    start_time = time.time()
                                    
                                    if len(world.rss_sensor.predictions.keys()) > 0:
                                        while True:
                                            frames = world.rss_sensor.predictions.keys()
                                            
                                            if frame in frames:
                                                rss_pred_thisframe = world.rss_sensor.predictions[frame]
                                                remove_list = []
                                                # print(world.rss_sensor.predictions[frame])
                                                if not rss_pred_thisframe['EgoCarIsSafe']:
                                                    remove_list = rss_pred_thisframe['DangerousIds']
                                                    remove_list = [int(id) for id in remove_list]
                                                break
                                            if time.time() - start_time > 0.5:
                                                break
                                    # print(frame, frames)
                                    print(remove_list)

                                elif args.method == 5:

                                    ## method 5
                                    #  KalmanFilter

                                    from kf_one_step import kf_inference
                                    vehicle_list = []
                                    traj_df = pd.DataFrame(df_list, columns=['FRAME', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
                                    for _, remain_df in traj_df.groupby('TRACK_ID'): # actor id 
                                        filter = (remain_df.FRAME > ( (frame-1) - 20))

                                        remain_df = remain_df[filter].reset_index(drop=True)
                                        remain_df = remain_df.reset_index(drop=True)
                                        # print(remain_df)

                                        actor_pos_x = float( remain_df.loc[19, 'X'])
                                        actor_pos_y = float( remain_df.loc[19, 'Y'])
                                        dist_x = actor_pos_x - ego_location.x
                                        dist_y = actor_pos_y - ego_location.y
                                        if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                                            vehicle_list.append(remain_df)

                                    # print("-------------------")
                                    # print( (len(ped_id) + len(vehicle_id)))
                                    # print(len(vehicle_list))
                                    remove_list = kf_inference( frame-1, vehicle_list, ego_car_id, ped_id, vehicle_id)

                                elif args.method == 6:

                                    # method 6  
                                    # Social-GAN 
                                
                                    from social_gan.social_gan_one_step import socal_gan_inference
                                    
                                    vehicle_list = []
                                    traj_df = pd.DataFrame(df_list, columns=['FRAME', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
                                    traj_df['X'] = traj_df['X'].astype(float)
                                    traj_df['Y'] = traj_df['Y'].astype(float)


                                    for _, remain_df in traj_df.groupby('TRACK_ID'): # actor id 
                                        filter = (remain_df.FRAME > ( (frame-1) - 20))

                                        remain_df = remain_df[filter].reset_index(drop=True)
                                        remain_df = remain_df.reset_index(drop=True)
                                        # print(remain_df)

                                        actor_pos_x = float( remain_df.loc[19, 'X'])
                                        actor_pos_y = float( remain_df.loc[19, 'Y'])
                                        dist_x = actor_pos_x - ego_location.x
                                        dist_y = actor_pos_y - ego_location.y
                                        if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                                            vehicle_list.append(remain_df)
                                    remove_list = socal_gan_inference(vehicle_list, frame, ego_car_id , ped_id, vehicle_id, _args, generator)
                                

                                elif args.method == 7:

                                    ## method 7
                                    # MANTRA
                                
                                    from mantra.mantra_one_step import mantra_inference
                                    vehicle_list = []
                                    traj_df = pd.DataFrame(df_list, columns=['FRAME', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
                                    traj_df['X'] = traj_df['X'].astype(float)
                                    traj_df['Y'] = traj_df['Y'].astype(float)
                                    for _, remain_df in traj_df.groupby('TRACK_ID'): # actor id 
                                        filter = (remain_df.FRAME > ( (frame-1) - 20))

                                        remain_df = remain_df[filter].reset_index(drop=True)
                                        remain_df = remain_df.reset_index(drop=True)
                                        # print(remain_df)

                                        actor_pos_x = float( remain_df.loc[19, 'X'])
                                        actor_pos_y = float( remain_df.loc[19, 'Y'])
                                        dist_x = actor_pos_x - ego_location.x
                                        dist_y = actor_pos_y - ego_location.y
                                        if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                                            vehicle_list.append(remain_df)

                                    remove_list = mantra_inference(vehicle_list, frame, ego_car_id , ped_id, vehicle_id)

                                elif args.method == 8:
                                    ## method 8
                                    # DSA-RNN   

                                    if not os.path.exists("./inference_test/baseline2/"):
                                        os.mkdir("./inference_test/baseline2/")

                                    if not os.path.isfile("./inference_test/baseline2/model_19"):
                                        print("Download SA weight")
                                        url = "https://drive.google.com/u/4/uc?id=1UPlnYMlYsSSAPvmzg7LM5PU_6Zx0e6i6&export=download"
                                        gdown.download(url, "./inference_test/baseline2/model_19")

                                    risk_obj = SA_inference(device, "./roi_two_stage/inference/test_data/", "./inference_test/baseline2/model_19", start_frame=frame)
                                    remove_list = (risk_obj[-1])

                                elif args.method == 9:
                                    ## method 9
                                    # DSA-RNN-Supervised

                                    if not os.path.exists("./inference_test/baseline3/"):
                                        os.mkdir("./inference_test/baseline3/")

                                    if not os.path.isfile("./inference_test/baseline3/model_15"):
                                        print("Download risk region weight")
                                        url = "https://drive.google.com/u/4/uc?id=1WgP700b07kZGHSkZOmt8JrFIgFUuOPHG&export=download"
                                        gdown.download(url, "./inference_test/baseline3/model_15")
                                    risk_obj = inference(device, "./roi_two_stage/inference/test_data/", "./inference_test/baseline3/model_15", start_frame=frame)
                                    remove_list = (risk_obj[-1])

                                elif args.method == 10:
                                    ## method 10
                                    # BC single-stage  
                                    
                                    if frame == start_frame+1:
                                        remove_list = single_stage(start_frame=frame, clean_state = True)
                                    else:
                                        remove_list = single_stage(start_frame=frame)
                                    print(remove_list)

                                elif args.method == 11:
                                    
                                    ## method 11
                                    # BC two-stage
                                    
                                    if frame == start_frame+1:
                                        remove_list = roi_two_stage_inference(start_frame=frame, clean_state = True)
                                    else:
                                        remove_list = roi_two_stage_inference(start_frame=frame)


                            #######################

                            if frame > start_frame:
                                print(remove_list)
                                # print(unique_actor)
                                print(len(unique_actor))
                                for mask_id in remove_list:
                                    unique_actor = np.delete(unique_actor, np.where(unique_actor == int(mask_id)))
                                print(len(unique_actor))

                                
                                for mask_id in unique_actor:
                                    index_to_mask = np.argwhere(actor_id_array== int(mask_id))
                                    for i, j in index_to_mask:
                                        if args.method == 0:
                                            # method 0 
                                            # No Mask 
                                            pass
                                        else:
                                            topdown[i][j] = 3


                            index_of_ped = np.argwhere( topdown == 1 )
                            for m, n in index_of_ped:
                                
                                if (m+1) < 256:
                                    topdown[m+1][n] = 1
                                if (n+1) < 256:
                                    topdown[m][n+1] = 1
                                if (m-1) >= 0:
                                    topdown[m-1][n] = 1
                                if (m-1) >= 0:
                                    topdown[m][n-1] = 1

                            topdown_draw = common.COLOR[topdown]
                            topdown_draw = Image.fromarray(topdown_draw)
                            

                            # top_down_to_draw = copy.deepcopy(topdown) # pass to draw

                            topdown = torch.LongTensor(topdown)
                            topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()                            

                            
                            topdown = topdown.reshape([1, 6, 256, 256])
                            # inference

                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            
                            topdown = topdown.to(device)
                            target_xy = target_xy.to(device)


                            points_pred = net.forward(topdown, target_xy)
                            control = net.controller(points_pred).cpu().data.numpy()

                            ### draw 
                            _draw = ImageDraw.Draw(topdown_draw)
                            _draw.ellipse((target_xy[0][0]-2, target_xy[0][1]-2, target_xy[0][0]+2, target_xy[0][1]+2), (255, 255, 255))

                            if frame >= start_frame:
                                for x, y in points_pred.cpu().data.numpy()[0]:    
                                    x = (x + 1) / 2 * 256
                                    y = (y + 1) / 2 * 256
                                    _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))


                            if frame >= start_frame:
                                topdown_draw.save("./image/"+str(frame)+".png")
                                image_top = cv2.imread("./image/"+str(frame)+".png")

                                out_topview.write(image_top)

                                pygame.image.save(display, "screenshot.jpeg")
                                image = cv2.imread("screenshot.jpeg")
                                out.write(image)

                            #################### learned contorl ###################

                            control = control[0]
                            steer = control[0] * 1.3
                            desired_speed = control[1] 
                            speed = world.player.get_velocity()
                            speed = ((speed.x)**2 + (speed.y)**2+(speed.z)**2)**(0.5)
                            desired_speed += speed
                            if desired_speed < 0:
                                desired_speed += 1.75
                        
                            brake = desired_speed < 0.5 or (speed / desired_speed) > 1.1

                            delta = np.clip(desired_speed - speed, 0.0, 0.5)
                            throttle = ego_speed_controller.step(delta)
                            throttle = np.clip(throttle, 0.0, 0.5) 
                            throttle = throttle if not brake else 0.0

                            print("desired_speed and speed ")
                            print(f"desired speed: {desired_speed}")
                            print(f"current speed: {speed}")
                            print(f"speed / desired speed: {speed / desired_speed}")

                            
                            control = carla.VehicleControl()
                            control.steer = float(steer)
                            control.throttle = float(throttle) 
                            control.brake = float(brake)

                            #print(dist)
                            if frame >= start_frame:
                                if dist > 2.0 :
                                    world.player.apply_control(control)
                                    
                                else:
                                    # print(" reach the goal ")
                                    throttle = 0.0
                                    control = carla.VehicleControl()
                                    control.steer = float(0.0)
                                    control.throttle = float(0.0)
                                    control.brake = float(True)
                                    world.player.apply_control(control)
                                    abandon_scenario = True


                            # if frame >= start_frame:

                            #     keys = list( agents_dict.keys())
                            #     keys.remove('player')
                            #     gt_actor_id = agents_dict[keys[0]].id

                            #     control = agent.run_step(gt_actor_id)
                            #     print(control)
                            #     world.player.apply_control(control)

                            #     if agent.done():
                            #         abandon_scenario = True


                            # caculate the distance between ego car and gt interactor
                            ego_location = world.player.get_location()
                            
                            
                            keys = list( agents_dict.keys())
                            keys.remove('player')
                            gt_actor_location = agents_dict[keys[0]].get_transform().location



                            distance = math.sqrt( (ego_location.x - gt_actor_location.x)**2 + (ego_location.y - gt_actor_location.y)**2)
                            if min_distance_with_gt > distance:
                                min_distance_with_gt = distance

                            if frame >= start_frame and frame < start_frame + 40: 
                                total_distance += distance
                                distance_counter += 1
                                avg_distance = float(total_distance/distance_counter)


                if not False in finish.values():
                    break

                if controller.parse_events(client, world, clock) == 1:
                    return

                if "vehicle" in str(agents_dict[keys[0]]):
                    distance_threadhold = 1.7
                else :
                    distance_threadhold = 2.5
                if min_distance_with_gt < distance_threadhold:
                    abandon_scenario = True
                    is_collision = True

                if world.collision_sensor.collision and frame > start_frame : 
                    print('unintentional collision, abandon scenario')
                    abandon_scenario = True
                    is_collision = True
                elif world.collision_sensor.wrong_collision and frame > start_frame:
                    print('collided with wrong object, abandon scenario')
                    abandon_scenario = True
                    is_collision = True

                if abandon_scenario and frame > start_frame:
                    world.abandon_scenario = True
                    break

            if iter_tick == iter_toggle:
                if not args.no_save:
                    time.sleep(10)
                    world.camera_manager.toggle_recording(stored_path)
                    world.imu_sensor.toggle_recording_IMU()
                    world.gnss_sensor.toggle_recording_Gnss()
                    traj_col = threading.Thread(target=collect_trajectory, args=(
                        world, world.player, args.scenario_id, period, stored_path, clock))
                    traj_col.start()
                    topo_col = threading.Thread(target=collect_topology, args=(
                        world, world.player, args.scenario_id, half_period, root, stored_path, clock))
                    topo_col.start()
                    
            world.tick(clock)
            world.render(display)
            pygame.display.flip()


        if not args.no_save and not abandon_scenario:
            world.imu_sensor.toggle_recording_IMU()
            world.save_ego_data(stored_path)
            world.collision_sensor.save_history(stored_path)
            time.sleep(10)
            world.camera_manager.toggle_recording(stored_path)
            save_description(world, args, stored_path, weather, agents_dict, nearest_obstacle)
            world.finish = True
        try:
            if traj_col:
                traj_col.join()
                topo_col.join()
        except:
            pass
        
    finally:

        out.release()
        out_topview.release()

        # print(f"{args.scenario_type} {args.scenario_id} {weather} {args.random_actors} {is_collision}")
        cap = cv2.VideoCapture( output_name)

        if cap.isOpened():
            f = open(f"./test_result/result_{args.scenario_type}.txt", 'a')
            f.write(f"{args.scenario_id}#{weather}#{args.random_actors} {is_collision} {avg_distance} {min_distance_with_gt}\n")
            f.close()
        cap.release()
        
        # print('Closing...')
        if not args.no_save:
            client.stop_recorder() # end recording
        
        if (world and world.recording_enabled):
            client.stop_recorder()

        # print('destroying vehicles')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        # print('destroying walkers')
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

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
        default='1280x720',
        help='window resolution (default: 1280x720)')
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
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
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
        '--no_save',
        default=False,
        action='store_true',
        help='run scenarios only')
    argparser.add_argument(
        '--save_rss',
        default=False,
        action='store_true',
        help='save rss predictinos')
    argparser.add_argument(
        '--replay',
        default=False,
        action='store_true',
        help='use random seed to generate the same behavior')
    argparser.add_argument(
        '--input_target',
        default=False,
        action='store_true',
        help='if input target point')
    argparser.add_argument(
        '--target_x',
        type=float,
        default=0.0,
        )
    argparser.add_argument(
        '--target_y',
        type=float,
        default=0.0,
        )
    argparser.add_argument(
        '--start_frame',
        type=int,
        default=60,
        )
    argparser.add_argument(
        '--random_seed',
        type=int,
        default=1000,
        )
    argparser.add_argument(
        '--method',
        type=int,
        default=0,
        )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    # exec("args.weather = carla.WeatherParameters.%s" % args.weather)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
