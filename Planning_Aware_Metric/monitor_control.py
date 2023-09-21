#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h
    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light
    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle
    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)
    R            : toggle recording images to disk
    O            : Set coordinate
    E            : Save every vehicles' coordinate.
    K            : Teleport all vehicles to save coordinate
    F            : Changing traffic light state.
    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser


import carla

from carla import ColorConverter as cc
from carla import Transform, Location, Rotation

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import json


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
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
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


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.scenario_type = args.scenario_type
        self.hud = hud
        self.player = None
        self.npc = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
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

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
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
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(
            self.player, self.hud, self._gamma, self.scenario_type)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
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
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self.control_list = []
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
        self.r = 2
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get(
            'G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, client, world, clock):

        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.r = 1
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == 10:  # R3
                    self.r = world.camera_manager.toggle_recording()
                    return self.r

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    self.r = 1
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
                    npc_id = int(input("NPC id ? "))
                    if npc_id == "None":
                        world.npc = None
                    else:
                        world.npc = world.world.get_actor(npc_id)
                elif event.key == K_f:
                    if world.player.is_at_traffic_light():
                        traffic_light = world.player.get_traffic_light()
                        if traffic_light.get_state() == carla.TrafficLightState.Red:
                            print("Set traffic light to green.")
                            traffic_light.set_state(
                                carla.TrafficLightState.Green)
                        elif traffic_light.get_state() == carla.TrafficLightState.Green:
                            print("Set traffic light to red.")
                            traffic_light.set_state(
                                carla.TrafficLightState.Red)
                elif event.key == K_e:
                    self.save_act_transform = []
                    actors = world.world.get_actors().filter('vehicle.*')
                    for actor in actors:
                        self.save_act_transform.append(actor.get_transform())
                    print("save finish")
                elif event.key == K_k:
                    if self.save_act_transform is not None:
                        actors = world.world.get_actors().filter('vehicle.*')
                        for i, actor in enumerate(actors):
                            actor.set_transform(self.save_act_transform[i])
                        print("set finish")
                
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_o:
                    if_npc = True if input("NPC or player? ") == "NPC" else False
                    xyz = [float(s) for s in input(
                        'Enter coordinate: x , y , z  : ').split()]
                    new_location = carla.Location(xyz[0], xyz[1], xyz[2])
                    if if_npc:
                        world.npc.set_location(new_location)
                    else:
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
                    self.r = world.camera_manager.toggle_recording()
                    return self.r
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

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(
                    pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
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
                    pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)
            return self.r

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
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
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self.control_list = []
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
        self.r = 2
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.save_act_transform = None

    def parse_events(self, client, world, clock):

        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.r = 1
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    self.r = 1
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
                    npc_id = int(input("NPC id ? "))
                    if npc_id == "None":
                        world.npc = None
                    else:
                        world.npc = world.world.get_actor(npc_id)
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_e:
                    self.save_act_transform = []
                    actors = world.world.get_actors().filter('vehicle.*')
                    for actor in actors:
                        self.save_act_transform.append(actor.get_transform())
                    print("save finish")
                elif event.key == K_k:
                    if self.save_act_transform is not None:
                        actors = world.world.get_actors().filter('vehicle.*')
                        for i, actor in enumerate(actors):
                            actor.set_transform(self.save_act_transform[i])
                        print("set finish")
                elif event.key == K_f:
                    if world.player.is_at_traffic_light():
                        traffic_light = world.player.get_traffic_light()
                        if traffic_light.get_state() == carla.TrafficLightState.Red:
                            print("Set traffic light to green.")
                            traffic_light.set_state(
                                carla.TrafficLightState.Green)
                        elif traffic_light.get_state() == carla.TrafficLightState.Green:
                            print("Set traffic light to red.")
                            traffic_light.set_state(
                                carla.TrafficLightState.Red)
                elif event.key == K_o:
                    if_npc = True if input("NPC or player? ") == "NPC" else False
                    xyz = [float(s) for s in input(
                        'Enter coordinate: x , y , z  : ').split()]
                    new_location = carla.Location(xyz[0], xyz[1], xyz[2])
                    if if_npc:
                        world.npc.set_location(new_location)
                    else:
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
                    self.r = world.camera_manager.toggle_recording()
                    return self.r
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
            world.player.apply_control(self._control)
            return self.r

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
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

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
        # compass = world.imu_sensor.compass
        # heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        # heading += 'S' if 90.5 < compass < 269.5 else ''
        # heading += 'E' if 0.5 < compass < 179.5 else ''
        # heading += 'W' if 180.5 < compass < 359.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
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
        # self._info_text += [
        #     '',
        #     'Collision:',
        #     collision,
        #     '',
        #     'Number of vehicles: % 8d' % len(vehicles)]
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
        self._notifications.render(display)
        self.help.render(display)


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
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


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
    def __init__(self, parent_actor):
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


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
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

            norm_velocity = detect.vescnario_name_maplocity / \
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
    def __init__(self, parent_actor, hud, gamma_correction, scenario_type):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.scenario_type = scenario_type
        self.hud = hud
        self.recording = False
        self.record_image = []
        self.start_time = None
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y,
                 z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=+0.8*bound_x,
                 y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+
                 1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y,
                 z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
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
                'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                 'lens_circle_falloff': '3.0',
                 'chromatic_aberration_intensity': '0.5',
                 'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
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
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        restart = 0
        if not self.recording:
            end_time = time.time()

            # original description

            # scenario_name_actor_type = {'c': 'car', 't': 'truck', 'b': 'bike', 'm': 'motor', 'p': 'pedestrian', '0': 'None'}
            # scenario_name_actor_type_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',\
            #  'sr': 'slide_right', 'u': 'u-turn', 's':'stop', 'c': 'crossing', '0': 'None'}
            # scenario_name_my_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',\
            #  'sr': 'slide_right', 'u': 'u-turn'}
            # scenario_name_interaction = {'0': 'False', '1':  'True'}
            # scenario_name_violated_rule = {'0': 'None','p': 'parking', 'j': 'jay-walker',
            # 'rl': 'running traffic light', 's': 'driving on sidewalk', 'ss': 'stop sign'}

            # enter your scenario id
            # and there are some format requirement need to clarify
            scenario_name_map = {'1': 'Town01', '2': 'Town02', '3': 'Town03', '4': 'Town04', '5': 'Town05',
                                 '6': 'Town06', '7': 'Town07', '10': 'Town10HD', "A1": "A1", "A6": "A6"}
            scenario_name_is_traffic_light = {'1': 'true', '0':  'false'}

            if self.scenario_type == 'interactive':

                scenario_name_actor_type = {
                    'c': 'car', 't': 'truck', 'b': 'bike', 'm': 'motor', 'p': 'pedestrian'}

                scenario_name_actor_type_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn',
                                                   'sl': 'slide_left', 'sr': 'slide_right', 'u': 'u-turn',
                                                   'c': 'crossing', 'w': 'walking on sidewalk', 'j': 'jaywalking',
                                                   'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'}

                scenario_name_my_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                                           'sr': 'slide_right', 'u': 'u-turn', 'gi': 'go into roundabout',
                                           'gr': 'go around roundabout', 'er': 'exit roundabout'}

                scenario_name_interaction = {'0': 'False', '1':  'True'}

                scenario_name_violated_rule = {'0': 'None', 'j': 'jay-walker',
                                               'rl': 'running traffic light', 's': 'driving on sidewalk', 'ss': 'stop sign'}

                while True:
                    scenario_name = ""
                    print("Scenario Categorization:")
                    print("Input map_id:")
                    for key in scenario_name_map:
                        print(key + ': ' + scenario_name_map[key] + ' ')
                    input_option = str(input())
                    # if record is wrong press 0
                    if input_option == '0':
                        print("Cancelling")
                        restart = 1
                        break
                    if input_option not in scenario_name_map:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += input_option

                    print("Input road_id:")
                    input_option = str(input())
                    scenario_name += "_" + input_option

                    print("Input is_traffic_light:")
                    for key in scenario_name_is_traffic_light:
                        print(key + ': ' +
                              scenario_name_is_traffic_light[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_is_traffic_light:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input actor_type:")
                    for key in scenario_name_actor_type:
                        print(key + ': ' + scenario_name_actor_type[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_actor_type:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input actor_action:")
                    for key in scenario_name_actor_type_action:
                        print(key + ': ' +
                              scenario_name_actor_type_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_actor_type_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input name_my_action:")
                    for key in scenario_name_my_action:
                        print(key + ': ' + scenario_name_my_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_my_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("is_interactive: 1: True")
                    input_option = '1'
                    scenario_name += "_" + input_option

                    print("Input name_violated_rule:")
                    for key in scenario_name_violated_rule:
                        print(key + ': ' +
                              scenario_name_violated_rule[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_violated_rule:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += "_" + input_option                    
                    break

            elif self.scenario_type == 'non-interactive':
                scenario_name_actor_type = {'0': 'None'}
                scenario_name_actor_type_action = {'0': 'None'}
                scenario_name_my_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                                           'sr': 'slide_right', 'u': 'u-turn', 'gi': 'go into roundabout',
                                           'gr': 'go around roundabout', 'er': 'exit roundabout'}
                scenario_name_interaction = {'0': 'False'}
                scenario_name_violated_rule = {'0': 'None', 'j': 'jay-walker',
                                               'rl': 'running traffic light', 's': 'driving on sidewalk', 'ss': 'stop sign'}

                while True:
                    scenario_name = ""
                    print("Scenario Categorization:")
                    print("Input map_id:")
                    for key in scenario_name_map:
                        print(key + ': ' + scenario_name_map[key] + ' ')
                    input_option = str(input())
                    # if record is wrong press 0
                    if input_option == '0':
                        print("Cancelling")
                        restart = 1
                        break
                    if input_option not in scenario_name_map:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += input_option

                    print("Input road_id:")
                    input_option = str(input())
                    scenario_name += "_" + input_option

                    print("Input is_traffic_light:")
                    for key in scenario_name_is_traffic_light:
                        print(key + ': ' +
                              scenario_name_is_traffic_light[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_is_traffic_light:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("actor_type: 0: None")
                    for key in scenario_name_actor_type:
                        print(key + ': ' + scenario_name_actor_type[key] + ' ')
                    input_option = '0'
                    scenario_name += "_" + input_option

                    print("actor_action: 0: None")
                    input_option = '0'
                    scenario_name += "_" + input_option

                    print("Input name_my_action:")
                    for key in scenario_name_my_action:
                        print(key + ': ' + scenario_name_my_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_my_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("is_interactive: 0: False")
                    input_option = '0'
                    scenario_name += "_" + input_option

                    print("Input name_violated_rule:")
                    for key in scenario_name_violated_rule:
                        print(key + ': ' +
                              scenario_name_violated_rule[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_violated_rule:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += "_" + input_option
                    break

            elif self.scenario_type == 'obstacle':
                scenario_name_my_initial_action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left', 'sr': 'slide_right',
                                                   'u': 'u-turn', 'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'}

                scenario_name_my_action = {'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left', 'sr': 'slide_right',
                                           'u': 'u-turn', 'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'}

                obstacle_type = {'0': 'traffic cone',
                                 '1': 'street barrier', '2': 'traffic warning', '3': 'illegal parking'}

                while True:
                    scenario_name = ""
                    print("Scenario Categorization:")
                    print("Input map_id:")
                    for key in scenario_name_map:
                        print(key + ': ' + scenario_name_map[key] + ' ')
                    input_option = str(input())
                    # if record is wrong press 0
                    if input_option == '0':
                        print("Cancelling")
                        restart = 1
                        break
                    if input_option not in scenario_name_map:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += input_option

                    print("Input road_id:")
                    input_option = str(input())
                    scenario_name += "_" + input_option

                    print("Input obstacle_type:")
                    for key in obstacle_type:
                        print(key + ': ' +
                              obstacle_type[key] + ' ')
                    input_option = str(input())
                    if input_option not in obstacle_type:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input ego's initial action:")
                    for key in scenario_name_my_initial_action:
                        print(key + ': ' +
                              scenario_name_my_initial_action[key] + ' ')

                    input_option = str(input())
                    if input_option not in scenario_name_my_initial_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input ego's reaction:")
                    for key in scenario_name_my_action:
                        print(key + ': ' + scenario_name_my_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_my_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += "_" + input_option
                    break
            
            else:
                # Collision
                scenario_name_actor_type = {
                    'c': 'car', 't': 'truck', 'b': 'bike', 'm': 'motor', 'p': 'pedestrian', 's': 'static_object'}

                scenario_name_actor_type_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn',
                                                   'sl': 'slide_left', 'sr': 'slide_right', 'u': 'u-turn',
                                                   'c': 'crossing', 'w': 'walking on sidewalk', 'j': 'jaywalking',
                                                   'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout', '0': 'None'}

                scenario_name_my_action = {'f': 'forward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                                           'sr': 'slide_right', 'u': 'u-turn', 'gi': 'go into roundabout', 'gr': 'go around roundabout',
                                           'er': 'exit roundabout', 'ri': 'Crash into refuge island'}

                # scenario_name_interaction = {'0': 'False', '1':  'True'}

                scenario_name_violated_rule = {'0': 'None', 'p': 'parking', 'j': 'jay-walker',
                                               'rl': 'running traffic light', 's': 'driving on sidewalk', 'ss': 'stop sign'}

                while True:
                    scenario_name = ""
                    print("Scenario Categorization:")
                    print("Input map_id:")
                    for key in scenario_name_map:
                        print(key + ': ' + scenario_name_map[key] + ' ')
                    input_option = str(input())
                    # if record is wrong press 0
                    if input_option == '0':
                        print("Cancelling")
                        restart = 1
                        break
                    if input_option not in scenario_name_map:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue

                    scenario_name += input_option

                    print("Input road_id:")
                    input_option = str(input())
                    scenario_name += "_" + input_option

                    print("Input is_traffic_light:")
                    for key in scenario_name_is_traffic_light:
                        print(key + ': ' +
                              scenario_name_is_traffic_light[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_is_traffic_light:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input actor_type:")
                    for key in scenario_name_actor_type:
                        print(key + ': ' + scenario_name_actor_type[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_actor_type:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input actor_action:")
                    for key in scenario_name_actor_type_action:
                        print(key + ': ' +
                              scenario_name_actor_type_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_actor_type_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    scenario_name += "_" + input_option

                    print("Input name_my_action:")
                    for key in scenario_name_my_action:
                        print(key + ': ' + scenario_name_my_action[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_my_action:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    
                    scenario_name += "_" + input_option

                    # print("Input is_interactive:")
                    # for key in scenario_name_interaction:
                    #     print(key + ': ' +
                    #           scenario_name_interaction[key] + ' ')
                    # input_option = str(input())
                    # if input_option not in scenario_name_interaction:
                    #     print("INVALID INPUT! RESTART NAMING SCENARIO.")
                    #     continue
                    # scenario_name += "_" + input_option

                    print("Input name_violated_rule:")
                    for key in scenario_name_violated_rule:
                        print(key + ': ' +
                              scenario_name_violated_rule[key] + ' ')
                    input_option = str(input())
                    if input_option not in scenario_name_violated_rule:
                        print("INVALID INPUT! RESTART NAMING SCENARIO.")
                        continue
                    
                    scenario_name += "_" + input_option
                    break

            if not restart:
                scenario_num = 0
                path = os.path.join(
                    'data_collection', self.scenario_type, scenario_name)
                
                if os.path.isdir(path):
                    scenario_num += 1
                    
                    while True:
                        if os.path.isdir(path + '_' + str(scenario_num)):
                            scenario_num += 1
                        else:
                            scenario_name = scenario_name + \
                                '_' + str(scenario_num)
                            break

                self.scenario_id = scenario_name
                # for img in self.record_image:
                #     if img.frame % 100 == 0:
                #         img.save_to_disk(
                #             '_out/%s/%s/%08d' % (self.sensors[self.index][2], scenario_name, img.frame))
                print('Recorded video time : %4.2f seconds' %
                      (end_time-self.start_time))

            self.record_image = []
        else:
            self.start_time = time.time()

        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))
        if self.recording:
            return 3
        elif restart:
            return 6
        else:
            return 4

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
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
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            self.record_image.append(image)


def record_transform(actor_dict, world):
    actor_list = [world.player, world.npc]
    for i, actor in enumerate(actor_list):
        if actor is None:
            continue
        transform = actor.get_transform()
        np_transform = np.zeros(7)
        np_transform[0:3] = [transform.location.x,
                             transform.location.y, transform.location.z]
        np_transform[3:6] = [transform.rotation.pitch,
                             transform.rotation.yaw, transform.rotation.roll]
        if i == 0:
            actor_dict['player']['transform'].append(np_transform)
        else:
            actor_dict[str(actor.id)]['transform'].append(np_transform)
    return actor_dict


def record_ped_control(control_dict, world):
    actor = world.npc
    if actor is not None:
        if 'pedestrian' in actor.type_id:
            control = actor.get_control()
            np_control = np.zeros(5)
            np_control[0:5] = [control.direction.x, control.direction.y, control.direction.z,
                               control.speed, control.jump]

            control_dict[str(actor.id)]['control'].append(np_control)
    return control_dict


def record_velocity(actor_dict, world):
    actor_list = [world.player, world.npc]
    for i, actor in enumerate(actor_list):
        if actor is None:
            continue
        velocity = actor.get_velocity()
        np_velocity = np.zeros(3)
        np_velocity = [velocity.x, velocity.y, velocity.z]
        if i == 0:
            actor_dict['player']['velocity'].append(np_velocity)
        else:
            actor_dict[str(actor.id)]['velocity'].append(np_velocity)
    return actor_dict


def extract_actor(actor_dict, control_dict, world):
    actor_list = [world.player, world.npc]
    for actor in actor_list:
        if actor is None:
            continue
        # if actor.id not in actor_dict:
        if actor.id == world.player.id:
            actor_dict['player'] = {'filter': str(world.player.type_id),
                                    'transform': [],
                                    'velocity': []}
        elif 'vehicle' in str(actor.type_id) or 'pedestrian' in str(actor.type_id):
            actor_dict[str(actor.id)] = {'filter': actor.type_id,
                                         'transform': [],
                                         'velocity': []}
        if 'pedestrian' in str(actor.type_id):
            control_dict[str(actor.id)] = {'control': []}
    return actor_dict, control_dict


def save_actor(stored_path, actor_dict, control_dict, timestamp_list, obstacle_list):
    if not os.path.exists(os.path.join(stored_path, 'transform')):
        os.mkdir(os.path.join(stored_path, 'transform'))
    if not os.path.exists(os.path.join(stored_path, 'ped_control')):
        os.mkdir(os.path.join(stored_path, 'ped_control'))
    if not os.path.exists(os.path.join(stored_path, 'velocity')):
        os.mkdir(os.path.join(stored_path, 'velocity'))
    if not os.path.exists(os.path.join(stored_path, 'filter')):
        os.mkdir(os.path.join(stored_path, 'filter'))
    if not os.path.exists(os.path.join(stored_path, 'timestamp/')):
        os.mkdir(os.path.join(stored_path, 'timestamp'))
    if not os.path.exists(os.path.join(stored_path, 'obstacle')):
        os.mkdir(os.path.join(stored_path, 'obstacle'))

    for actor_id, data in actor_dict.items():
        np.save(stored_path + '/transform/%s' %
                (actor_id), np.array(data['transform']))
        # velocity list np array saved as a .npy file
        np.save(stored_path + '/velocity/%s' %
                (actor_id), np.array(data['velocity']))
        with open(stored_path + "/filter/%s.txt" % (actor_id), "w") as text_file:
            text_file.write(str(data['filter']))
        data['transform'] = []
        data['velocity'] = []
        data['filter'] = []

    with open(stored_path + "/obstacle/obstacle_list.json", "w") as f:
        json.dump(obstacle_list, f, indent=4)

    for actor_id, data in control_dict.items():
        np.save(stored_path + '/ped_control/%s' %
                (actor_id), np.array(data['control']))
        data['control'] = []

    time_file = open(stored_path + "/timestamp.txt", "w")
    for time in timestamp_list:
        time_file.write(str(time[0]) + ',' + str(time[1]) + "\n")
    time_file.close()
    timestamp_list = []
    return actor_dict, control_dict, timestamp_list


def save_description(stored_path, scenario_type, scenario_name, carla_map):
    description = scenario_name.split('_')
    topo = description[1].split('-')[0]
    d = dict()

    if 'i' in topo:
        d['topology'] = '4_way_intersection'
    elif 't' in topo:
        if topo[1] == '1':
            d['topology'] = '3_way_intersection_1'
        elif topo[1] == '2':
            d['topology'] = '3_way_intersection_2'
        elif topo[1] == '3':
            d['topology'] = '3_way_intersection_3'
    elif 'r' in topo:
        d['topology'] = 'roundabout'
    elif 's' in topo:
        d['topology'] = 'straight'

    if scenario_type == 'interactive':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]
        actor = {'c': 'car', 't': 'truck', 'b': 'bike',
                 'm': 'motor', 'p': 'pedestrian'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                  'sr': 'slide_right', 'u': 'u-turn', 'c': 'crossing',
                  'w': 'walking on sidewalk', 'j': 'jaywalking',
                  # roundabout
                  'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'
                  }

        violation = {'0': 'None', 'p': 'parking', 'j': 'jay-walker', 'rl': 'running traffic light',
                     's': 'driving on a sidewalk', 'ss': 'stop sign'}

        interaction = {'1': 'True'}

        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[7]]
        d['map'] = carla_map

    elif scenario_type == 'non-interactive':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]
        actor = {'0': 'None'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'lane-change-left',
                  'sr': 'lane-change-right', 'u': 'u-turn', 'b': 'backward', 'j': 'jaywalking',
                  # roundabout
                  'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout',
                  '0': 'None'}

        violation = {'0': 'None', 'rl': 'running traffic light',
                     's': 'driving on a sidewalk', 'ss': 'stop sign'}

        interaction = {'0': 'False'}

        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[7]]
        d['map'] = carla_map

    elif scenario_type == 'obstacle':
        initial_action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'lane-change-left', 'sr': 'lane-change-right',
                          'u': 'u-turn', 'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'}

        action = {'l': 'left_turn', 'r': 'right_turn', 'sl': 'lane-change-left', 'sr': 'lane-change-right', 'u': 'u-turn',
                  # roundabout
                  'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout'}

        obstacle_type = {'0': 'traffic cone',
                         '1': 'street barrier', '2': 'traffic warning', '3': 'illegal parking'}

        d['obstacle type'] = obstacle_type[description[2]]
        d['my_initial_action'] = initial_action[description[3]]
        d['my_action'] = action[description[4]]
        d['map'] = carla_map

    elif scenario_type == 'collision':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]

        actor = {'c': 'car', 't': 'truck', 'b': 'bike',
                 'm': 'motor', 'p': 'pedestrian', 's': 'static_object'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'lane-change-left',
                  'sr': 'lane-change-right', 'u': 'u-turn', 'c': 'crossing',
                  'w': 'walking on sidewalk', 'j': 'jaywalking',
                  # roundabout
                  'gi': 'go into roundabout', 'gr': 'go around roundabout', 'er': 'exit roundabout',
                  '0': 'None'}

        violation = {'0': 'None', 'p': 'parking', 'j': 'jay-walker', 'rl': 'running traffic light',
                     's': 'driving on a sidewalk', 'ss': 'stop sign'}

        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        # d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[6]]
        d['map'] = carla_map

    with open(os.path.join(stored_path, 'scenario_description.json'), 'w') as f:
        json.dump(d, f, indent=4)


def record_traffic_lights(lights_dict, lights):
    for l in lights:
        if not l.id in lights_dict:
            lights_dict[l.id] = []
            location = l.get_location()
            lights_dict[l.id].append([location.x, location.y, location.z])
        lights_dict[l.id].append([str(l.get_state()), 0, 0])
    return lights_dict


def save_traffic_lights(stored_path, lights_dict):

    if not os.path.exists(stored_path + '/traffic_light/'):
        os.mkdir(stored_path + '/traffic_light/')

    for l_id, state in lights_dict.items():
        np.save(stored_path + '/traffic_light/%s' %
                (str(l_id)), np.array(lights_dict[l_id]))


def generate_obstacle(world, n, area, map, scenario_tag):

    blueprint_library = world.get_blueprint_library()
    all_wp = world.get_map().generate_waypoints(6)
    random.shuffle(all_wp)

    intersection_coordinators = scenario_tag_to_location(map)
    spawn_transform = [carla.Transform(location=carla.Location(
        *intersection_coordinators[scenario_tag]))]

    obstacle_list = []
    trans_list = []
    stat_prop = ["static.prop.trafficcone01", "static.prop.trafficcone02",
                 "static.prop.trafficwarning", "static.prop.streetbarrier"]

    def dist(t, L, limit=30):
        for trans in L:
            if trans.location.distance(t.location) < limit:
                return False
        return True

    def spawn_junction(vec, trans, id):
        new_rotation = carla.Rotation(
            pitch=0, yaw=trans.rotation.yaw-90, roll=0)
        new_trans = carla.Transform(trans.location, trans.rotation)

        r = 2.5
        new_trans.location += (vec)*r

        actor = world.spawn_actor(
            blueprint_library.filter(stat_prop[id])[0], new_trans)

        obstacle_attr = {"obstacle_type": stat_prop[id],
                         "basic_id": actor.id,
                         "location": {"x": new_trans.location.x, "y": new_trans.location.y, "z": new_trans.location.z},
                         "rotation": {"pitch": new_trans.rotation.pitch, "yaw": new_trans.rotation.yaw, "roll": new_trans.rotation.roll}}

        obstacle_list.append(obstacle_attr)
        trans_list.append(new_trans)

    def spawn_twoWarning(trans, id):
        new_rotation = carla.Rotation(
            pitch=0, yaw=trans.rotation.yaw-90, roll=0)
        new_trans = carla.Transform(trans.location, new_rotation)

        actor = world.spawn_actor(
            blueprint_library.filter(stat_prop[id])[0], new_trans)

        obstacle_attr = {"obstacle_type": stat_prop[id],
                         "basic_id": actor.id,
                         "location": {"x": new_trans.location.x, "y": new_trans.location.y, "z": new_trans.location.z},
                         "rotation": {"pitch": new_trans.rotation.pitch, "yaw": new_trans.rotation.yaw, "roll": new_trans.rotation.roll}}

        obstacle_list.append(obstacle_attr)
        trans_list.append(new_trans)

    def spawn_straight(vec, trans, id):
        new_rotation = carla.Rotation(
            pitch=0, yaw=trans.rotation.yaw-90, roll=0)
        new_trans = carla.Transform(trans.location, new_rotation)

        new_trans.location += vec

        actor = world.spawn_actor(
            blueprint_library.filter(stat_prop[id])[0], new_trans)

        obstacle_attr = {"obstacle_type": stat_prop[id],
                         "basic_id": actor.id,
                         "location": {"x": new_trans.location.x, "y": new_trans.location.y, "z": new_trans.location.z},
                         "rotation": {"pitch": new_trans.rotation.pitch, "yaw": new_trans.rotation.yaw, "roll": new_trans.rotation.roll}}

        obstacle_list.append(obstacle_attr)
        trans_list.append(new_trans)

    if n == 1:
        for k, wp in enumerate(all_wp):
            if dist(wp.transform, spawn_transform, 60):
                continue
            if dist(wp.transform, trans_list, 30) and random.randint(0, 2) == 0:

                trans = wp.transform
                vec_0 = carla.Vector3D(0, 0, 0)
                vec_f = trans.get_forward_vector()
                vec_r = trans.get_right_vector()

                trans.location += vec_f*5

                rand = random.randint(0, 2)

                if rand < 2:
                    spawn_junction(vec_0, trans, 0)
                    spawn_junction(vec_r, trans, 0)
                    spawn_junction(vec_f, trans, 0)
                    spawn_junction(vec_f+vec_r, trans, 0)
                else:
                    spawn_junction(vec_0, trans, 3)
                    spawn_junction(vec_r, trans, 3)

    elif 2 <= n <= 3:
        k = -1
        for i in range(area):

            while k+1 < len(all_wp):
                k += 1
                wp = all_wp[k]
                if dist(wp.transform, spawn_transform, 60):
                    continue
                elif scenario_tag == "r-1" and not dist(wp.transform, trans_list, 40):
                    break
                elif not wp.is_junction and dist(wp.transform, trans_list, 40) and dist((wp.next_until_lane_end(4))[-1].transform, trans_list, 40):
                    break

            if k >= len(all_wp):
                break

            # while True:
            #     wp = random.choice(all_wp)
            #     if not wp.is_junction and dist(wp.transform, trans_list, 40)    \
            #             and dist((wp.next_until_lane_end(4))[-1].transform, trans_list, 40):
            #         break

            wp_list = []
            # wp_list = (wp.previous_until_lane_start(4))[::-1]
            wp_list.extend(wp.next_until_lane_end(4))

            if len(wp_list) < 5:
                continue

            print('###', i, '###', len(wp_list))

            rand = random.randint(4, len(wp_list)-1)
            spawn_twoWarning(wp_list[rand].transform, n)
            spawn_twoWarning(wp_list[rand-4].transform, n)

    elif n >= 4:
        k = -1
        n_obstacle = 0

        while n_obstacle < n:

            while k+1 < len(all_wp):
                k += 1
                wp = all_wp[k]
                if dist(wp.transform, spawn_transform, 60):
                    continue
                elif not wp.is_junction and dist(wp.transform, trans_list, 40) and dist((wp.next_until_lane_end(4))[-1].transform, trans_list, 40):
                    break

            if k >= len(all_wp):
                break

            wp_list = []
            # wp_list = (wp.previous_until_lane_start(4))[::-1]
            wp_list.extend(wp.next_until_lane_end(4))

            for idx in range(len(wp_list)-2, -1, -1):
                if wp_list[idx].is_junction or not dist(wp_list[idx].transform, [wp_list[idx+1].transform], 3):
                    for _ in range(idx):
                        wp_list[::-1].pop(0)
                    break

            if len(wp_list) < 5:
                continue
            else:
                n_obstacle += 1
                print('###', n_obstacle, '###', len(wp_list))

            vector = (wp_list[0].transform.location.x - wp_list[1].transform.location.x,
                      wp_list[0].transform.location.y - wp_list[1].transform.location.y)

            for (k, waypoint) in enumerate(wp_list, start=1):
                if k == len(wp_list):
                    spawn_straight(carla.Vector3D(0, 0, 0),
                                   waypoint.transform, 2)
                else:
                    r = (1.0/k-0.55)
                    right_wpt = waypoint.get_right_lane()

                    if right_wpt and right_wpt.lane_type == carla.LaneType.Driving:
                        vec = carla.Vector3D(-vector[1]*r, vector[0]*r, 0)
                    else:
                        vec = carla.Vector3D(vector[1]*r, -vector[0]*r, 0)

                    spawn_straight(vec, waypoint.transform, 0)

    return obstacle_list


def generate_parking(world, n, map, scenario_tag):

    blueprint_library = world.get_blueprint_library()
    intersection_coordinators = scenario_tag_to_location(map)
    spawn_transform = [carla.Transform(location=carla.Location(
        *intersection_coordinators[scenario_tag]))]
    parking_list = []

    motor_list = ['bh.crossbike', 'yamaha.yzf', 'vespa.zx125',
                  'gazelle.omafiets', 'diamondback.century', 'harley-davidson.low_rider']
    vehicle_list = []

    for i in blueprint_library.filter('vehicle.*'):
        if (i.id)[8:] not in motor_list:
            vehicle_list.append(i.id)

    def dist(t, L, limit=7):
        for trans in L:
            if trans.location.distance(t.location) < limit:
                return False
        return True

    trans_list = []
    all_wp = world.get_map().generate_waypoints(8)  # all_wp
    random.shuffle(all_wp)
    # print('All waypoints: ', len(all_wp))

    num = 0
    for idx, wp in enumerate(all_wp):

        if idx >= n:
            break

        waypoint = world.get_map().get_waypoint(location=wp.transform.location,
                                                lane_type=carla.LaneType.Shoulder)  # Shoulder Driving
        if not dist(waypoint.transform, trans_list, limit=7) or dist(waypoint.transform, spawn_transform, limit=60):
            continue

        vector = waypoint.transform.get_right_vector()
        r = random.choice([0, 0.6, 1, 1.5, 1.8, 1.9])

        cur_location = carla.Location(
            x=waypoint.transform.location.x-r*vector.x, y=waypoint.transform.location.y-r*vector.y, z=waypoint.transform.location.z+1)
        # cur_location = carla.Location(
        #     x=waypoint.transform.location.x, y=waypoint.transform.location.y, z=waypoint.transform.location.z+1)
        cur_trans = carla.Transform(cur_location, carla.Rotation(
            pitch=0, yaw=waypoint.transform.rotation.yaw, roll=0))

        try:

            random_vehicle = random.choice(vehicle_list)
            vehicle = blueprint_library.filter(random_vehicle)[0]
            # vehicle = random.choice(blueprint_library.filter('vehicle.*'))
            actor = world.spawn_actor(vehicle, cur_trans)

            print(num, actor.type_id, actor.id)
            num += 1

            parking_attr = {"obstacle_type": actor.type_id,
                            "basic_id": actor.id,
                            "location": {"x": cur_trans.location.x, "y": cur_trans.location.y, "z": cur_trans.location.z},
                            "rotation": {"pitch": cur_trans.rotation.pitch, "yaw": cur_trans.rotation.yaw, "roll": cur_trans.rotation.roll}}
            parking_list.append(parking_attr)

            trans_list.append(waypoint.transform)

        except Exception:
            pass

    return parking_list


def scenario_tag_to_location(town):

    if town == "Town01":
        intersection_coordinators = {
            't-1':
            (336, 327, 0.0),
            't-2':
            (336, 2, 0),
            't-3':
            (336, 197, 0.0),
            't-4':
            (336, 132, 0.0),
            't-5':
            (336, 58, 0.0),
            't-8':
            (156, 56, 0.0),
            't-9':
            (157, 1, 0.0),
            't-10':
            (91, 195.8, 0.0),
            't-11':
            (90.6, 56, 0.0),
            't-12':
            (90.6, 328, 0.0),
            't-13':
            (90.6, 2, 0.0),
            't-14':
            (91.6, 130.6, 0.0),

            # straight
            's-1':
            (221, 328, 0.0),
            's-2':
            (336, 255, 0.0),
            's-3':
            (225, 195, 0.0),
            's-4':
            (217, 130, 0.0),
            's-5':
            (250, 56, 0.0),
            's-6':
            (253, 1, 0.0),
            's-7':
            (1, 2, 0.0),
            's-8':
            (0, 163, 0.0),
            's-9':
            (393, 174, 0.0),
            's-10':
            (391, 1, 0.0)
        }

    elif town == "Town02":
        intersection_coordinators = {
            't-1':
            (-3, 190, 0),
            't-2':
            (44, 189, 0),
            't-3':
            (45, 248, 0.0),
            't-4':
            (43, 303, 0.0),
            't-5':
            (134, 190, 0.0),
            't-6':
            (134, 238, 0.0),
            't-7':
            (191, 191, 0.0),
            't-8':
            (191, 237, 0.0),

            # straight
            's-1':
            (90, 107, 0.0),
            's-2':
            (-4, 147, 0.0),
            's-3':
            (87, 190, 0.0),
            's-4':
            (-4, 249, 0.0),
            's-5':
            (126, 305, 0.0),
            's-6':
            (135, 216, 0.0),
            's-7':
            (45, 273, 0.0)
        }

    elif town == 'Town03':
        intersection_coordinators = {
            'i-1': (-80, 131, 0),
            'i-2': (-2, 132, 0),
            'i-4': (-78, -138, 0),
            'i-5': (3, -138, 0),

            't-2': (169.0, 64, 0),
            't-3': (232, 60, 0),
            't-6': (225, 2, 0),
            't-7': (9.8, -200, 0, 0),
            't-8': (83, -200, 0),
            't-10': (82, -74, 0),
            't-11': (151, -71, 0),

            # straight
            's-1': (-79.7, 69.9, 0),
            's-2': (-80.1, -69.2, 0),
            's-3': (233.4, -179.3, 0),
            's-4': (219.3, 175.0, 0),
            's-5': (-1.6,  80.3, 0),
            's-6': (1.6,  -81.3, 0),
            's-7': (114.5, 9.8, 0),
            's-8': (120.8, 130.8, 0),
        }

    elif town == "Town04":
        intersection_coordinators = {
            'i-1':
            (258.7, -248.2, 0.0),
            'i-2':
            (313.7, -247.2, 0.0),
            'i-3':
            (310.3, -170.5, 0.0),
            'i-4':
            (349.5, -170.5, 0.0),
            'i-5':
            (255.0, -170.2, 0.0),
            'i-6':
            (313.0, -119.7, 0.0),
            'i-7':
            (200.3, -173.6, 0.0),
            'i-8':
            (293.4, -247.7, 0.0),
            'i-9':
            (203.8, -309.0, 0.0),

            't-1':
            (256.3, -309.0, 0.0),
            't-2':
            (130.5, -172.7, 0.0),
            't-3':
            (61.9, -172.3, 0.0),
            't-4':
            (21.2, -172.2, 0.0),
            't-5':
            (256.5, -122.2, 0.0),

            # straight
            's-1': (-489.0, 58.4, 0),
            's-2': (-216.7, -97.8, 0),
            's-3': (211.0,  -390.7, 0),
            's-4': (183.5,  205.7, 0),
            's-5': (-227.7, 420.3, 0),
            's-6': (-445.5, 372.0, 0),
            's-7': (-222.8, 22.2, 0),
            's-8': (91.9,   -280.5, 0),
            's-9': (144.3,  -223.7, 0),
            's-10': (256.5, -206.8, 0),
            's-11': (251.6, 24.9, 0),
            's-12': (-3.7,  265.8, 0),
        }

    elif town == "Town05":
        intersection_coordinators = {
            'i-1':
            (-189.3, -89.3, 0.0),
            'i-2':
            (-190.5, 1.3, 0.0),
            'i-3':
            (-189.7, 89.6, 0.0),
            'i-6':
            (-125.6, 90.4, -0.0),
            'i-5':
            (-126.2, 0.3, -0.0),
            'i-14':
            (-126, -89.1, -0.0),
            'i-7':
            (-50.5, -89.4, -0.0),
            'i-8':
            (-49.1, 0.8, 0.0),
            'i-9':
            (-50.3, 89.6, 0.0),
            'i-12':
            (29.9, 90.0, 0.0),
            'i-11':
            (29.9, 0.2, 0.0),
            'i-10':
            (31.7, -89.2, 0.0),
            'i-13':
            (101.9, -0.1, 0.0),

            't-2':
            (-270, 1, 0.0),
            't-4':
            (-127, -140, 0.0),
            't-5':
            (-125, 146, 0.0),
            't-6':
            (35, -148, 0.0),
            't-7':
            (31, 142, 0.0),
            't-8':
            (31, 195, 0.0),
            't-9':
            (152, -0.3, 0.0),
            't-10':
            (30, 196, 0.0),

            # straight
            's-1': (197.5, 1.4, 0),
            's-2': (-216.9, -166.2, 0),
            's-3': (137.0, 124.3, 0),
            's-4': (84.7, -79.6, 0),
            's-5': (-90.5, 1.1, 0),
        }

    elif town == "Town07":
        intersection_coordinators = {
            'i-1':
            (-103, 53, 0.0),
            'i-2':
            (-3, -3, 0),
            'i-3':
            (-150.7, -35, 0.0),

            't-1':
            (-152, 50, 0),
            't-2':
            (66, 60, 0),
            't-3':
            (-100, -0.5, 0),
            't-4':
            (-200, -35, 0),
            't-5':
            (-3, -160, 0),
            't-6':
            (-100, -96, 0),
            't-7':
            (-100, -62, 0),
            't-8':
            (-100, -34, 0),
            't-9':
            (-199, -161, 0),
            't-10':
            (-2, -240, 0),
            't-11':
            (-5, 60, 0),
            't-12':
            (66, -1, 0),
            't-13':
            (-200, 49, 0),
            't-14':
            (-110, 114, 0),

            # straight
            's-1': (67.5, -126.1, 0),
            's-2': (-30.3, 122.3, 0),
            's-3': (-200.3, -90.1, 0),
            's-4': (-189.4, -244.8, 0),
            's-5': (-52.9, 58.6, 0),
            's-6': (-54.0, -65.6, 0),
        }

    elif town == "Town10HD":
        intersection_coordinators = {
            'i-1':
            (-46.0, 18.8, 0.0),
            't-2':
            (42.5, 63.3, 0.0),
            't-3':
            (42.4, 29.1, 0.0),
            't-1':
            (102.3, 21.4, 0.0),
            't-5':
            (-42.0, 69.0, 0.0),
            # 't4':
            # (-47.0, 127.9, 0.0),
            't-7':
            (-106.2, 21.8, 0.0),
            't-6':
            (-47.0, -60.2, 0.0),

            # straight
            's-1':
            (90, 118, 0.0),
            's-2':
            (62, -66, 0.0),
            's-3':
            (-100, -49, 0.0),
            's-4':
            (-98, 105, 0.0),
            's-5':
            (27, 12, 0.0),
            's-6':
            (-46, -23, 0.0)
        }

    elif town == "A1":
        intersection_coordinators = {
            'r-1':
            (-15.9, -24.7, 0.0),
        }

    elif town == "A6":
        intersection_coordinators = {
            'r-1':
            (20.1, 1.3, 0.0),
        }

    return intersection_coordinators


def scenario_id_exist(scenario_tag, map):

    intersection_coordinators = scenario_tag_to_location(map)
    return scenario_tag in intersection_coordinators


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        stored_path = 'data_collection'
        if not os.path.exists(stored_path):
            os.mkdir(stored_path)
        stored_path = os.path.join('data_collection', args.scenario_type)
        if not os.path.exists(stored_path):
            os.mkdir(stored_path)

        hud = HUD(args.width, args.height)
        world = World(client.load_world(args.map), hud, args)

        if args.controller == 'wheel':  # controller type
            controller = DualControl(world, args.autopilot)
        else:
            controller = KeyboardControl(world, args.autopilot)

        obstacle_list = []

        if args.obstacle != 0:
            obstacle_list = generate_obstacle(
                client.get_world(), args.obstacle, args.area, args.map, args.scenario_tag)
        if args.parking != 0:
            obstacle_list = generate_parking(
                client.get_world(), args.parking, args.map, args.scenario_tag)

        lights = []
        actors = world.world.get_actors()
        for l in actors:
            if 5 in l.semantic_tags and 18 in l.semantic_tags:
                lights.append(l)
        actor_dict = {}
        control_dict = {}
        timestamp_list = []
        traffic_light = dict()
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(40)
            code = controller.parse_events(client, world, clock)
            # exception
            if controller.r == 1:
                return
            # k_r click
            elif code == 3:
                timestamp_list.append(
                    [client.get_world().wait_for_tick().frame, time.time()])
                actor_dict = record_transform(actor_dict, world)
                actor_dict = record_velocity(actor_dict, world)
                control_dict = record_ped_control(control_dict, world)
                traffic_light = record_traffic_lights(traffic_light, lights)
            # stop recording
            elif code == 4:
                scenario_name = world.camera_manager.scenario_id
                stored_path = os.path.join(stored_path, scenario_name)
                if not os.path.exists(stored_path):
                    os.mkdir(stored_path)
                start_time = timestamp_list[0]
                end_time = timestamp_list[-1]
                print('start time: ' + str(start_time))
                print('end time: ' + str(end_time))
                actor_dict, control_dict, timestamp_list = save_actor(
                    stored_path, actor_dict, control_dict, timestamp_list, obstacle_list)
                save_traffic_lights(stored_path, traffic_light)
                save_description(stored_path, args.scenario_type,
                                 scenario_name, args.map)
                controller.r = 2

                print('has finished saving')
                actor_dict = {}
                control_dict = {}
                timestamp_list = []
                traffic_light = dict()
                stored_path = os.path.join(
                    'data_collection', args.scenario_type)

            # not recording
            elif code == 6:
                actor_dict = {}
                timestamp_list = []
                traffic_light = dict()
                actor_dict, control_dict = extract_actor(
                    actor_dict, control_dict, world)
            else:
                actor_dict, control_dict = extract_actor(
                    actor_dict, control_dict, world)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


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
        '-m', '--map',
        default='Town03',
        type=str,
        help='map name')
    argparser.add_argument(
        '--controller',
        default='keyboard',
        type=str,
        help='controller: keyboard, wheel (default: keyboard)')
    argparser.add_argument(
        '-obstacle',
        type=int,
        default=0,
        help='add obstacle')
    argparser.add_argument(
        '-area',
        type=int,
        default=10,
        help='spawn obstacle area')
    argparser.add_argument(
        '-parking',
        type=int,
        default=0,
        help='add parking')
    argparser.add_argument(
        '--scenario_type',
        type=str,
        choices=['interactive', 'collision', 'obstacle', 'non-interactive'],
        required=True,
        help='enable roaming actors')
    argparser.add_argument(
        '--scenario_tag',
        default="None",
        type=str,
        help='enable roaming actors')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    if args.scenario_type == "obstacle":
        if not scenario_id_exist(args.scenario_tag, args.map):
            print(f'scenario_tag \'{args.scenario_tag}\' does not exist!!')
            exit()

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
