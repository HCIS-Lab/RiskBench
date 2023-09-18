import random
import torch
from dataset import CarlaDataset
from converter import Converter
from map_model import MapModel
import common
import pathlib
import uuid
import copy
from collections import deque
# from controller import VehiclePIDController
import cv2

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
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

        # if DEBUG:
        #     import cv2

        #     canvas = np.ones((100, 100, 3), dtype=np.uint8)
        #     w = int(canvas.shape[1] / len(self._window))
        #     h = 99

        #     for i in range(1, len(self._window)):
        #         y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
        #         y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

        #         cv2.line(
        #                 canvas,
        #                 ((i-1) * w, int(y1 * h)),
        #                 ((i) * w, int(y2 * h)),
        #                 (255, 255, 255), 2)

        #     canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

        #     cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
        #     cv2.waitKey(1)

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LBCAgent(object):
    net = MapModel.load_from_checkpoint('/home/momoparadox/LBC/epoch=34_01.ckpt')
    '''
    This is an agent using Learning-By-Cheating end-to-end model,
    you'll have to initialize some parameters to load your model and setup your starting point.
    '''
    def __init__(self, world):
        self.world = world
    
    def spawn_at(self, wp):
        blueprint = random.choice(get_actor_blueprints(self.world, 'vehicle.*', '2'))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        

