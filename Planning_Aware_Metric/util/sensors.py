import carla
import weakref
import json
import os 
import math
from carla import ColorConverter as cc
import numpy as np
import pygame


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.other_actor_id = 0  # init as 0 for static object
        self.other_actor_ids = []  # init as 0 for static object
        self.wrong_collision = False

        self.true_collision = False
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

        self.collision_actor_id = None
        self.collision_actor_type = None

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
        self.history.append(
            {'frame': event.frame, 'actor_id': event.other_actor.id})
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        self.collision = True
        self.collision_actor_id = event.other_actor.id
        self.collision_actor_type = actor_type

        

        if event.other_actor.id in self.other_actor_ids:
            self.true_collision = True
        if event.other_actor.id == self.other_actor_id: 
            self.true_collision = True
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
        self.frame = 0

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
        self.frame = sensor_data.frame


    def toggle_recording_IMU(self):
        self.recording = not self.recording


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
    def __init__(self, parent_actor, hud, gamma_correction, save_mode, inference_mode):

        self.ss_top = None
        self.sensor_top = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode
        self.inference_mode = inference_mode
        
        self.rgb_front = None
        self.rgb_left = None
        self.rgb_right = None
        self.ss_front = None
        self.ss_left = None
        self.ss_right = None

        self.depth_front = None
        self.depth_left = None
        self.depth_right = None
        self.lidar = None

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
                 z=100.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),

                # sensor config for transfuser camera settings
                #  front view 8
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)), Attachment.Rigid),
                # left view  9
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-60.0)), Attachment.Rigid),
                # right view 10
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=60.0)), Attachment.Rigid),
                # rear 11
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=180.0)), Attachment.Rigid),
                # rear left 12
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-120.0)), Attachment.Rigid),
                # rear right 13
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=1.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=120.0)), Attachment.Rigid),
                # ins top 14
                (carla.Transform(carla.Location(x=0.0, y=0.0,
                 z=50.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid),

                # lidar 15 
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=1.3), carla.Rotation(yaw=-90.0)), Attachment.Rigid),
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
            ['sensor.camera.instance_segmentation', cc.Raw,
                'Camera Instance Segmentation (CityScapes Palette)', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        self.bev_seg_bp = bp_library.find(
            'sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(512))
        self.bev_seg_bp.set_attribute('image_size_y', str(512))
        self.bev_seg_bp.set_attribute('fov', str(50.0))
        
        
        # transfuser sensor setting

        self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        self.sensor_rgb_bp.set_attribute('image_size_x', str(960))
        self.sensor_rgb_bp.set_attribute('image_size_y', str(480))
        self.sensor_rgb_bp.set_attribute('fov', str(60.0))

        self.sensor_ss_bp = bp_library.find(
            'sensor.camera.instance_segmentation')
        self.sensor_ss_bp.set_attribute('image_size_x', str(960))
        self.sensor_ss_bp.set_attribute('image_size_y', str(480))
        self.sensor_ss_bp.set_attribute('fov', str(60.0))

        self.sensor_depth_bp = bp_library.find('sensor.camera.depth')
        self.sensor_depth_bp.set_attribute('image_size_x', str(960))
        self.sensor_depth_bp.set_attribute('image_size_y', str(480))
        self.sensor_depth_bp.set_attribute('fov', str(60.0))
        
        
        # hank 120 fov setting
        
        self.front_cam_bp = bp_library.find('sensor.camera.rgb')
        self.front_cam_bp.set_attribute('image_size_x', str(640))
        self.front_cam_bp.set_attribute('image_size_y', str(256))
        self.front_cam_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_cam_bp.set_attribute('focal_distance', str(500))
        if self.front_cam_bp.has_attribute('gamma'):
            self.front_cam_bp.set_attribute('gamma', str(gamma_correction))

        self.front_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.front_seg_bp.set_attribute('image_size_x', str(640))
        self.front_seg_bp.set_attribute('image_size_y', str(256))
        self.front_seg_bp.set_attribute('fov', str(120.0))
        self.front_seg_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_seg_bp.set_attribute('lens_circle_falloff', '0.0')

        self.depth_bp = bp_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute('image_size_x', str(640))
        self.depth_bp.set_attribute('image_size_y', str(256))
        self.depth_bp.set_attribute('fov', str(120.0))
        self.depth_bp.set_attribute('lens_circle_falloff', '0.0')

        self.sensor_lidar_bp = bp_library.find('sensor.lidar.ray_cast')
        self.sensor_lidar_bp.set_attribute('range', str(100))
        self.sensor_lidar_bp.set_attribute('rotation_frequency', str(20))
        self.sensor_lidar_bp.set_attribute('points_per_second', str(1200000))

        for item in self.sensors:

            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)

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

            self.sensor_top = self._parent.get_world().spawn_actor(
                self.sensors[0][-1],
                self._camera_transforms[6][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[6][1])
            
            if self.inference_mode:
                
                self.sensor_rgb_front = self._parent.get_world().spawn_actor(
                    self.front_cam_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_ss_front = self._parent.get_world().spawn_actor(
                    self.front_seg_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
            else:

                if self.save_mode:

                    # inst top
                    self.sensor_ss_top = self._parent.get_world().spawn_actor(
                        self.bev_seg_bp,
                        self._camera_transforms[14][0],
                        attach_to=self._parent,
                        attachment_type=self._camera_transforms[14][1])
                    # front

                    self.sensor_rgb_front = self._parent.get_world().spawn_actor(
                        self.front_cam_bp,
                        self._camera_transforms[8][0],
                        attach_to=self._parent,
                        attachment_type=self._camera_transforms[0][1])

                    self.sensor_ss_front = self._parent.get_world().spawn_actor(
                        self.front_seg_bp,
                        self._camera_transforms[8][0],
                        attach_to=self._parent,
                        attachment_type=self._camera_transforms[0][1])

                    self.sensor_depth_front = self._parent.get_world().spawn_actor(
                        self.depth_bp,
                        self._camera_transforms[8][0],
                        attach_to=self._parent,
                        attachment_type=self._camera_transforms[0][1])

                    # lidar sensor
                    self.sensor_lidar = self._parent.get_world().spawn_actor(
                        # self.sensors[6][-1],
                        self.sensor_lidar_bp,
                        self._camera_transforms[15][0],
                        attach_to=self._parent,
                        attachment_type=self._camera_transforms[0][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            # self.sensor_lbc_img.listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'lbc_img'))
            self.sensor_top.listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'top'))
            
            if self.inference_mode:
                
                self.sensor_rgb_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'rgb_front'))

                self.sensor_ss_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'ss_front'))

            else:
                if self.save_mode:

                    self.sensor_ss_top.listen(
                        lambda image: CameraManager._parse_image(weak_self, image, 'ss_top'))

                    self.sensor_rgb_front.listen(
                        lambda image: CameraManager._parse_image(weak_self, image, 'rgb_front'))

                    self.sensor_ss_front.listen(
                        lambda image: CameraManager._parse_image(weak_self, image, 'ss_front'))

                    self.sensor_depth_front.listen(
                        lambda image: CameraManager._parse_image(weak_self, image, 'depth_front'))

                    self.sensor_lidar.listen(
                        lambda image: CameraManager._parse_image(weak_self, image, 'lidar'))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

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

        elif view == 'ss_top':
            self.ss_top = image
        elif view == 'rgb_front':
            self.rgb_front = image
        elif view == 'rgb_left':
            self.rgb_left = image
        elif view == 'rgb_right':
            self.rgb_right = image
            self.rgb_rear_right = image
        elif view == 'depth_front':
            self.depth_front = image
        elif view == 'depth_left':
            self.depth_left = image
        elif view == 'depth_right':
            self.depth_right = image
        elif view == 'ss_front':
            self.ss_front = image
        elif view == 'ss_left':
            self.ss_left = image
        elif view == 'ss_right':
            self.ss_right = image
        elif view == 'lidar':
            self.lidar = image
