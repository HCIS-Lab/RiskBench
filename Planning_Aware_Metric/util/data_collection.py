import json
import numpy as np
import carla 
from carla import ColorConverter as cc
import math
import os
from multiprocessing import Process
import time
import cv2

class Data_Collection():
    def __init__(self) -> None:

        self.scenario_type = "interactive"
        self.gt_interactor = -1
        self.rgb_front = []
        self.rgb_left = []
        self.rgb_right = []
        self.ss_front = []
        self.ss_left = []
        self.ss_right = []
        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        self.sensor_lidar = []
        self.ss_top = []
        self.frame_list = []
        self.data_list = []
        self.sensor_data_list = []
        self.ego_list = []
        self.topology_list = []
        self.static_dict = {}
        self.compass = 0
        self.actor_attri_dict = {}

    def set_attribute(self, scenario_type, scenario_id, weather, actor, random_seed, map):
        self.scenario_type = scenario_type
        self.scenario_id = scenario_id
        self.weather = weather
        self.actor = actor
        self.seed = random_seed
        self.map = map

    def set_start_frame(self, frame):
        self.start_frame = frame

    def set_end_frame(self, frame):
        self.end_frame = frame

    def set_scenario_type(self, sceanrio):
        self.scenario_type = sceanrio

    def set_ego_id(self, world):
        self.ego_id = world.player.id

    def set_gt_interactor(self, id):
        self.gt_interactor = id

    def collect_sensor(self, frame, world):

        while True:
            if world.camera_manager.ss_top.frame == frame:

                self.ss_top.append(world.camera_manager.ss_top)
                break
        while True:
            if world.camera_manager.rgb_front.frame == frame:
                self.rgb_front.append(world.camera_manager.rgb_front)
                break

        while True:
            if world.camera_manager.ss_front.frame == frame:
                self.ss_front.append(world.camera_manager.ss_front)
                break

        # depth
        while True:
            if world.camera_manager.depth_front.frame == frame:
                self.depth_front.append(world.camera_manager.depth_front)
                break

        while True:
            if world.camera_manager.lidar.frame == frame:
                self.sensor_lidar.append(world.camera_manager.lidar)
                break

        while True:
            if world.imu_sensor.frame == frame:
                self.compass = world.imu_sensor.compass
                break

        # store all actor
        self.frame_list.append(frame)

        self.sensor_data_list.append(self.collect_camera_data(world))

        data = self.collect_actor_data(world)


        self.data_list.append(data)
        


        self.ego_list.append(data[self.ego_id])
        self.topology_list.append(self.collect_topology(world))

    def collect_actor_attr(self, world):
        # Here we get all actor attributes
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
                return {"x": x, "y": y, "z": z}

        ego_id = {}
        _id = world.player.id
        ego_type_id = world.player.type_id
        ego_semantic_tags = world.player.semantic_tags
        ego_attributes = world.player.attributes
        ego_bbox = world.player.bounding_box
        ego_bounding_box = {"extent": get_xyz(ego_bbox.extent), "location": get_xyz(
            ego_bbox.location)}

        ego_id[_id] = {}
        ego_id[_id]["type_id"] = ego_type_id
        ego_id[_id]["semantic_tags"] = ego_semantic_tags
        ego_id[_id]["attributes"] = ego_attributes
        ego_id[_id]["bounding_box"] = ego_bounding_box

        vehicle_ids = {}
        pedestrian_ids = {}
        traffic_light_ids = {}
        obstacle_ids = {}

        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes
            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}

            vehicle_ids[_id] = {}
            vehicle_ids[_id]["type_id"] = type_id
            vehicle_ids[_id]["semantic_tags"] = semantic_tags
            vehicle_ids[_id]["attributes"] = attributes
            vehicle_ids[_id]["bounding_box"] = bounding_box

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes
            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}

            pedestrian_ids[_id] = {}
            pedestrian_ids[_id]["type_id"] = type_id
            pedestrian_ids[_id]["semantic_tags"] = semantic_tags
            pedestrian_ids[_id]["attributes"] = attributes
            pedestrian_ids[_id]["bounding_box"] = bounding_box

        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            traffic_light_ids[_id] = {}
            traffic_light_ids[_id]["type_id"] = type_id
            traffic_light_ids[_id]["semantic_tags"] = semantic_tags
            traffic_light_ids[_id]["location"] = location
            traffic_light_ids[_id]["rotation"] = rotation
            traffic_light_ids[_id]["bounding_box"] = bounding_box
            traffic_light_ids[_id]["cord_bounding_box"] = cord_bounding_box

        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location), "rotation": get_xyz(bbox.rotation, True)}

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            obstacle_ids[_id] = {}
            obstacle_ids[_id]["type_id"] = type_id
            obstacle_ids[_id]["semantic_tags"] = semantic_tags
            obstacle_ids[_id]["attributes"] = attributes
            obstacle_ids[_id]["location"] = location
            obstacle_ids[_id]["rotation"] = rotation
            obstacle_ids[_id]["bounding_box"] = bounding_box
            obstacle_ids[_id]["cord_bounding_box"] = cord_bounding_box

        self.actor_attri_dict = {"vehicle": vehicle_ids,
                                 "pedestrian": pedestrian_ids,
                                 "traffic_light": traffic_light_ids,
                                 "obstacle": obstacle_ids,
                                 "ego_id": world.player.id,
                                 "interactor_id": self.gt_interactor}

    def collect_topology(self, get_world):
        town_map = get_world.world.get_map()
        try:
            while True:
                if get_world.abandon_scenario:
                    print('Abandom, killing thread.')
                    return
                waypoint = town_map.get_waypoint(
                    get_world.player.get_location())
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
                                turn_direction = "right"  # right
                            elif (after_yaw < before_yaw):
                                turn_direction = "left"  # left
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
                    lane_feature_ls.append(
                        [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (i, j)])
                # print("topology collection finished")
                return lane_feature_ls

        except:
            print("topology collection error.")
        pass

    def save_collision_frame(self, frame, collision_id, type, path):


        # path 
        if (frame >= self.start_frame):
            frame = frame - self.start_frame

            path = f'{path}/collision_frame.json'

            collision_dict =  {}
            collision_dict["frame"] = frame
            collision_dict["id"] = collision_id
            collision_dict["type"] = type

            f = open(path, "w")
            json.dump(collision_dict, f, indent=4)
            f.close()

    def collect_static_actor_data(self, world):
        id_counter = 0
        data = {}

        static = world.world.get_level_bbs(carla.CityObjectLabel.Car)
        for bbox in static:
            _id = id_counter
            id_counter += 1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Car"
        data["num_of_id"] = id_counter

        static = world.world.get_level_bbs(carla.CityObjectLabel.Truck)
        for bbox in static:
            _id = id_counter
            id_counter += 1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Truck"

        static = world.world.get_level_bbs(carla.CityObjectLabel.Bus)
        for bbox in static:
            _id = id_counter
            id_counter += 1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Bus"

        static = world.world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
        for bbox in static:
            _id = id_counter
            id_counter += 1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Motorcycle"

        static = world.world.get_level_bbs(carla.CityObjectLabel.Bicycle)
        for bbox in static:
            _id = id_counter
            id_counter += 1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Bicycle"

        data["num_of_id"] = id_counter

        self.static_dict = data

    def collect_actor_data(self, world):

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
            rotation = get_xyz(actor.get_transform().rotation, True)

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

            
            vehicles_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            v = actor.get_velocity()

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

        pedestrian_id_list = []

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            
            pedestrian_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
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

        traffic_id_list = []

        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:

            _id = actor.id

            traffic_light_state = int(actor.state)  # traffic light state
            actor_loc = actor.get_location()
            distance = ego_loc.distance(actor_loc)

            
            traffic_id_list.append(_id)

            data[_id] = {}
            data[_id]["state"] = traffic_light_state
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            data[_id]["location"] = location
            data[_id]["distance"] = distance
            data[_id]["type"] = "traffic_light"

            trigger = actor.trigger_volume
            verts = [v for v in trigger.get_world_vertices(carla.Transform())]

            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id]["tigger_cord_bounding_box"] = cord_bounding_box
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["trigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y]

        obstacle_id_list = []

        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:

            _id = actor.id

            actor_loc = actor.get_location()
            distance = ego_loc.distance(actor_loc)

            obstacle_id_list.append(_id)

            data[_id] = {}
            data[_id]["distance"] = distance
            data[_id]["type"] = "obstacle"

        data["obstacle_ids"] = obstacle_id_list
        data["traffic_light_ids"] = traffic_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list

        return data # , obstacle_id_list, traffic_id_list, pedestrian_id_list

    def _get_forward_speed(self, transform, velocity):
        """ Convert the vehicle transform directly to forward speed """

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def collect_camera_data(self, world):

        data = {}

        intrinsic = np.identity(3)
        intrinsic[0, 2] = 640 / 2.0
        intrinsic[1, 2] = 256 / 2.0
        intrinsic[0, 0] = intrinsic[1, 1] = 640 / (
            2.0 * np.tan(120 * np.pi / 360.0)
        )
        # sensor_location
        data["front"] = {}
        data["front"]["extrinsic"] = world.camera_manager.sensor_rgb_front.get_transform(
        ).get_matrix()  # camera 2 world
        data["front"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_rgb_front
        data["front"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["front"]["w2c"] = np.array(
            world.camera_manager.sensor_rgb_front.get_transform().get_inverse_matrix())
        
        intrinsic = np.identity(3)
        intrinsic[0, 2] = 512 / 2.0
        intrinsic[1, 2] = 512 / 2.0
        intrinsic[0, 0] = intrinsic[1, 1] = 512 / (
            2.0 * np.tan(50 * np.pi / 360.0)
        )

        data["top"] = {}
        data["top"]["extrinsic"] = world.camera_manager.sensor_ss_top.get_transform(
        ).get_matrix()
        data["top"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_ss_top
        data["top"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["top"]["w2c"] = np.array(
            world.camera_manager.sensor_ss_top.get_transform().get_inverse_matrix())

        return data

    def save_json_data(self, frame_list, data_list, path, start_frame, end_frame, folder_name):

        counter = 0
        stored_path = os.path.join(path, folder_name)
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        for idx in range(len(frame_list)):
            frame = frame_list[idx]
            data = data_list[idx]
            if (frame >= start_frame) and (frame < end_frame):
                frame = frame - start_frame
                counter += 1
                actors_data_file = stored_path + ("/%08d.json" % frame)
                f = open(actors_data_file, "w")
                json.dump(data, f, indent=4)
                f.close()

        print(folder_name + " save finished. Total: ", counter)

    def save_np_data(self, frame_list, data_list, path, start_frame, end_frame, folder_name):

        counter = 0

        stored_path = os.path.join(path, folder_name)
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        for idx in range(len(frame_list)):

            frame = frame_list[idx]
            data = data_list[idx]
            if (frame >= start_frame) and (frame < end_frame):
                frame = frame-start_frame

                counter += 1
                sensor_data_file = stored_path + ("/%08d.npy" % frame)
                np.save(sensor_data_file, np.array(data, dtype=object))

        print(folder_name + " save finished. Total: ", counter)

    def save_img(self, img_list, sensor, path, start_frame, end_frame, view='top'):

        sensors = [
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

        modality = sensors[sensor][0].split('.')[-1]
        counter = 0
        for img in img_list:
            if (img.frame >= start_frame) and (img.frame < end_frame):

                counter += 1
                frame = img.frame - start_frame

                if 'seg' in modality:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, frame), cc.Raw)
                elif 'depth' in modality:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, frame), cc.Raw)#cc.Depth)  # cc.LogarithmicDepth
                elif 'lidar' in view:
                    points = np.frombuffer(img.raw_data, dtype=np.dtype('f4'))
                    points = np.reshape(points, (int(points.shape[0] / 4), 4))
                    if not os.path.exists('%s/%s/' % (path, view)):
                        os.makedirs('%s/%s/' % (path, view))
                    np.save('%s/%s/%08d.npy' %
                            (path, view, frame), points, allow_pickle=True)
                else:
                    img.convert(cc.Raw)
                    array = np.frombuffer(
                        img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    array = array[:, :, :3]
                    # array = array[:, :, ::-1]
                    if not os.path.exists('%s/%s/%s/' % (path, modality, view)):
                        os.makedirs('%s/%s/%s/' % (path, modality, view))
                    cv2.imwrite('%s/%s/%s/%08d.jpg' % (path, modality,
                                view, frame), array, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print("%s %s save finished. Total: %d" %
              (sensors[sensor][2], view, counter))

    def save_data(self, path):

        t_ss_top = Process(target=self.save_img, args=(self.ss_top, 10, path,
                                                       self.start_frame, self.end_frame, 'top'))

        t_rgb_front = Process(target=self.save_img, args=(self.rgb_front, 0, path,
                              self.start_frame, self.end_frame, 'front'))
        t_ss_front = Process(target=self.save_img, args=(
            self.ss_front, 10, path, self.start_frame, self.end_frame, 'front'))
        
        t_depth_front = Process(target=self.save_img, args=(self.depth_front, 1, path,
                                self.start_frame, self.end_frame, 'front'))
        t_lidar = Process(target=self.save_img, args=(self.sensor_lidar, 6,
                          path, self.start_frame, self.end_frame, 'lidar'))
        t_actors_data = Process(target=self.save_json_data, args=(
            self.frame_list, self.data_list, path, self.start_frame, self.end_frame, "actors_data"))
        t_sensor_data = Process(target=self.save_np_data, args=(
            self.frame_list, self.sensor_data_list, path, self.start_frame, self.end_frame, "sensor_data"))
        t_ego_data = Process(target=self.save_json_data, args=(
            self.frame_list, self.ego_list, path, self.start_frame, self.end_frame, "ego_data"))

        t_topology = Process(target=self.save_np_data, args=(
            self.frame_list, self.topology_list, path, self.start_frame, self.end_frame, "topology"))

        start_time = time.time()

        t_ss_top.start()
        t_rgb_front.start()
        t_ss_front.start()
        t_depth_front.start()
        t_lidar.start()
        t_actors_data.start()
        t_sensor_data.start()
        t_ego_data.start()
        t_topology.start()
        # ------------------------------ #
        t_ss_top.join()
        t_rgb_front.join()
        t_ss_front.join()
        t_depth_front.join()
        t_lidar.join()
        t_actors_data.join()
        t_sensor_data.join()
        t_ego_data.join()
        t_topology.join()

        with open(f"{path}/static_data.json", "w") as f:
            json.dump(self.static_dict, f, indent=4)
            f.close()
            print("static_data save finished.")

        with open(f"{path}/actor_attribute.json", "w") as f:
            json.dump(self.actor_attri_dict, f, indent=4)
            f.close()
            print("actor attribute save finished.")

        with open("./result.txt", "a") as f:
            f.write(
                f"{self.scenario_type}#{self.scenario_id}#{self.map}#{self.weather}#{self.actor}#{self.seed}\n")

        end_time = time.time()

        print('ALL save done in %s ' % (end_time-start_time))
        print("")

        # empty list
        self.ss_top = []
        self.rgb_front = []
        self.ss_front = []
        self.depth_front = []
        self.ego_list = []
        self.sensor_lidar = []
        self.data_list = []
        self.sensor_data_list = []
        self.actor_attri_dict = {}
        self.frame_list = []
        self.static_dict = {}
        self.topology_list = []