import os
import math
import carla
import numpy as np




# data_generator


def read_traffic_lights(path, lights):
    
    path = os.path.join(path, 'traffic_light')
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    light_dict = dict()
    light_transform_dict = dict()
    for i, f in enumerate(files):
        l_id = int(f.split('.npy')[0])
        light_state = np.load(os.path.join(path, f), allow_pickle=True)
        new_light_state = []
        for iter, state in enumerate(light_state):
            if iter == 0:
                l_loc = carla.Location(float(state[0]), float(state[1]), float(state[2]))
            if state[0] == 'Red':
                new_light_state.append(carla.TrafficLightState.Red)
            elif state[0] == 'Yellow':
                new_light_state.append(carla.TrafficLightState.Yellow)
            elif state[0] == 'Green':
                new_light_state.append(carla.TrafficLightState.Green)
            elif state[0] == 'Off':
                new_light_state.append(carla.TrafficLightState.Off)
            else:
                new_light_state.append(carla.TrafficLightState.Unknown)

        min_d = 500.0
        light = None
        for new_l in lights:
            if new_l.get_location().distance(l_loc) < min_d:
                min_d = new_l.get_location().distance(l_loc)
                new_id = new_l.id
                light = new_l
        light_dict[new_id] = new_light_state
        light_transform_dict[light] = light.get_transform()

    return light_dict, light_transform_dict

def get_next_traffic_light(actor, world, light_transform_dict):

    location = actor.get_transform().location
    waypoint = world.get_map().get_waypoint(location)
    # Create list of all waypoints until next intersection
    list_of_waypoints = []
    while waypoint and not waypoint.is_intersection:
        list_of_waypoints.append(waypoint)
        waypoint = waypoint.next(2.0)[0]

    # If the list is empty, the actor is in an intersection
    if not list_of_waypoints:
        return None

    relevant_traffic_light = None
    distance_to_relevant_traffic_light = float("inf")

    for traffic_light, transform in light_transform_dict.items():
        if hasattr(traffic_light, 'trigger_volume'):
            tl_t = light_transform_dict[traffic_light]
            transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
            distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

            if distance < distance_to_relevant_traffic_light:
                relevant_traffic_light = traffic_light
                distance_to_relevant_traffic_light = distance

    return relevant_traffic_light

def get_trafficlight_trigger_location(traffic_light):    # pylint: disable=invalid-name
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

        return carla.Vector3D(x_, y_, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)

def annotate_trafficlight_in_group(ref, lights, world):
    """
    Get dictionary with traffic light group info for a given traffic light
    """
    if ref:
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # Get the waypoints
        ref_location = get_trafficlight_trigger_location(ref)
        ref_waypoint = world.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw


        for target_tl in lights:
            if ref.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = get_trafficlight_trigger_location(target_tl)
                target_waypoint = world.get_map().get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360

                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)
        return dict_annotations



def set_light_state(lights, light_dict, index, annotate):
    for l in lights:
        if l.id in light_dict:
            index = index if len(light_dict[l.id]) > index else -1
            if l in annotate['opposite']:
                state = light_dict[annotate['ref'][0].id][index]
            else:
                state = light_dict[l.id][index]
            l.set_state(state)
