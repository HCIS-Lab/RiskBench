import sys
sys.path.append('/home/carla/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
sys.path.append('/home/carla/carla/PythonAPI')
sys.path.append('/home/carla/carla/PythonAPI/carla/')
sys.path.append('/home/carla/carla/PythonAPI/carla/agents')

import carla
import numpy as np
import time
import csv
import os
import logging
import math
import argparse
from numpy import random
from carla import VehicleLightState as vls
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
import json


def write_json(filename, index, seed ):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        y = {str(index):seed}
        file_data.update(y)
        file.seek(0)
        json.dump(file_data, file, indent = 4)


client = carla.Client('localhost', 2000)
# client.set_timeout(5.0)
def spawn_actor_nearby(args, store_path, distance=100, v_ratio=0.8, pedestrian=10, transform_dict={}): 


    # if args.replay:
    # P = store_path.split("/")
    # with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
    #     data = json.load(outfile)
    #     seed_4 = int(data["4"])
    # seed = 
    # get world and spawn points
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()

    """
    topo = map.get_topology()
    topo_dict = {}

    # iterate topology list to create topology dictionary 
    for edge in topo:
        if topo_dict.get(edge[0].id) == None:
            topo_dict[edge[0].id] = []

        topo_dict[edge[0].id].append(edge[1].id)
    """

    #get_spawn_points() get transform
    waypoint_list = []
    
    for waypoint in spawn_points:
        point = map.get_waypoint(waypoint.location)
        """
        d = waypoint.location.distance(actor_transform_list[0]['transform'][0].location)
        closer_node = 0
        next_nodes = pt.next(10)
        #distinguish whether the next waypoint is closer
        num_of_edges = len(next_nodes)
        for node in next_nodes:
            if node.transform.location.distance(actor_transform_list[0]['transform'][0].location) < d:
                closer_node += 1
        """
        flag = True
        
        mid = len(transform_dict['player']) // 2

        if (waypoint.location.distance(transform_dict['player'][mid].location)) < distance:
            flag = False
            
            for actor_id, traj in transform_dict.items():
                for pt in traj:
                    if waypoint.location.distance(pt.location) < 3:
                        flag = True
                        break
                if flag:
                    break

        if (waypoint.location.distance(transform_dict['player'][0].location)) < 5:
            flag = True

        if not flag:
            for i in range(50):
                next_pt = point.next(5)
                point = next_pt[0]
                for actor_id, traj in transform_dict.items():
                    interval = len(traj) // 50
                    lower = (i-1) * interval if i != 0 else i*interval
                    upper = (i+2) * interval if i != 49 else i+1*interval
                    for pt in traj[lower:upper]:
                        if point.transform.location.distance(pt.location) < 0.1:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    break
            if flag:
                break
        
        """
        if (waypoint.location.distance(center) < distance and 
            closer_node > num_of_edges//2) and (not flag):
            waypoint_list.append(waypoint)
        """

        if not flag:
            waypoint_list.append(waypoint)
            
    seed_4 = int(time.time()) 


    if args.replay:
        P = store_path.split("/")
        with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
            data = json.load(outfile)
            seed_4 = int(data["4"])
    else:
        write_json(store_path + "/random_seeds.json", 4, seed_4 )

    random.seed(seed_4)
    print("seed_4: ", seed_4)
    random.shuffle(waypoint_list)
    
    # print(len(waypoint_list))

    # --------------
    # Spawn vehicles
    # --------------
    num_of_vehicles = 0
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    traffic_manager = client.get_trafficmanager()
    # keep distance
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    traffic_manager.set_random_device_seed(10)
    
    synchronous_master = False
    vehicles_list = []
    walkers_list = []
    all_id = []

    batch = []

    # vehicle = min(math.ceil(len(waypoint_list) * 0.8), vehicles) if len(waypoint_list) >= 20 else len(waypoint_list)
    vehicle = math.ceil(len(waypoint_list) * v_ratio)
    for n, transform in enumerate(waypoint_list):
        if n >= vehicle:
            break
        num_of_vehicles += 1
        
        seed_5 = int(time.time()) 

        if args.replay:
            P = store_path.split("/")
            with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                data = json.load(outfile)
                seed_5 = int(data["5"])
        else:
            if num_of_vehicles == 1:
                write_json(store_path + "/random_seeds.json", 5, seed_5 )

        seed_5 += 5*num_of_vehicles



        print("seed_5: ", seed_5)
        random.seed(seed_5)
        # print(seed_5)
        # print(num_of_vehicles)
        # print("********************")
        blueprint = random.choice(blueprints)
        
        #print(blueprint)
        
        if blueprint.has_attribute('color'):

            seed_6 = int(time.time())
            
            if args.replay:
                P = store_path.split("/")
                with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                    data = json.load(outfile)
                    seed_6 = int(data["5"])
            else:

                if num_of_vehicles == 1:
                    write_json(store_path + "/random_seeds.json", 6, seed_6 )
            seed_6 += 6*num_of_vehicles
            print("seed_6: ", seed_6)
            random.seed(seed_6)
            
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            
            seed_7 = int(time.time())
            
            if args.replay:
                P = store_path.split("/")
                with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                    data = json.load(outfile)
                    seed_7 = int(data["5"])
            else:

                if num_of_vehicles == 1:
                    write_json(store_path + "/random_seeds.json", 7, seed_7 )
            seed_7 += 7*num_of_vehicles
            print("seed_7: ", seed_7)
            random.seed(seed_7)
            
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        light_state = vls.Position | vls.LowBeam | vls.LowBeam 

        #client.get_world().spawn_actor(blueprint, transform)
        # spawn the cars and set their autopilot and light state all together
        
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # print(f"Spawn {len(waypoint_list)} Vehicle")
    """
    Apply local planner to all vehicle
    Random choose destination and behavior type
    """
    """
    behavior_type = ['aggressive', 'normal', 'cautious']
    actors = client.get_world().get_actors()
    map = client.get_world().get_map()
    vehicles = []
    agents_map = {}

    # set attribute of vehicles
    for actor in actors:
        #print(actor.type_id)
        if actor.type_id[0:7] == 'vehicle':
            randtype = random.choice(behavior_type)
            agent = BehaviorAgent(actor, behavior="aggressive")
            #agent = BasicAgent(actor)

            vehicles.append(actor)
            agents_map[actor] = agent
            
            coord = actor.get_location()
            lane_now = map.get_waypoint(coord).lane_id
            agent.set_destination(coord, coord)

            # sp meams spawn point
            # find destination which in different lane with current location
            random.shuffle(waypoint_list)
            for sp in waypoint_list:
                if map.get_waypoint(sp.location).lane_id != lane_now:
                    agent.set_destination(coord, sp.location)
                    print("Successfully spawn {} actor with initial location {} and destination {}!".format(randtype, coord, sp.location))
                    break
    
    """

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 0.5      # how many pedestrians will run
    percentagePedestriansCrossing = 0.5     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    '''
    loc_list = []
    loc_dict = {}
    loc = world.get_random_location_from_navigation()
    count = 1
    
    while loc != None:
        count += 1
        print(loc)
        loc = world.get_random_location_from_navigation()
        if loc_dict.get(loc) == None:
            loc_dict[loc] = True
        else:
            print("overlap")
        loc_list.append(loc)
    print("count:", count) 
    '''
    loc_dict = {}
    for i in range(pedestrian):
        spawn_point = carla.Transform()
        
        flag = False
        
        # Number of try to find spawn points
        num_try = 100000
        # while True:
        for j in range(num_try):
            loc = world.get_random_location_from_navigation()
            temp = carla.Location(int(loc.x), int(loc.y), int(loc.z))
            if (loc.distance(transform_dict['player'][mid].location) < distance) and (loc_dict.get(temp) == None):
                loc_dict[temp] = True
                spawn_point.location = loc
                spawn_points.append(spawn_point)
                flag = True
                break
        #print("Pedestrian#", i, " spawn: ", flag)

        #if (loc != None):
            #spawn_point.location = loc
            #spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    n=0
    for spawn_point in spawn_points:
        
        n+=1
        seed_8 = int(time.time())
        if args.replay:
            P = store_path.split("/")
            with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                data = json.load(outfile)
                seed_8 = int(data["8"])
        else:

            if n == 1:
                write_json(store_path + "/random_seeds.json", 8, seed_8 )
        seed_8 += 8*n
        random.seed(seed_8)
        
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            seed_9 = int(time.time())


            if args.replay:
                P = store_path.split("/")
                with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
                    data = json.load(outfile)
                    seed_9 = int(data["9"])
            else:
                if n == 1:
                    write_json(store_path + "/random_seeds.json", 9, seed_9 )


            seed_9 += 9*n
            random.seed(seed_9)
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    # if not True or not synchronous_master:
    #     world.wait_for_tick()
    # else:
    #     world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point

        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        #all_actors[i].go_to_location(center)

        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(walkers_list)))
    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(10.0)
    return vehicles_list, all_actors,all_id
