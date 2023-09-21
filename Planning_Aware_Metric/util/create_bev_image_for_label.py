import os
import random
from six.moves import cPickle as pickle # for performance

import numpy as np
import ujson
from bird_eye_view.Mask import PixelDimensions, square_fitting_rect_at_any_rotation, MapMaskGenerator, RenderingWindow, BirdViewMasks, Coord, Loc, COLOR_OFF, COLOR_ON
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
import math
import cv2 



PIXELS_PER_METER = 5

def get_town( current_actor_path):
    if "/10_" in current_actor_path:
        Town = "Town10HD"
    elif "/7_" in current_actor_path:
        Town = "Town07"
    elif "/6_" in current_actor_path:
        Town = "Town06"
    elif "/5_" in current_actor_path:
        Town = "Town05"
    elif "/3_" in current_actor_path:
        Town = "Town03"
    elif "/2_" in current_actor_path:
        Town = "Town02"
    elif "/1_" in current_actor_path:
        Town = "Town01"     
    elif "/A0" in current_actor_path:
        Town = "A0"      
    elif "/A1" in current_actor_path:
        Town = "A1"   
    elif "/A6" in current_actor_path:
        Town = "A6"   
    elif "/B3" in current_actor_path:
        Town = "B3"     
    elif "/B7" in current_actor_path:
        Town = "B7"   
    elif "/B8" in current_actor_path:
        Town = "B8" 
        
    return Town

def get_scneario( current_actor_path):
    
    if "interactive" in current_actor_path:
        scenario = "interactive"
    elif "obstacle" in current_actor_path:
        scenario = "obstacle"
    elif "collision" in current_actor_path:
        scenario = "collision"
    else:
        scenario = "non-interactive"
        
    return scenario

def save_dict( di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict( filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

with open("./bird_eye_view/maps/ori_MapBoundaries.json", 'r') as f:
    map_data = ujson.load(f)
 
def carla_to_map(town, pos):
        
    min_x= map_data[town]["min_x"]
    min_y= map_data[town]["min_y"]

    min_point = np.array([min_x, min_y])
    world_point = np.array([pos[0], pos[1]])
    
    return 10.0 * ( world_point - min_point)
        


if __name__ == "__main__":
    dataset_dir = "./data_collection"

    all_data = []
    scenario_list = [ "interactive", "non-interactive", "collision", "obstacle"]
    for scenario in scenario_list:
        basic_scenario_list = os.listdir(os.path.join(dataset_dir, scenario))
        for basic_scenario in basic_scenario_list:
            variant_scenario_list = os.listdir(os.path.join(dataset_dir, scenario, basic_scenario, "variant_scenario"))
            for variant_scenario in variant_scenario_list:
                path = os.path.join(dataset_dir, scenario, basic_scenario, "variant_scenario", variant_scenario )
                all_data.append(path)
                break

    # print(len(all_data)) # total 7218

    # Vis obstacle and ego car trajectory in one MAP
    for path in all_data:

        town = get_town(path)
        scenario = get_scneario(path)
        
        
        frame_list = sorted( os.listdir(os.path.join(path, "ego_data")) ) 
        num_of_frame = len(frame_list)
        
        
        
        if scenario == "collision":
            with open(os.path.join(path,"collision_frame.json") ) as f:
                collision_frame_json = ujson.load(f)
                num_of_frame = collision_frame_json["frame"]
                
                
        ego_bbox = []
        
        
        with open(os.path.join(path, "actor_attribute.json"), 'rt') as f1:
            data = ujson.load(f1)

        ego_id = data["ego_id"]
        
        # if scneario is non-interactive, obstacle --> interactor_id is -1 
        interactor_id = data["interactor_id"] 
        # interactive id 
        vehicle_id_list = list(data["vehicle"].keys())
        # pedestrian id list 
        pedestrian_id_list = list(data["pedestrian"].keys())
        # obstacle id list 
        obstacle_id_list = list(data["obstacle"].keys())
        
        ill_parking_id_list = load_dict(f'{path}/ill_parking_id.pkl')["ill_parking_id"]
        
        obstacle_bbox_list = []
        # obstacle bbox store in actor_attribute.json
        for id in obstacle_id_list:
            pos_0 = carla_to_map(town, data["obstacle"][str(id)]["cord_bounding_box"]["cord_0"])
            pos_1 = carla_to_map(town, data["obstacle"][str(id)]["cord_bounding_box"]["cord_4"])
            pos_2 = carla_to_map(town, data["obstacle"][str(id)]["cord_bounding_box"]["cord_6"])
            pos_3 = carla_to_map(town, data["obstacle"][str(id)]["cord_bounding_box"]["cord_2"])
            
            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            
        vehicle_bbox_list = []
        
        ill_parking_bbox_list = []
        
    #  print(ill_parking_id)
        for id in vehicle_id_list:
            if str(id) != str(ego_id):
                index = 1
                
                file_path = os.path.join(path, "actors_data", f"{index:08}.json")
            
                with open(file_path, 'rt') as f1:
                    data = ujson.load(f1)
                    
                pos_0 = carla_to_map(town, data[str(id)]["cord_bounding_box"]["cord_0"])
                pos_1 = carla_to_map(town, data[str(id)]["cord_bounding_box"]["cord_4"])
                pos_2 = carla_to_map(town, data[str(id)]["cord_bounding_box"]["cord_6"])
                pos_3 = carla_to_map(town, data[str(id)]["cord_bounding_box"]["cord_2"])
                
                
                
                
                if id in ill_parking_id_list:
                    ill_parking_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
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
                


        ego_bbox_list = []
        for i in range(1, num_of_frame  ):
            file_path = os.path.join(path, "actors_data", f"{i:08}.json")
            
            with open(file_path, 'rt') as f1:
                data = ujson.load(f1)
                
                pos_0 = carla_to_map(town, data[str(ego_id)]["cord_bounding_box"]["cord_0"])
                pos_1 = carla_to_map(town, data[str(ego_id)]["cord_bounding_box"]["cord_4"])
                pos_2 = carla_to_map(town, data[str(ego_id)]["cord_bounding_box"]["cord_6"])
                pos_3 = carla_to_map(town, data[str(ego_id)]["cord_bounding_box"]["cord_2"])
                
                ego_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                    Loc(x=pos_1[0], y=pos_1[1]), 
                                    Loc(x=pos_2[0], y=pos_2[1]), 
                                    Loc(x=pos_3[0], y=pos_3[1]), 
                                    ])
                
                
        # print(len(ill_parking_bbox_list) )
                
                
                
        ### draw BEV image for label 
        
            
        canvas = cv2.imread(f"./bird_eye_view/maps/{town}/map.png")
        counter = 0
        for corners in ego_bbox_list:
            corners = [loc for loc in corners]
            cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=(255, 0, counter))
            counter+=2
            
        for corners in obstacle_bbox_list:
            corners = [loc for loc in corners]
            cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=(0, 0, 255))
            
            
        for corners in vehicle_bbox_list:
            corners = [loc for loc in corners]
            cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=(0, 255, 0))    
            
        for corners in ill_parking_bbox_list:
            corners = [loc for loc in corners]
            cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=(0, 0, 255))    
            
            
            
        
        
        
        
        
        # draw bbox 

        
        
        basic_scenario = path.split("/")[3]

        if not os.path.exists("./label"):
            os.makedirs("./label")
        
        
        cv2.imwrite(f"./label/{scenario}#{basic_scenario}.png", canvas)
        
        
        
            
