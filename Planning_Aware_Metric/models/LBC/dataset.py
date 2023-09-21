import os
import random
from six.moves import cPickle as pickle # for performance
import torch
import numpy as np
import math
import ujson

from bird_eye_view.Mask import PixelDimensions, square_fitting_rect_at_any_rotation, MapMaskGenerator, RenderingWindow, BirdViewMasks, Coord, Loc, COLOR_OFF, COLOR_ON
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView




from PIL import Image
from torchvision import transforms

# Reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 10.
GAP = 5 #10 
STEPS = 4
# N_CLASSES = len(common.COLOR) #6  --> 10 classes 

N_CLASSES = 7


# 0 BirdViewMasks.UNLABELES
# 1 BirdViewMasks.ROAD
# 2 BirdViewMasks.ROAD_LINE
# 3 BirdViewMasks.VEHICLES
# 4 BirdViewMasks.PEDESTRIANS
# 5 BirdViewMasks.OBSTACLES
# 6 BirdViewMasks.AGENT



PIXELS_PER_METER = 5



def preprocess_semantic(topdown):
    
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


def make_heatmap(size, pt, sigma=8):
    img = np.zeros(size, dtype=np.float32)
    pt = [
            np.clip(pt[0], sigma // 2, img.shape[1]-sigma // 2),
            np.clip(pt[1], sigma // 2, img.shape[0]-sigma // 2)
            ]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def get_dataset(args, is_train):
    
    if is_train:    
        traindata = CarlaDataset(root_dir=args.dataset_dir, is_train=True)
        trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=True
        )
        return trainloader
    else:
        valdata = CarlaDataset(root_dir=args.dataset_dir, is_train=False)
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        return valloader


class CarlaDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # SECOND
    def __init__(self, root_dir, is_train):
        super(CarlaDataset, self).__init__()
        
        self.dataset_dir = root_dir
        self.is_train  = is_train

        if not os.path.exists("./train_val_dataset.pkl"):
            # scenario_list = ["interactive", "non-interactive", "collision", "obstacle"]
            scenario_list = [ "obstacle", "collision" , "interactive", "non-interactive"]
            train_data, val_data = [], []
            for scenario in scenario_list:
                basic_scenario_list = os.listdir(os.path.join(self.dataset_dir, scenario))
                for basic_scenario in basic_scenario_list:
                    variant_scenario_list = os.listdir(os.path.join(self.dataset_dir, scenario, basic_scenario, "variant_scenario"))
                    for variant_scenario in variant_scenario_list:
    
                        path = os.path.join(self.dataset_dir, scenario, basic_scenario, "variant_scenario", variant_scenario )
                        if random.random() > 0.2:
                            train_data.append(path)
                        else:
                            val_data.append(path)  
                        
                            
            dataset_dict = {}
            dataset_dict["train"] = train_data
            dataset_dict["val"] = val_data

            self.save_dict(dataset_dict, "train_val_dataset.pkl")
        else:
            dataset_dict = self.load_dict("train_val_dataset.pkl")   
            train_data = dataset_dict["train"]
            val_data = dataset_dict["val"] 
            
            
        require_data = train_data if self.is_train else val_data
        
        
        
        # self.frames = list()
        
        # ego data
        # self.measurements = list()
        
        # all actor_data 
        self.actors_data = list()
        
        self.actor_attribute = list()
        
        for subroot in require_data:

            if "interactive" in subroot:
                scenario = "interactive"
            elif "obstacle" in subroot:
                scenario = "obstacle"
            elif "collision" in subroot:
                scenario = "collision"
            else:
                scenario = "non-interactive"
                
            # All frame starts from 1     
            frame_path_list =  sorted(os.listdir( os.path.join(subroot,"actors_data")))
            num_of_frame = len(frame_path_list)

                    
            # if scenario is collision -- > we use collision frame as final frame 
            if scenario == "collision":
                with open(os.path.join(subroot,"collision_frame.json") ) as f:
                    collision_frame_json = ujson.load(f)
                    collision_frame = collision_frame_json["frame"]
                    num_of_frame = collision_frame - 1
            
            for i in range(1, num_of_frame + 1 - GAP * STEPS):
                sub_frame_path = []
                path = os.path.join(subroot, "actors_data", f"{i:08}.json")
                sub_frame_path.append(path)
                for skip in range(1, STEPS+1):
                    j = i + GAP * skip
                    
                    path = os.path.join(subroot, "actors_data", f"{j:08}.json")
                    sub_frame_path.append(path)

                    
                self.actors_data.append(sub_frame_path)
                self.actor_attribute.append(os.path.join(subroot,"actor_attribute.json"))
                
                
            
    def __len__(self):
        return len(self.actors_data)
    
    
    
    
    def __getitem__(self, i):
        
        # topview resolution 
        # --> 
        
        actor_datas = self.actors_data[i]
        actor_attribute_path = self.actor_attribute[i]
        
        
        
        current_actor_path = actor_datas[0]
        
        
        # current Town
        # Find Town 
        # No Town04 
        
        scenario = self.get_scneario(current_actor_path)
        Town = self.get_town(current_actor_path)
        
        
        with open(current_actor_path.replace("actors_data", "ego_data"), 'rt') as f1:
            data = ujson.load(f1)
    
        ego_pos = Loc(x=data["location"]["x"], y=data["location"]["y"]) # data["pos_global"]
        ego_yaw = data["rotation"]["yaw"]
        
        
        # Draw BEV map 
        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []
              
        with open(actor_attribute_path, 'rt') as f1:
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


                
        
        
        tmp_path = current_actor_path.split("actors_data")[0]
        ill_parking_id_list = self.load_dict(os.path.join(tmp_path,'ill_parking_id.pkl'))["ill_parking_id"]
        
        # obstacle bbox store in actor_attribute.json
        for id in obstacle_id_list:
            pos_0 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_2"]
            
            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            
        with open(current_actor_path, 'rt') as f1:
            data = ujson.load(f1)
                
        for id in vehicle_id_list:
            pos_0 = data[str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data[str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data[str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data[str(id)]["cord_bounding_box"]["cord_2"]
            
            # print(id)

            if int(id) == int(ego_id):
                # print("ego id ")                
                agent_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            elif int(id) == int(interactor_id):
                
                if scenario != "collision":
                
                    vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                            Loc(x=pos_1[0], y=pos_1[1]), 
                                            Loc(x=pos_2[0], y=pos_2[1]), 
                                            Loc(x=pos_3[0], y=pos_3[1]), 
                                            ])
                    
            elif id in ill_parking_id_list:
                obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                    Loc(x=pos_1[0], y=pos_1[1]), 
                                    Loc(x=pos_2[0], y=pos_2[1]), 
                                    Loc(x=pos_3[0], y=pos_3[1]), 
                                    ])
            else:
                
                # self.is_train
                # data argumentation 
                
                if self.is_train:
                    
                    if random.random() > 0.5:
                        vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
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
            pos_0 = data[str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data[str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data[str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data[str(id)]["cord_bounding_box"]["cord_2"]
            
            
            if int(id) == int(interactor_id):
                
                if scenario != "collision":
                    pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
                    
            else:
                
                if self.is_train:
                    if random.random() > 0.5:
                        pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                    Loc(x=pos_1[0], y=pos_1[1]), 
                                                    Loc(x=pos_2[0], y=pos_2[1]), 
                                                    Loc(x=pos_3[0], y=pos_3[1]), 
                                                    ])
                        
                else:
                    pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                                Loc(x=pos_1[0], y=pos_1[1]), 
                                                Loc(x=pos_2[0], y=pos_2[1]), 
                                                Loc(x=pos_3[0], y=pos_3[1]), 
                                                ])
                        
                        
            
        birdview_producer = BirdViewProducer(
                                    Town, 
                                    PixelDimensions(width=256, height=256), 
                                    pixels_per_meter=PIXELS_PER_METER)
        

        birdview: BirdView = birdview_producer.produce(ego_pos, yaw=ego_yaw,
                                                       agent_bbox_list=agent_bbox_list, 
                                                       vehicle_bbox_list=vehicle_bbox_list,
                                                       pedestrians_bbox_list=pedestrian_bbox_list,
                                                       obstacle_bbox_list=obstacle_bbox_list)
        

        with open(current_actor_path.replace("actors_data", "ego_data"), 'rt') as f1:
            data = ujson.load(f1)
        # load Map topdown
        np_path = current_actor_path.replace("actors_data", "topview")
        np_path = np_path.replace(".json", ".npy")
        
        
        
        topdown = np.load(np_path)
        
        
        birdview[1] = topdown[1]
        birdview[2] = topdown[2]
        
        
        
        birdview =  BirdViewProducer.as_ss(birdview)
        
        
        topdown = preprocess_semantic(birdview)
        
        
        
        ## future waypoints 

        u = np.float32([ego_pos.x, ego_pos.y])
                  
        # yaw = theta - 450
        theta = ego_yaw + 450
        
        theta = math.radians(theta)
        if np.isnan(theta):
             theta = 0.0
        
        # theta = theta #+ np.pi / 2
        
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        points = list()
        
        # for point_path in self.actors_data
        
        
        for index in range(1, 5):
            with open(actor_datas[index].replace("actors_data", "ego_data"), 'rt') as f1:
                data = ujson.load(f1)
    
                x=float(data["location"]["x"])
                y=float(data["location"]["y"])
                v = np.array([x, y])

                target = R.T.dot(v - u)
                target *= PIXELS_PER_METER
                target += [128, 128] 
                points.append(target)
                
        points = np.array(points)
        points = torch.FloatTensor(points)
        points = torch.clamp(points, 0, 256) # 0, 256
        points = (points / 256) * 2 - 1 # 256
        
        

        
        ## target points 
        
        variant_root_path = current_actor_path.split("actors_data")[0]
        
        
        target = self.load_dict(os.path.join(variant_root_path, "target_point.pkl"))["target_point"]
        
        x=float(target[0])
        y=float(target[1])
        v = np.array([x, y])

        target = R.T.dot(v - u)
        target *= PIXELS_PER_METER
        target += [128, 128] 
        
        target = np.clip(target, 0, 256)
        target = torch.FloatTensor(target)
        
    
        heatmap = make_heatmap((256, 256), target) 
        heatmap = torch.FloatTensor(heatmap).unsqueeze(0)
        
        
        
        
        # get ego steering and speed 
        
        with open(current_actor_path.replace("actors_data", "ego_data"), 'rt') as f1:
            data = ujson.load(f1)
    
        # ego_pos = Loc(x=data["location"]["x"], y=data["location"]["y"]) # data["pos_global"]
        # ego_yaw = data["rotation"]["yaw"]
        
        speed = float(data["speed"])
        steer = float(data["control"]["steer"])
        
        actions = np.float32([steer, speed])
    
        actions[np.isnan(actions)] = 0.0
        actions = torch.FloatTensor(actions)
        
        
        
        
        

        return topdown, points, target, actions #, meta
        
        # return torch.cat((rgb, rgb_left, rgb_right)), topdown, points, target, actions, meta
        
        
                
    
    
    def get_town(self, current_actor_path):
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

    def get_scneario(self, current_actor_path):
        
        if "interactive" in current_actor_path:
            scenario = "interactive"
        elif "obstacle" in current_actor_path:
            scenario = "obstacle"
        elif "collision" in current_actor_path:
            scenario = "collision"
        else:
            scenario = "non-interactive"
            
        return scenario
    
    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di



   


if __name__ == '__main__':

    import cv2
    from PIL import ImageDraw
    from utils.heatmap import ToHeatmap



    COLOR = np.uint8([
            (  0,   0,   0),    # 0 unlabeled  # 0
            (105, 105, 105),    # 1 ROAD
            (255, 255, 255),    # 2 ROAD LINE
            (252, 175, 62),    # 3 VEHICLES
            (233, 185, 110),    # 4 PEDESTRIANS
            (50, 50, 50),    # 5 OBSTACLES
            (138, 226, 52),    # 6 AGENT

            ])


    data = CarlaDataset(root_dir="/media/hcis-s21/DATA/LBC/dataset", is_train=True)

    to_heatmap = ToHeatmap()
    for i in range(len(data)):
        topdown, points, target, actions = data[i] 
        
        # bgr = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)
        
        # bev  = BirdViewProducer.as_ss(birdview)


        # print(bev.shape)   
        # print(set(bev.flatten()))
        
        heatmap = to_heatmap(target[None], topdown[None]).squeeze()

        # print(topdown.shape) # 7, 256, 256

        _topdown = COLOR[topdown.argmax(0).cpu().numpy()]
        _topdown[heatmap > 0.1] = 255
        
    
        
        _topdown = Image.fromarray(_topdown)
        _draw_map = ImageDraw.Draw(_topdown)
        
        
        points_unnormalized = (points + 1) / 2 * 256
        
        for x, y in points_unnormalized:
            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))
            
        
        img2 = cv2.cvtColor(np.asarray(_topdown), cv2.COLOR_RGB2BGR)
        cv2.imshow('opencv image', img2)
        cv2.waitKey(0)
        
        
        # break
                


        
        
    
    