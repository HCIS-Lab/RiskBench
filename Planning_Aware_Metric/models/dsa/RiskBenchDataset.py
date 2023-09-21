import os
import numpy as np
import json
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image


AREA_THRESHOLD = 200
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
def read_json(f):
    with open(f) as json_data:
        data = json.load(json_data)
    json_data.close()
    return data

def parse_riskyid_frame(bbox_id,risky_id,label): 
    frame = bbox_id.shape[0]
    out = []
    if risky_id == -1 or not label:
        return [0] * frame
    for per_frame in bbox_id:
        index = np.where(per_frame==risky_id)[0]
        assert len(index) <= 1
        if len(index) == 1:
            out.append(index[0]+1)
        else:
            out.append(0)
    return out

class RiskBench_dataset(Dataset):
    # each RiskBench_dataset is a basic scenario
    def __init__(self,root,s_type,s_id,object_num=20,frame_num=40,load_img_first=False,inference=False,raw_img=False,designate=None):
        self.label = True if s_type == "collision" else False
        self.raw_img = raw_img
        self.s_type = s_type
        self.s_id = s_id
        self.variant = []
        self.img = []
        self.bbox = []
        self.bbox_id = []
        self.load_img_first = load_img_first
        self.risky_id = []
        path = os.path.join(root,s_type,s_id)
        for variant in os.listdir(os.path.join(path,'variant_scenario')):
            if designate is not None:
                if variant != designate:
                    continue
            self.variant.append(variant)
            variant_path = os.path.join(path,'variant_scenario',variant)
            rgb_files = sorted(os.listdir(os.path.join(variant_path,'rgb','front')))
            rgb_files = [os.path.join(variant_path,'rgb','front',img) for img in rgb_files]
            bboxs = read_json(os.path.join(variant_path,'bbox.json'))
            self.risky_id.append(read_json(os.path.join(variant_path,'actor_attribute.json'))['interactor_id'])

            if self.label:
                collision_info = read_json(os.path.join(variant_path,'collision_frame.json'))
                collision_frame = collision_info["frame"]
                collision_index = int(collision_frame) - 1
            # Arrange rgb frames & bbox
            # Positive data
            if not inference:
                if self.label:
                    start, end = collision_index-int(frame_num*0.8)+1, collision_index-int(frame_num*0.8)+1+frame_num
                    if start<0 or len(rgb_files) < collision_index-int(frame_num*0.8)+1+frame_num:
                        continue
                # Negative data
                else:
                    start, end = 0, frame_num
                    if len(rgb_files)<frame_num:
                        continue
            else:
                start, end = 0, len(rgb_files)
            imgs = rgb_files[start:end]
            bbox = []
            bbox_id = []
            for i,frame in enumerate(bboxs):
                if i == end:
                    break
                if i >= start:
                    tmp_box = np.zeros((object_num,4))
                    tmp_obj_id = np.zeros((object_num))-1 # initialize with -1
                    counter = 0
                    for obj_id,box in bboxs[frame].items():
                        # if exceed max number of objects
                        if counter == object_num:
                            break
                        if (box[2] - box[0])*(box[3] - box[1]) < AREA_THRESHOLD:
                            continue
                        tmp_box[counter] = box
                        tmp_obj_id[counter] = obj_id
                        counter += 1
                    bbox.append(tmp_box)
                    bbox_id.append(tmp_obj_id)
            self.bbox.append(np.array(bbox).astype(np.float32))
            self.bbox_id.append(np.array(bbox_id))

            if load_img_first:
                imgs = [Image.open(img) for img in imgs]
            self.img.append(np.array(imgs))

    def __getitem__(self, index):
        imgs = self.img[index]
        if not self.load_img_first:
            raw_imgs = [Image.open(img) for img in imgs]
        
        imgs = [transform(img) for img in raw_imgs]
        imgs = torch.stack(imgs) # 40 3 H W
        label = [0,1] if self.label else [1,0]
        label = torch.from_numpy(np.array(label).astype(np.float32))
        bbox, bbox_id = self.bbox[index], self.bbox_id[index]
        risky_id_frame = np.array(parse_riskyid_frame(bbox_id,self.risky_id[index],self.label))
        bbox, bbox_id, risky_id_frame = torch.from_numpy(bbox), torch.from_numpy(bbox_id), torch.from_numpy(risky_id_frame).type(torch.LongTensor)
        if self.raw_img:
            raw_imgs = [np.array(img) for img in raw_imgs]
        else:
            raw_imgs = [0]
        return {'img':imgs,'bbox':bbox,'bbox_id':bbox_id,'label':label,'s_type':self.s_type,'s_id':self.s_id,'variant':self.variant[index],'risky_id':risky_id_frame,'raw_img':raw_imgs}

    def __len__(self):
        return len(self.img)