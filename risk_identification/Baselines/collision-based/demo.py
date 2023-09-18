import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

import torch
from torch.utils.data import DataLoader

from dataset.common import get_dataset
from models.backbone import Riskbench_backbone
from models.DSA_RRL import Baseline_SA
from common import get_parser, get_one_hot

def vis(imgs,all_alphas,bbox,index):
    # batch
    show = False
    if not show:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(f"./vis/{index}.mp4"), fourcc, 12.0, (640,  256))
    all_alphas = all_alphas.unsqueeze(0)
    b, n_frame, n_obj = all_alphas.shape
    imgs = inv_normalize(imgs)
    for i in range(b):
        for j in range(n_frame):
            img = (imgs[i,j].permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
            # Convert RGB to BGR 
            img = img[:, :, ::-1].copy() 
            for k in range(n_obj):
                alpha = all_alphas[i,j,k].cpu().numpy()
                box = bbox[i,j,k].cpu().numpy()
                cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0, 255), 1)
                cv2.putText(img, str(np.round(alpha,2)), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if show:
                cv2.imshow('img',img)
                cv2.waitKey(10)
            else:
                out.write(img)
        if show:
            cv2.destroyAllWindows()
    if not show:
        out.release()
    return

def find_folder(args,root='../../Risk_identification_tool/model/'):
    if args.supervised:
        s = 'RRL'
    else:
        s = 'DSA'
    if args.intention:
        s += '_intention'
    elif args.state:
        s += '_state'
    else:
        s += '_vanilla'
    s += '_'
    model_result_list = os.listdir(root)
    model_result_list = sorted([i for i in model_result_list if i.startswith(s)],key=lambda x:int(x[-1]))
    if len(model_result_list)>=1:
        s += str(int(model_result_list[-1][-1])+1)
    else:
        s += '1'
    os.mkdir(os.path.join(root,s))
    return s

if __name__ == '__main__':
    args = get_parser()
    vis_ = args.vis
    if vis_ :
        import torchvision.transforms as transforms

        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        if not os.path.isdir('./vis/'):
            os.mkdir('./vis')
    else:
        s = find_folder(args)
        non_interactive, interactive, collision, obstacle = {}, {}, {}, {}
    n_frame = 40
    object_num = 20
    intention = args.intention
    supervised = args.supervised
    state = args.state
    model_path = os.path.join('logs',args.model_path,'best_model.pt')
    setting = {"object_num":object_num,"frame_num":n_frame,"load_img_first":args.load_first,'inference':True}
    cuda = True
    device = torch.device('cuda') if cuda else torch.device('cpu')
    backbone = Riskbench_backbone(8,object_num,intention=intention)
    model = Baseline_SA(backbone,n_frame,object_num,intention=intention,supervised=supervised,state=state)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.eval()
    if cuda:
        model = model.cuda()
    val_dataset = get_dataset(args.root,setting,mode='test',scenario_type=['collision','interactive','non-interactive','obstacle'])
    val_loader = DataLoader(val_dataset,batch_size=1, shuffle=True, num_workers=5,pin_memory=True)
    intention_input = None
    state_in = None
    with torch.no_grad():
        for index,batch in enumerate(val_loader):
            print(f"{index+1}/{len(val_dataset)}",end='\r')
            if index == 5 and vis_:
                break
            img = batch['img']
            bbox = batch['bbox']
            if intention:
                intention_input = get_one_hot(batch['s_type'],batch['s_id'])
                intention_input = torch.from_numpy(np.array(intention_input).astype(np.float32))
            if state:
                state_in = batch['state']
            if cuda:
                img = img.cuda()
                bbox = bbox.cuda()
                if intention:
                    intention_input = intention_input.cuda()
                elif state :
                    state_in = state_in.cuda()
            pred, all_alphas, _ = model(img,bbox,intention=intention_input,state=state_in)
            all_alphas = all_alphas[0]
            if vis_:
                pred = pred.softmax(dim=2)
                vis(img,all_alphas,bbox,index)
            else:
                s_type, s_id, variant, bbox_id = batch['s_type'][0], batch['s_id'][0], batch['variant'][0], batch['bbox_id'][0]
                current = s_id + '_' +variant
                n_frame = all_alphas.shape[0]
                tmp = {}
                for i in range(n_frame):
                    tmp[i+1] = {}
                    for score, id in zip(all_alphas[i],bbox_id[i]):
                        score, id = score.cpu().numpy(), id.cpu().numpy()
                        if id == -1:
                            break
                        tmp[i+1][str(int(id))] = round(float(score),2)
                if s_type == 'interactive':
                    interactive[current] = tmp
                elif s_type == 'collision':
                    collision[current] = tmp
                elif s_type == 'obstacle':
                    obstacle[current] = tmp
                else:
                    non_interactive[current] = tmp 
    if not vis_:
        with open(f"../../Risk_identification_tool/model/{s}/interactive.json", 'w') as f:
            json.dump(interactive, f)  
        with open(f"../../Risk_identification_tool/model/{s}/non-interactive.json", 'w') as f:
            json.dump(non_interactive, f)
        with open(f"../../Risk_identification_tool/model/{s}/collision.json", 'w') as f:
            json.dump(collision, f)
        with open(f"../../Risk_identification_tool/model/{s}/obstacle.json", 'w') as f:
            json.dump(obstacle, f)
    

# python demo.py --root ../dataset/ --batch 8 --model_path logs/8_26_20_58/best_model.pt