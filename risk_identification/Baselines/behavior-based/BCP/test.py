from models import GCN as Model
import config as cfg

import os
import json
import copy
import argparse
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
from torchvision import transforms


def to_device(x, device):
    return x.unsqueeze(0).to(device)


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def load_weight(model, checkpoint):

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


def create_model(args, device):

    model = Model(args.time_steps, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_intention=args.use_intention, NUM_BOX=args.num_box)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)

    model = nn.DataParallel(model).to(device)
    count_parameters(model)

    if isinstance(model, nn.DataParallel):
        model = model.module

    return model


def load_state(state_root, scenario):
    
    state_dict = {}

    for (basic, variant, data_type) in scenario:
        state_path = os.path.join(state_root, data_type, basic+"_"+variant+".json")
        state = json.load(open(state_path))   
        state_dict[basic+"_"+variant] = state

    return state_dict


def normalize_box(trackers, scale_w, scale_h):
    """
        return normalized_trackers TxNx4 ndarray:
        [BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_BOTRIGHT_X, BBOX_BOTRIGHT_Y]
    """

    normalized_trackers = trackers.copy()
    normalized_trackers[:, :,
                        0] = normalized_trackers[:, :, 0] * scale_w
    normalized_trackers[:, :,
                        2] = normalized_trackers[:, :, 2] * scale_w
    normalized_trackers[:, :,
                        1] = normalized_trackers[:, :, 1] * scale_h
    normalized_trackers[:, :,
                        3] = normalized_trackers[:, :, 3] * scale_h

    return normalized_trackers


def find_tracker(args, basic, variant, tracking, start, end, actor_state_dict=None):
    """
        tracking Kx10 ndarray:
        [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
        e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
    """

    INTENTIONS = {'r': 1, 'sl': 2, 'f': 3, 'gi': 4, 'l': 5, 'gr': 6, 'u': 7, 'sr': 8,'er': 9}
    

    num_object = args.num_box
    time_steps = args.time_steps
    height, width = args.img_size
    scale_w = args.img_resize[1]/args.img_size[1]
    scale_h = args.img_resize[0]/args.img_size[0]
    
    t_array = tracking[:, 0]
    tracking_index = tracking[np.where(t_array == end-1)[0], 1]
    
    trackers = np.zeros([time_steps, num_object, 4]).astype(np.float32)   # TxNx4
    intentions = np.zeros(10).astype(np.float32)   # 10
    states = np.zeros([time_steps, num_object+1, 2]).astype(np.float32)   # Tx(N+1)x2

    basic_token = basic.split('_')
    ego_intention = basic_token[3] if "obstacle" in args.data_type else basic_token[5]
    intentions[INTENTIONS[ego_intention]] = 1


    for t in range(start, end):
        current_tracking = tracking[np.where(t_array == t)[0]]
        state_dict = actor_state_dict[basic+"_"+variant][str(t)]

        for i, object_id in enumerate(tracking_index):
            current_actor_id_idx = np.where(
                current_tracking[:, 1] == object_id)[0]

            if len(current_actor_id_idx) != 0:
                # x1, y1, x2, y2
                bbox = current_tracking[current_actor_id_idx, 2:6]
                bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
                bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, width)
                bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
                bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, height)
                trackers[t-start, i, :] = bbox

                if str(int(object_id)%65536) in state_dict:
                    states[t-start, i+1] = state_dict[str(object_id)]
                elif str(int(object_id)%65536+65536) in state_dict:
                    states[t-start, i+1] = state_dict[str(int(object_id)%65536+65536)]
                else:
                    states[t-start, i+1] = 0

    trackers = normalize_box(trackers, scale_w, scale_h)
    return trackers, tracking_index, intentions, states


def save_RiskScore(roi_path, scenario_id_weather, risk_score_dict):

    with open(roi_path) as f:
        ROI_result = json.load(f)

    ROI_result[scenario_id_weather] = risk_score_dict
    with open(roi_path, "w") as f:
        json.dump(ROI_result, f, indent=4)


def testing(args, model, test_set, tracking_list, device):

    time_steps = args.time_steps
    num_box = args.num_box
    use_continuous_hidden_state = args.continuous_hidden_state
    
    camera_transforms = transforms.Compose([
        # transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    start_frame = 1
    
    actor_state_dict = None
    if args.use_intention:
        state_root = "./datasets/state"
        actor_state_dict = load_state(state_root, test_set)


    for cnt, (test_sample, tracking_results) in enumerate(zip(test_set, tracking_list), 1):

        basic, variant, data_type = test_sample
        test_sample_path = os.path.join(
            args.data_root, data_type, basic, "variant_scenario", variant)

        if args.verbose:
            print("="*20)
            print(basic, variant)

        img_path = os.path.join(test_sample_path, 'rgb/front')
        N = len(os.listdir(img_path))
        first_frame_id = start_frame + time_steps - 1
        last_frame_id = start_frame + N - 1

        state_dict = dict()
        risk_score_dict = dict()

        for frame_id in range(first_frame_id, last_frame_id+1):

            et = frame_id
            st = et - (time_steps-1)

            #  trackers: TxNx4 (TOPLEFT_X, TOPLEFT_Y, BOTRIGHT_X, BOTRIGHT_Y) with normalization
            trackers, tracking_id, intention_inputs, states = find_tracker(
                args, basic, variant, tracking_results, st, et+1, actor_state_dict)
            trackers = to_device(torch.from_numpy(trackers), device)
            intention_inputs = to_device(torch.from_numpy(intention_inputs), device)
            states = to_device(torch.from_numpy(states), device)

            camera_inputs = []
            action_logits = [[0.0, 1.0]]    # dummy logit

            """ without intervention """
            # initialize LSTM
            hx = torch.zeros((num_box+1, model.hidden_size)).to(device)
            cx = torch.zeros((num_box+1, model.hidden_size)).to(device)

            if use_continuous_hidden_state and st-1 in state_dict:

                # ego previous state
                hx[0], cx[0] = state_dict[st-1][-1]

                # other actors state
                for idx, actor_id in enumerate(tracking_id, 1):
                    if actor_id in state_dict[st-1]:
                        hx[idx], cx[idx] = state_dict[st-1][actor_id]

            for l in range(st, et+1):

                camera_name = f"{l:08d}.jpg"
                camera_path = os.path.join(img_path, camera_name)
                img = Image.open(camera_path).convert('RGB')

                camera_input = camera_transforms(img)
                camera_input = to_device(camera_input, device)

                # save for later usage in intervention
                camera_inputs.append(camera_input.clone().detach())

                mask = torch.ones(
                    (1, 3, args.img_resize[0], args.img_resize[1])).to(device)

                """ ego feature """
                if args.partial_conv:
                    ego_feature = model.backbone.features(camera_input, mask)
                else:
                    ego_feature = model.backbone.features(camera_input)

                # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
                ego_feature = model.camera_features(
                    ego_feature).reshape(1, 1, -1)

                """ object feature """
                # 1xTxNx4 -> 1xNx4
                tracker = trackers[:, (l - st)].reshape(-1, num_box, 4)

                # 1xNx512
                _, obj_feature = model.object_backbone(camera_input, tracker)

                # 1x(1+N)x512
                feature_input = torch.cat((ego_feature, obj_feature), 1)

                if args.use_intention:

                    # 1xTx(1+N)x2 -> 1x(1+N)x2
                    state = states[:, l-st]
                    # 1x(1+N)x2  -> 1x(1+N)x128
                    state_feature = model.state_model(state)
                    # 1x(1+N)x512 -> 1x(1+N)x(512+128)
                    feature_input = torch.cat(
                        (feature_input, state_feature), -1)

                    # # 1x(1+N)x256
                    # intention_feature = model.intention_model(intention_inputs)
                    # # 1x(1+N)x512 -> 1x(1+N)x(512+256)
                    # feature_input = torch.cat((feature_input, intention_feature), -1)

                # 1x(1+N)x512 -> (1+N)x512
                feature_input = feature_input.reshape(-1, model.fusion_size)

                # LSTM
                hx, cx = model.step(feature_input, hx, cx)


            # save object state for next iteraton, -1 represnet ego id here
            if use_continuous_hidden_state:
                state_dict[et] = {-1: (hx[0], cx[0])}
                for idx, actor_id in enumerate(tracking_id, 1):
                    state_dict[et][actor_id] = (hx[idx], cx[idx])

            updated_feature, attn_weights = model.message_passing(
                hx, trackers, device)

            vel = model.vel_classifier(model.drop(updated_feature))
            vel = model.sigmoid(vel).reshape(-1)

            confidence_go = 1-vel.to('cpu').numpy()[0]

            """ with intervention """
            for i in range(len(tracking_id)):

                # initialize LSTM
                hx = torch.zeros((num_box+1, model.hidden_size)).to(device)
                cx = torch.zeros((num_box+1, model.hidden_size)).to(device)

                if use_continuous_hidden_state and st-1 in state_dict:
                    # ego previous state
                    hx[0], cx[0] = state_dict[st-1][-1]

                    # other actors state
                    for idx, actor_id in enumerate(tracking_id, 1):
                        if actor_id in state_dict[st-1]:
                            hx[idx], cx[idx] = state_dict[st-1][actor_id]

                for l in range(st, et+1):

                    camera_input = camera_inputs[l-st].clone().detach()

                    y1 = int(trackers[0, l-st, i, 1])  # TOPLEFT_Y
                    y2 = int(trackers[0, l-st, i, 3])  # BOTRIGHT_Y
                    x1 = int(trackers[0, l-st, i, 0])  # TOPLEFT_X
                    x2 = int(trackers[0, l-st, i, 2])  # BOTRIGHT_X

                    camera_input[:, :, y1:y2, x1: x2] = 0
                    mask = torch.ones(
                        (1, 3, args.img_resize[0], args.img_resize[1])).to(device)
                    mask[:, :, y1:y2, x1:x2] = 0

                    """ ego feature """
                    if args.partial_conv:
                        ego_feature = model.backbone.features(
                            camera_input, mask)
                    else:
                        ego_feature = model.backbone.features(camera_input)

                    # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
                    ego_feature = model.camera_features(
                        ego_feature).reshape(1, 1, -1)

                    """ object feature """
                    # 1xTxNx4 -> 1xNx4
                    tracker = trackers[:, (l - st)].reshape(-1,
                                                            num_box, 4).clone().detach()
                    tracker[:, i, :] = 0

                    # 1xNx512
                    _, obj_feature = model.object_backbone(
                        camera_input, tracker)

                    # 1x(1+N)x512
                    feature_input = torch.cat((ego_feature, obj_feature), 1)

                    if args.use_intention:

                        # 1xTx(1+N)x2 -> 1x(1+N)x2 
                        state = states[:, l-st]
                        # 1x(1+N)x2  -> 1x(1+N)x128
                        state_feature = model.state_model(state)
                        # 1x(1+N)x512 -> 1x(1+N)x(512+128)
                        feature_input = torch.cat(
                            (feature_input, state_feature), -1)

                        # # 1x(1+N)x256
                        # intention_feature = model.intention_model(intention_inputs)
                        # # 1x(1+N)x512 -> 1x(1+N)x(512+256)
                        # feature_input = torch.cat((feature_input, intention_feature), -1)

                    # 1x(1+N)x512 -> (1+N)x512
                    feature_input = feature_input.reshape(-1, model.fusion_size)

                    # LSTM
                    hx, cx = model.step(feature_input, hx, cx)

                # for name, param in model.named_parameters():
                #     if param.requires_grad and "vel_classifier.2.weight" in name:
                #         print(name, param)

                intervened_trackers = torch.ones(
                    (1, time_steps, num_box, 4)).to(device)
                intervened_trackers[:, :, i, :] = 0.0
                intervened_trackers = intervened_trackers * trackers

                updated_feature, _ = model.message_passing(
                    hx, intervened_trackers, device)

                vel = model.vel_classifier(model.drop(updated_feature))
                vel = model.sigmoid(vel).reshape(-1)

                # score go and score stop
                s_stop = vel.to('cpu').numpy()[0]
                s_go = 1 - s_stop
                action_logits.append([s_go, s_stop])

            if args.verbose:
                print(f"{st:3d} to {et:3d} Scenario s_go: {confidence_go:4f}")

            action_logits = np.clip(action_logits, 0, 1)

            risk_score_dict[str(frame_id)] = {}
            for actor_id, score, attn in zip(tracking_id, action_logits[1:len(tracking_id)+1], attn_weights[1:len(tracking_id)+1]):
                risk_score_dict[str(frame_id)][str(
                    actor_id)] = [score[0], attn.detach().to('cpu').numpy().item()]

            risk_score_dict[str(frame_id)]["scenario_go"] = np.float64(
                confidence_go)

        if args.save_roi:
            scenario_id_weather = basic + '_' + variant
            save_RiskScore(args.roi_path, scenario_id_weather, risk_score_dict)

        print(f'Sample: {cnt}/{len(test_set)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='all', type=str, required=True)
    parser.add_argument('--phases', default=['test'], type=list)
    parser.add_argument('--ckpt_path', default="", type=str, required=True)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--time_steps', default=5, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_intention', action='store_true', default=False)
    parser.add_argument('--data_augmentation', default=True, type=bool)
    parser.add_argument('--continuous_hidden_state', action='store_true', default=False)
    parser.add_argument('--save_roi', default=True)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--Method', default="", type=str)

    args = cfg.parse_args(parser)
    args = cfg.read_data(args, valiation=False)

    if args.save_roi:
        cfg.create_ROI_result(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(args, device)
    model.train(False)

    test_set, tracking_list = args.test_session_set, args.tracking_list

    with torch.no_grad():
        testing(args, model, test_set, tracking_list, device)
