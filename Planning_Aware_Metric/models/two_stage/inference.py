import numpy as np
import torch

def to_device(x, device):
    return x.unsqueeze(0).to(device)


def testing(model, test_imgs, trackers, tracking_id, time_steps=5, num_box=25, device='cuda'):
    
    trackers = to_device(torch.from_numpy(trackers.astype(np.float32)), device)
    action_logits = [[0.0, 1.0]]    # dummy logit

    """ without intervention """
    # initialize LSTM
    hx = torch.zeros((num_box+1, model.hidden_size)).to(device)
    cx = torch.zeros((num_box+1, model.hidden_size)).to(device)

    for l in range(time_steps):

        camera_input = test_imgs[l].clone().detach()
        camera_input = to_device(camera_input, device)

        mask = torch.ones((1, 3, 256, 640)).to(device)

        """ ego feature """
        ego_feature = model.backbone.features(camera_input, mask)

        # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
        ego_feature = model.camera_features(
            ego_feature).reshape(1, 1, -1)

        """ object feature """
        # 1xTxNx4 -> 1xNx4
        tracker = trackers[:, l].reshape(-1, num_box, 4)

        # 1xNx512
        _, obj_feature = model.object_backbone(camera_input, tracker)

        # 1x(1+N)x512
        feature_input = torch.cat((ego_feature, obj_feature), 1)

        # (1+N)x512
        feature_input = feature_input.reshape(-1, model.fusion_size)

        # LSTM
        hx, cx = model.step(feature_input, hx, cx)

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

        for l in range(time_steps):

            camera_input = test_imgs[l].clone().detach()
            camera_input = to_device(camera_input, device)

            y1 = int(trackers[0, l, i, 1])  # TOPLEFT_Y
            y2 = int(trackers[0, l, i, 3])  # BOTRIGHT_Y
            x1 = int(trackers[0, l, i, 0])  # TOPLEFT_X
            x2 = int(trackers[0, l, i, 2])  # BOTRIGHT_X

            camera_input[:, :, y1:y2, x1: x2] = 0
            mask = torch.ones(
                (1, 3, 256, 640)).to(device)
            mask[:, :, y1:y2, x1:x2] = 0

            """ ego feature """
            ego_feature = model.backbone.features(camera_input, mask)

            # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
            ego_feature = model.camera_features(
                ego_feature).reshape(1, 1, -1)

            """ object feature """
            # 1xTxNx4 -> 1xNx4
            tracker = trackers[:, l].reshape(-1, num_box, 4).clone().detach()
            tracker[:, i, :] = 0

            # 1xNx512
            _, obj_feature = model.object_backbone(
                camera_input, tracker)

            # 1x(1+N)x512
            feature_input = torch.cat((ego_feature, obj_feature), 1)

            # (1+N)x512
            feature_input = feature_input.reshape(
                -1, model.fusion_size)

            # LSTM
            hx, cx = model.step(feature_input, hx, cx)


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


    action_logits = np.clip(action_logits, 0, 1)

    single_result = []
    single_score = []
    two_result = []
    two_score = []

    two_stage_dict = {}
    single_dict = {}

    for actor_id, score, attn in zip(tracking_id, action_logits[1:len(tracking_id)+1], attn_weights[1:len(tracking_id)+1]):

        # print(str(actor_id), f"{attn.item():.4f}, {score[0].item():.4f}, {confidence_go.item():.4f}")        
        # single_result[str(actor_id)] = bool(attn>0.19)
        # two_result[str(actor_id)] = bool(score[0]-confidence_go>0.03 and confidence_go<0.5)
        two_stage_dict[int(actor_id)] = score[0]
        single_dict[int(actor_id)] = attn

        if attn>0.35 and confidence_go < 0.4:
            single_result.append(int(actor_id))
            single_score.append(attn)

        if score[0]-confidence_go>0.2 and confidence_go<0.5:
            two_result.append(int(actor_id))
            two_score.append(score[0]-confidence_go)

        # [single_score.index(max(single_score))]

    # print(single_score)
    # print(two_score)

    if len(single_result) != 0:
        single_score = [single_result[single_score.index(max(single_score))]]
    if len(two_score) != 0:
        two_score = [two_result[two_score.index(max(two_score))]]
        


    return single_score, two_score, two_stage_dict, single_dict
                

