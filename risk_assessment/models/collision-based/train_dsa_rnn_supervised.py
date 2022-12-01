import dsa_rnn_supervised
import carla_dataset
import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import json
import matplotlib.pyplot as plt
import GT_loader

from torch.utils.tensorboard import SummaryWriter


def detection_collate(batch):
    frames = None
    rois = None
    collision_labels = []
    risk_labels = []
    file_name = []

    for sample in batch:
        frame, roi, collision_label, risk_label, path = sample
        if frames is not None:
            frames = torch.cat((frames, frame.unsqueeze(0)))
            rois = torch.cat((rois, roi.unsqueeze(0)))
        else:
            frames = frame.unsqueeze(0)
            rois = roi.unsqueeze(0)
        
        collision_labels.append(collision_label)
        risk_labels.append(risk_label)
        file_name.append(path)

    collision_labels, risk_labels = np.array(collision_labels), np.array(risk_labels)
    return frames, rois, torch.from_numpy(collision_labels), torch.from_numpy(risk_labels), file_name


def training(batch_size, n_epoch, l_r, local_path, nas_path, tracking, clip_time=60):
    #tracking: obj per frame
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    # net = torch.nn.DataParallel(net).to(device)
    # print(net)

    weight_c = torch.tensor([0.63, 2.45], dtype=torch.float).to(device)
    weight_r = torch.tensor([0.5, 97.8], dtype=torch.float).to(device)
    criterion1 = baseline3_model.collision_loss(clip_time*0.8, weight_c).to(device)
    criterion2 = baseline3_model.risk_obj_loss(weight_r).to(device)
    # criterion2 = baseline3_model.risk_obj_focal_loss().to(device)
    # criterion1, criterion2 = torch.nn.DataParallel(criterion1).to(device), torch.nn.DataParallel(criterion2).to(device)
    # risk_paras = list(map(id, net.risk_output.parameters()))#net.module.risk_output.parameters()
    # base_paras = filter(lambda p: id(p) not in risk_paras, net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=l_r)
    # optimizer = optim.Adam([
    #             {'params': net.risk_output.parameters(), 'lr': 1e-5},
    #             {'params': base_paras}],
    #             lr=l_r)

    trainloader, _, valloader = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time)
    print("finish data loading.")

    batch_n, val_batch_n  = len(trainloader), len(valloader)
    best_vloss = float('inf')
    # loss_list, val_loss_list = [], []
    colli_result, risk_result = [], []
    val_colli_result, val_risk_result = [], []
    risk_thres = 0.5

    for epoch in range(n_epoch):
        print(f"[Epoch: {epoch+1}/{n_epoch}]")
        net.train()
        start_t = time.time()
        running_loss = 0.0
        running_loss_c, running_loss_r = 0.0, 0.0
        TP_c, FP_c, TN_c, FN_c = 0.0, 0.0, 0.0, 0.0
        TP_r, FP_r, TN_r, FN_r = 0.0, 0.0, 0.0, 0.0

        for i, data in enumerate(trainloader):
            frame, roi, collision_labels, risk_labels, _ = data
            frame, roi = frame.to(device), roi.to(device)
            collision_labels, risk_labels = collision_labels.to(device),  risk_labels.to(device)
            collision_labels, risk_labels = collision_labels.long(), risk_labels.long()

            optimizer.zero_grad()
            pred_c, logits_c, pred_r, score_r = net(roi, frame)
            
            pred_c = pred_c.argmax(dim=2)[:, int(clip_time*0.8)]#bxtx2 ->b
            tp, fp, tn, fn = carla_dataset.calculate_matric(pred_c, collision_labels)
            TP_c += tp; FP_c += fp; TN_c += tn; FN_c += fn

            pred = (pred_r[:, int(clip_time*0.8)] > risk_thres).float()#bxn
            for (p, l) in zip(pred, risk_labels):
                tp, fp, tn, fn = carla_dataset.calculate_matric(p, l)
                # print(tp,fp,tn,fn)
                TP_r += tp; FP_r += fp; TN_r += tn; FN_r += fn
            
            loss_c = criterion1(logits_c, collision_labels, device)#.mean()
            loss_r = criterion2(pred_r, risk_labels, device)#.mean()
            # loss_r = criterion2(score_r, risk_labels, device)#.mean()
            loss = loss_c + loss_r
            loss.backward()
            running_loss += loss.item(); running_loss_c += loss_c.item(); running_loss_r += loss_r.item()
            optimizer.step()

            # print(f"\tBatch: {i+1}/{batch_n}, loss_c: {loss_c.item()}, loss_r: {loss_r.item()}", end="\r")


        # avg_loss = running_loss / batch_n
        # loss_list.append(avg_loss)
        avg_loss_c, avg_loss_r = running_loss_c / batch_n, running_loss_r / batch_n
        
        acc_c = (TP_c+TN_c) / (TP_c+FP_c+TN_c+FN_c) * 100
        precision_c = (TP_c / (TP_c+FP_c) * 100) if (TP_c+FP_c)>0 else 0
        recall_c = (TP_c / (TP_c+FN_c) * 100) if (TP_c+FN_c)>0 else 0
        f1_c = ((2*recall_c*precision_c) / (precision_c+recall_c)) if (precision_c+recall_c)>0 else 0
        colli_result.append([acc_c, precision_c, recall_c, f1_c, avg_loss_c])

        acc_r = (TP_r+TN_r) / (TP_r+FP_r+TN_r+FN_r) * 100
        precision_r = (TP_r / (TP_r+FP_r) * 100) if (TP_r+FP_r)>0 else 0
        recall_r = (TP_r / (TP_r+FN_r) * 100) if (TP_r+FN_r)>0 else 0
        f1_r = ((2*recall_r*precision_r) / (precision_r+recall_r)) if (precision_r+recall_r)>0 else 0
        risk_result.append([acc_r, precision_r, recall_r, f1_r, avg_loss_r])

        print("\n\ttotal_c:",TP_c, FP_c,TN_c,FN_c, "total_r:",TP_r, FP_r,TN_r,FN_r)
        print(f"\tloss_c: {avg_loss_c:.3f}, loss_r: {avg_loss_r:.3f}, Time:{time.time()-start_t:.1f}s")
        print(f"  [Collision] Acc: {acc_c:2.3f}%, Precision: {precision_c:2.3f}%, Recall: {recall_c:2.3f}%")
        print(f"  [Risk]      Acc: {acc_r:2.3f}%, Precision: {precision_r:2.3f}%, Recall: {recall_r:2.3f}%\n")

        if running_loss < best_vloss:
            best_vloss = running_loss
            model_path = f"baseline3_last/model_{epoch+1}"
            torch.save(net.state_dict(), model_path)
            print("Model saved.")
        
        '''validating'''
        if epoch%2 == 1:
            print("----------validating----------")
            net.eval()
            eval_loss, eval_loss_c, eval_loss_r = 0.0, 0.0, 0.0
            TP_c, FP_c, TN_c, FN_c = 0.0, 0.0, 0.0, 0.0
            TP_r, FP_r, TN_r, FN_r = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for i, data in enumerate(valloader):
                    frame, roi, collision_labels, risk_labels, _ = data
                    frame, roi = frame.to(device), roi.to(device)
                    collision_labels, risk_labels = collision_labels.to(device),  risk_labels.to(device)
                    collision_labels, risk_labels = collision_labels.long(), risk_labels.long()

                    pred_c, logits_c, pred_r, score_r = net(roi, frame)

                    pred_c = pred_c.argmax(dim=2)[:, int(clip_time*0.8)]    #b
                    tp, fp, tn, fn = carla_dataset.calculate_matric(pred_c, collision_labels)
                    TP_c += tp; FP_c += fp; TN_c += tn; FN_c += fn
                    
                    pred = (pred_r[:, int(clip_time*0.8)] > risk_thres).float()#bxn
                    for (p, l) in zip(pred, risk_labels):
                        tp, fp, tn, fn = carla_dataset.calculate_matric(p, l)
                        # print(tp,fp,tn,fn)
                        TP_r += tp; FP_r += fp; TN_r += tn; FN_r += fn

                    loss_c = criterion1(logits_c, collision_labels, device)#.mean()
                    loss_r = criterion2(pred_r, risk_labels, device)#.mean()
                    # loss_r = criterion2(score_r, risk_labels, device)#.mean()
                    loss = loss_c + loss_r
                    eval_loss += loss.item(); eval_loss_c += loss_c.item(); eval_loss_r += loss_r.item()

                # avg_loss = eval_loss / val_batch_n
                # val_loss_list.append(avg_loss)
                avg_loss_c, avg_loss_r = eval_loss_c / val_batch_n, eval_loss_r / val_batch_n
                
                precision_c = (TP_c / (TP_c+FP_c) * 100) if (TP_c+FP_c)>0 else 0
                recall_c = (TP_c / (TP_c+FN_c) * 100) if (TP_c+FN_c)>0 else 0
                f1_c = ((2*recall_c*precision_c) / (precision_c+recall_c)) if (precision_c+recall_c)>0 else 0
                val_colli_result.append([precision_c, recall_c, f1_c, avg_loss_c])

                precision_r = (TP_r / (TP_r+FP_r) * 100) if (TP_r+FP_r)>0 else 0
                recall_r = (TP_r / (TP_r+FN_r) * 100) if (TP_r+FN_r)>0 else 0
                f1_r = ((2*recall_r*precision_r) / (precision_r+recall_r)) if (precision_r+recall_r)>0 else 0
                val_risk_result.append([precision_r, recall_r, f1_r, avg_loss_r])

            print("\n\ttotal_c:",TP_c, FP_c,TN_c,FN_c, "total_r:",TP_r, FP_r,TN_r,FN_r)
            print(f"loss_c: {avg_loss_c:.3f}, loss_r: {avg_loss_r:.3f}")
            print(f"[Collision] Precision: {precision_c:2.3f}, Recall: {recall_c:2.3f}")
            print(f"[Risk]      Precision: {precision_r:2.3f}, Recall: {recall_r:2.3f}\n")

    print("Training finish.")

    colli_result, risk_result = np.array(colli_result), np.array(risk_result)
    val_colli_result, val_risk_result = np.array(val_colli_result), np.array(val_risk_result)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(1, n_epoch+1))
    plt.plot(range(1, n_epoch+1), colli_result[:,4], label="train-colli")
    plt.plot(range(1, n_epoch+1), risk_result[:,4], label="train-risk")
    # plt.plot(range(1, n_epoch+1), loss_list, label="training", color='r')
    # plt.plot(range(1, n_epoch+1, 2), val_loss_list, label="validation", color='g')
    plt.plot(range(1 ,n_epoch+1, 2), val_colli_result[:,3], label="val-colli")
    plt.plot(range(1, n_epoch+1, 2), val_risk_result[:,3], label="val-risk")
    plt.legend()
    plt.savefig("baseline3_last/Loss.png")

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('Precision')
    plt.xticks(range(1, n_epoch+1))
    plt.plot(range(1, n_epoch+1), colli_result[:,1], label="train-colli")
    plt.plot(range(1, n_epoch+1), risk_result[:,1], label="train-risk")
    plt.plot(range(1 ,n_epoch+1, 2), val_colli_result[:,0], label="val-colli")
    plt.plot(range(1, n_epoch+1, 2), val_risk_result[:,0], label="val-risk")
    plt.legend()
    plt.savefig("baseline3_last/Precision.png")

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('Recall')
    plt.xticks(range(1, n_epoch+1))
    plt.plot(range(1, n_epoch+1), colli_result[:,2], label="train-colli")
    plt.plot(range(1, n_epoch+1), risk_result[:,2], label="train-risk")
    plt.plot(range(1 ,n_epoch+1, 2), val_colli_result[:,1], label="val-colli")
    plt.plot(range(1, n_epoch+1, 2), val_risk_result[:,1], label="val-risk")
    plt.legend()
    plt.savefig("baseline3_last/Recall.png")

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('F1-score')
    plt.xticks(range(1, n_epoch+1))
    plt.plot(range(1, n_epoch+1), colli_result[:,3], label="train-colli")
    plt.plot(range(1, n_epoch+1), risk_result[:,3], label="train-risk")
    plt.plot(range(1 ,n_epoch+1, 2), val_colli_result[:,2], label="val-colli")
    plt.plot(range(1, n_epoch+1, 2), val_risk_result[:,2], label="val-risk")
    plt.legend()
    plt.savefig("baseline3_last/F1_score.png")

    return


def testing(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    # net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, _, valloader = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time)
    print("finish data loading.")
    
    batch_n = len(valloader)
    risk_thres = 0.5
    TP_c, FP_c, TN_c, FN_c = 0.0, 0.0, 0.0, 0.0
    TP_r, FP_r, TN_r, FN_r = 0.0, 0.0, 0.0, 0.0
    all_pred = None
    all_labels = None

    with torch.no_grad():
        for i, data in enumerate(valloader):
            print(f"Batch: {i+1}/{batch_n}", end="\r")
            frame, roi, collision_labels, risk_labels, _ = data
            frame, roi = frame.to(device), roi.to(device)
            collision_labels, risk_labels = collision_labels.to(device),  risk_labels.to(device)
            collision_labels, risk_labels = collision_labels.long(), risk_labels.long()

            pred_c, _, pred_r, _ = net(roi, frame)

            if i==0 :
                all_pred = pred_c[:,0:int(clip_time*0.8),1].cpu().numpy()
                all_labels = collision_labels.cpu().numpy()
            else:
                all_pred = np.concatenate((all_pred, pred_c[:,0:int(clip_time*0.8),1].cpu().numpy()), axis=0)
                all_labels = np.concatenate((all_labels, collision_labels.cpu().numpy()), axis=0)

            pred_c = pred_c.argmax(dim=2)[:, int(clip_time*0.8)]    #b 
            tp, fp, tn, fn = carla_dataset.calculate_matric(pred_c, collision_labels)
            TP_c += tp; FP_c += fp; TN_c += tn; FN_c += fn
            
            pred = (pred_r[:, int(clip_time*0.8)] > risk_thres).float()
            for (p, l) in zip(pred, risk_labels):
                tp, fp, tn, fn = carla_dataset.calculate_matric(p, l)
                # print(tp,fp,tn,fn)
                TP_r += tp; FP_r += fp; TN_r += tn; FN_r += fn
        
        evaluation(all_pred, all_labels, int(clip_time*0.8))
        precision_c = (TP_c / (TP_c+FP_c) * 100) if (TP_c+FP_c)>0 else 0
        recall_c = (TP_c / (TP_c+FN_c) * 100) if (TP_c+FN_c)>0 else 0
        f1_c = ((2*recall_c*precision_c) / (precision_c+recall_c)) if (precision_c+recall_c)>0 else 0

        precision_r = (TP_r / (TP_r+FP_r) * 100) if (TP_r+FP_r)>0 else 0
        recall_r = (TP_r / (TP_r+FN_r) * 100) if (TP_r+FN_r)>0 else 0
        f1_r = ((2*recall_r*precision_r) / (precision_r+recall_r)) if (precision_r+recall_r)>0 else 0

    print("\n[TEST result]")
    print("total_c:",TP_c, FP_c,TN_c,FN_c, "total_r:",TP_r, FP_r,TN_r,FN_r)
    print(f"[Collision] Precision: {precision_c:2.3f}, Recall: {recall_c:2.3f}, F1: {f1_c:2.3f}")
    print(f"[Risk]      Precision: {precision_r:2.3f}, Recall: {recall_r:2.3f}, F1: {f1_r:2.3f}\n")

    return


def evaluation(all_pred,all_labels, total_time = 48, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print ("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " +"{:.4}".format(np.mean(new_Time) * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print ("Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 3))


def ROI_sweep(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, _, valloader = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time, validation=True)
    print("finish data loading.")

    gt_stype = ["collision", "interactive", "obstacle"]
    threshold = [0.3, 0.4, 0.5, 0.6, 0.7 ,0.8]
    accumulate = [1, 2, 3, 4, 5]
    result = np.zeros((len(accumulate), len(threshold), 4))# 4:tp fp tn fn
    test = np.zeros(4)

    with torch.no_grad():
        for data in valloader:
            frame, roi, collision_labels, _, path = data
            frame, roi = frame.to(device), roi.to(device)

            cur_len = roi.shape[1]
            net.n_frame = cur_len
            pred_c, _, pred_r, _ = net(roi, frame)

            for i, label in enumerate(collision_labels):
                p_c = pred_c[i].cpu().numpy()#tx2
                # p_r = (pred_r[i] > 0.5).float().cpu().numpy()#txn
                s_r = pred_r[i].cpu().numpy()#txn

                file_name = path[i]
                s_type, s_name, _, weather = file_name.split('/')
                print(file_name)
                if s_type == "collision":
                    test[0] += 1
                elif s_type ==  "interactive":
                    test[1] += 1
                elif s_type == "obstacle":
                    test[2] += 1
                else:
                    test[3] += 1

                track_file = open(os.path.join(local_path, file_name, 'tracker.json'))
                track_data = json.load(track_file)
                track_file.close()

                if s_type in gt_stype:
                    # start_frame, gt_frame, gt_cause_id = GT_loader.getGTframe(s_type, s_name, weather)
                    _, gt_frame, gt_cause_id = GT_loader.getGTframe(s_type, s_name, weather)
                else:
                    # start_frame, gt_frame, gt_cause_id = None, None, None
                    gt_frame, gt_cause_id = None, None

                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])
                # if gt_frame is None:
                #     bbox_list = bbox_list[:clip_time]

                for accum in accumulate:
                    for k, thres in enumerate(threshold):
                        for j in range(cur_len):#test each frame
                            frame_num = int(bbox_list[j].split('.')[0])
                            bbox_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                            bbox_data = json.load(bbox_file)
                            bbox_file.close()

                            if (s_type in gt_stype) and (frame_num == gt_frame):
                                break
                            
                            flag = True #for accumulate condition check
                            # pos = True #frame before start frame is considered negative
                            # if start_frame and frame_num < start_frame:
                            #     pos = False

                            if j < accum-1:
                                flag = False
                            else:
                                for cnt in range(accum):#check accumulate condition
                                    if p_c[j-cnt, 1] < 0.5:
                                        flag = False
                                        break       
                            
                            if flag:#predict risk exist
                                for dict in bbox_data:    #check each actor that actually exist in current frame
                                    actor_id = dict['actor_id']
                                    predict = False    # if not in tracker then predict false
                                    if str(actor_id) in track_data:
                                        n = track_data[str(actor_id)]
                                        if (n < tracking) and (s_r[j, n] > thres):    #instance thres check
                                            predict = True

                                    if s_type in gt_stype:#and pos
                                        if predict:
                                            if (actor_id == gt_cause_id):
                                                result[accum-1, k, 0] += 1    # tp
                                            else:
                                                result[accum-1, k, 1] += 1   # fp
                                        else:
                                            if (actor_id == gt_cause_id):
                                                result[accum-1, k, 3] += 1    # fn
                                            else:
                                                result[accum-1, k, 2] += 1   # tn
                                    else:
                                        if predict:
                                            result[accum-1, k, 1] += 1   # fp
                                        else:
                                            result[accum-1, k, 2] += 1   # tn
                            else:# predict no risk
                                if s_type in gt_stype: # and pos
                                    result[accum-1, k, 3] += 1    # fn
                                    result[accum-1, k, 2] += len(bbox_data)-1    # tn
                                else:
                                    result[accum-1, k, 2] += len(bbox_data)    # tn
    print(test)
    calculate(result, threshold, accumulate)


def calculate(T, thres, accumulate):
    Recall = np.zeros((len(accumulate), len(thres)))
    Precision = np.zeros((len(accumulate), len(thres)))
    F1 = np.zeros((len(accumulate), len(thres)))

    for k, th in enumerate(thres):
        for accum in accumulate:
            print(th, accum)
            tp, fp, tn, fn = T[accum-1, k]
            print("\ttp:", tp, "fp:", fp, "tn:", tn, "fn:", fn)

            pre = tp / (tp+fp) if (tp+fp)>0 else 0
            rec = tp / (tp+fn) if (tp+fn)>0 else 0
            f1 = (2*rec*pre) / (pre+rec) if (pre+rec)>0 else 0
            print(f"\tPrecision: {pre*100:.2f}, Recall: {rec*100:.2f}, F1: {f1*100:.2f}")
            Recall[accum-1, k] = rec
            Precision[accum-1, k] = pre
            F1[accum-1, k] = f1
    
    plt.xlabel('Threshold')
    plt.ylabel('Recall(%)')
    plt.xticks(thres)
    plt.plot(thres, Recall[0]*100, label="accum_frame= 1")
    plt.plot(thres, Recall[1]*100, label="accum_frame= 2")
    plt.plot(thres, Recall[2]*100, label="accum_frame= 3")
    plt.plot(thres, Recall[3]*100, label="accum_frame= 4")
    plt.plot(thres, Recall[4]*100, label="accum_frame= 5")
    plt.legend()
    plt.savefig('baseline3_last/ROI_Sweep15/ROI_Recall.png')

    plt.clf()
    plt.xlabel('Threshold')
    plt.ylabel('Precision(%)')
    plt.xticks(thres)
    plt.plot(thres, Precision[0]*100, label="accum_frame= 1")
    plt.plot(thres, Precision[1]*100, label="accum_frame= 2")
    plt.plot(thres, Precision[2]*100, label="accum_frame= 3")
    plt.plot(thres, Precision[3]*100, label="accum_frame= 4")
    plt.plot(thres, Precision[4]*100, label="accum_frame= 5")
    plt.legend()
    plt.savefig('baseline3_last/ROI_Sweep15/ROI_Precision.png')

    plt.clf()
    plt.xlabel('Threshold')
    plt.ylabel('F1-score(%)')
    plt.xticks(thres)
    plt.plot(thres, F1[0]*100, label="accum_frame= 1")
    plt.plot(thres, F1[1]*100, label="accum_frame= 2")
    plt.plot(thres, F1[2]*100, label="accum_frame= 3")
    plt.plot(thres, F1[3]*100, label="accum_frame= 4")
    plt.plot(thres, F1[4]*100, label="accum_frame= 5")
    plt.legend()
    plt.savefig('baseline3_last/ROI_Sweep15/ROI_F1.png')


def ROI_test(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, testloader, _ = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time, validation=True)
    print("finish data loading.")

    gt_stype = ["collision", "interactive", "obstacle"]
    colli_thres = 0.5
    instance_thres = 0.8
    accum = 1
    result = np.zeros((4, 4))# 4:tp fp tn fn
    #0: collision, 1:interactive, 2:obstacle, 3: non-inter
    test = np.zeros(4)
    with torch.no_grad():
        for data in testloader:
            frame, roi, collision_labels, _, path = data
            frame, roi = frame.to(device), roi.to(device)

            cur_len = roi.shape[1]
            net.n_frame = cur_len
            pred_c, _, pred_r, _ = net(roi, frame)

            for i, label in enumerate(collision_labels):
                p_c = pred_c[i].cpu().numpy()#tx2
                # p_r = (pred_r[i] > 0.5).float().cpu().numpy()#txn
                s_r = pred_r[i].cpu().numpy()#txn

                file_name = path[i]
                s_type, s_name, _, weather = file_name.split('/')
                print(file_name)
                if s_type == "collision":
                    test[0] += 1
                elif s_type ==  "interactive":
                    test[1] += 1
                elif s_type == "obstacle":
                    test[2] += 1
                else:
                    test[3] += 1
       
                track_file = open(os.path.join(local_path, file_name, 'tracker.json'))
                track_data = json.load(track_file)
                track_file.close()

                if s_type in gt_stype:
                    _, gt_frame, gt_cause_id = GT_loader.getGTframe(s_type, s_name, weather)
                    k = gt_stype.index(s_type)
                else:
                    gt_frame, gt_cause_id = None, None
                    k = 3

                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])
                # if gt_frame is None:
                #     bbox_list = bbox_list[:clip_time]

                for j in range(cur_len):#test each frame
                    frame_num = int(bbox_list[j].split('.')[0])
                    bbox_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                    bbox_data = json.load(bbox_file)
                    bbox_file.close()

                    if (s_type in gt_stype) and (frame_num == gt_frame):
                        break

                    flag = True #for accumulate condition check
                    # pos = True #frame before start frame is considered negative
                    # if start_frame and frame_num < start_frame:
                    #     pos = False

                    if j < accum-1:
                        flag = False
                    else:
                        for cnt in range(accum):#check accumulate condition
                            if p_c[j-cnt, 1] < colli_thres:
                                flag = False
                                break

                    if flag:#predict risk exist
                        for dict in bbox_data:    #check each actor that actually exist in current frame
                            actor_id = dict['actor_id']
                            predict = False    # if not in tracker then predict false
                            if str(actor_id) in track_data:
                                n = track_data[str(actor_id)]
                                if (n < tracking) and (s_r[j, n] > instance_thres):    #instance thres check
                                    predict = True

                            if s_type in gt_stype:    #gt pos
                                if predict:
                                    if (actor_id == gt_cause_id):
                                        result[k, 0] += 1    # tp
                                    else:
                                        result[k, 1] += 1   # fp
                                else:
                                    if (actor_id == gt_cause_id):
                                        result[k, 3] += 1    # fn
                                    else:
                                        result[k, 2] += 1   # tn
                            else:    #gt neg
                                if predict:
                                    result[k, 1] += 1   # fp
                                else:
                                    result[k, 2] += 1   # tn
                    else:# predict no risk
                        if s_type in gt_stype:
                            result[k, 3] += 1    # fn
                            result[k, 2] += len(bbox_data)-1    # tn
                        else:
                            result[k, 2] += len(bbox_data)    # tn

    print(test)
    for i ,s_type in enumerate(gt_stype):
        tp, fp, tn, fn = result[i]
        print(f"{s_type}--> tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
        pre = tp / (tp+fp) if (tp+fp)>0 else 0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0
        f1 = (2*rec*pre) / (pre+rec) if (pre+rec)>0 else 0
        print(f"\tPrecision: {pre*100:.2f}, Recall: {rec*100:.2f}, F1: {f1*100:.2f}\n")
    
    total = result.sum(axis=0)
    tp, fp, tn, fn = total
    print(f"total--> tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
    pre = tp / (tp+fp) if (tp+fp)>0 else 0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0
    f1 = (2*rec*pre) / (pre+rec) if (pre+rec)>0 else 0
    print(f"\tPrecision: {pre*100:.2f}, Recall: {rec*100:.2f}, F1: {f1*100:.2f}\n")


def ROI_test_scenario(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, testloader, _ = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time, validation=True)
    print("finish data loading.")

    gt_stype = ["collision", "interactive", "obstacle"]
    colli_thres = 0.5
    instance_thres = 0.8
    accum = 1
    ss = ["night","rain","four_way","three_way","straight","low","mid","high"]
    total_count = np.zeros((len(ss),4))
    action_type = ['p_c_l', 'p_c_r', 'p_c_f', 'p_c_u', 
                   'f_f', 'f_l', 'f_r', 'f_sl', 'f_sr', 'f_u', 
                   'j_f', 'sl_sr', 'sr_sl', 'sl_f', 'sr_f', 'sr_r', 
                   'l_f', 'l_l', 'l_r', 'l_u', 'r_f', 'r_r', 'r_l',
                   'u_f', 'u_l', 'u_r']
    action_type_cnt = np.zeros(len(action_type))
    result = np.zeros((len(action_type),4))
    #0: collision, 1:interactive, 2:obstacle, 3: non-inter

    with torch.no_grad():
        for data in testloader:
            frame, roi, collision_labels, _, path = data
            frame, roi = frame.to(device), roi.to(device)

            cur_len = roi.shape[1]
            net.n_frame = cur_len
            pred_c, _, pred_r, _ = net(roi, frame)

            
            for i, label in enumerate(collision_labels):
                p_c = pred_c[i].cpu().numpy()#tx2
                # p_r = (pred_r[i] > 0.5).float().cpu().numpy()#txn
                s_r = pred_r[i].cpu().numpy()#txn

                file_name = path[i]
                s_type, s_name, _, weather = file_name.split('/')
                print(file_name)
                road = s_name.split('_')[1]
                weather_2 = weather.split('_')[0]
                random_actor = weather.split('_')[1]
                k = None
                for x, action in enumerate(action_type):
                    if action in s_name:
                        k = x
                        action_type_cnt[x] += 1
                        break
                if k is None:
                    continue

                track_file = open(os.path.join(local_path, file_name, 'tracker.json'))
                track_data = json.load(track_file)
                track_file.close()

                if s_type in gt_stype:
                    _, gt_frame, gt_cause_id = GT_loader.getGTframe(s_type, s_name, weather)
                else:
                    gt_frame, gt_cause_id = None, None

                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])
                # if gt_frame is None:
                #     bbox_list = bbox_list[:clip_time]


                for j in range(cur_len):#test each frame
                    frame_num = int(bbox_list[j].split('.')[0])
                    bbox_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                    bbox_data = json.load(bbox_file)
                    bbox_file.close()

                    if (s_type in gt_stype) and (frame_num == gt_frame):
                        break

                    flag = True #for accumulate condition check
                    # pos = True #frame before start frame is considered negative
                    # if start_frame and frame_num < start_frame:
                    #     pos = False

                    if j < accum-1:
                        flag = False
                    else:
                        for cnt in range(accum):#check accumulate condition
                            if p_c[j-cnt, 1] < colli_thres:
                                flag = False
                                break

                    if flag:#predict risk exist
                        for dict in bbox_data:    #check each actor that actually exist in current frame
                            actor_id = dict['actor_id']
                            predict = False    # if not in tracker then predict false
                            if str(actor_id) in track_data:
                                n = track_data[str(actor_id)]
                                if (n < tracking) and (s_r[j, n] > instance_thres):    #instance thres check
                                    predict = True

                            if s_type in gt_stype:    #gt pos
                                if predict:
                                    if (actor_id == gt_cause_id):
                                        result[k, 0] += 1    # tp
                                        index = 0
                                    else:
                                        result[k, 1] += 1   # fp
                                        index = 1
                                else:
                                    if (actor_id == gt_cause_id):
                                        result[k, 3] += 1    # fn
                                        index = 3
                                    else:
                                        result[k, 2] += 1   # tn
                                        index = 2
                            else:    #gt neg
                                if predict:
                                    result[k, 1] += 1   # fp
                                    index = 1
                                else:
                                    result[k, 2] += 1   # tn
                                    index = 2

                            if "Night" in weather_2:
                                total_count[0][index] += 1
                            if "Rain" in weather_2:
                                total_count[1][index] += 1
                            if random_actor == "low":
                                total_count[5][index] += 1
                            elif random_actor == "mid":
                                total_count[6][index] += 1
                            else:
                                total_count[7][index] += 1
                            if road[0] == 'i':
                                total_count[2][index] += 1
                            elif road[0] == 't':
                                total_count[3][index] += 1
                            elif road[0] == 's':
                                total_count[4][index] += 1
                    else:# predict no risk
                        if s_type in gt_stype:
                            result[k, 3] += 1    # fn
                            result[k, 2] += len(bbox_data)-1    # tn
                            index = 3
                            if "Night" in weather_2:
                                total_count[0][index] += 1
                            if "Rain" in weather_2:
                                total_count[1][index] += 1
                            if random_actor == "low":
                                total_count[5][index] += 1
                            elif random_actor == "mid":
                                total_count[6][index] += 1
                            else:
                                total_count[7][index] += 1
                            if road[0] == 'i':
                                total_count[2][index] += 1
                            elif road[0] == 't':
                                total_count[3][index] += 1
                            elif road[0] == 's':
                                total_count[4][index] += 1
                            index = 2
                            if "Night" in weather_2:
                                total_count[0][index] += len(bbox_data)-1
                            if "Rain" in weather_2:
                                total_count[1][index] += len(bbox_data)-1
                            if random_actor == "low":
                                total_count[5][index] += len(bbox_data)-1
                            elif random_actor == "mid":
                                total_count[6][index] += len(bbox_data)-1
                            else:
                                total_count[7][index] += len(bbox_data)-1
                            if road[0] == 'i':
                                total_count[2][index] += len(bbox_data)-1
                            elif road[0] == 't':
                                total_count[3][index] += len(bbox_data)-1
                            elif road[0] == 's':
                                total_count[4][index] += len(bbox_data)-1
                        else:
                            result[k, 2] += len(bbox_data)    # tn
                            index = 2
                            if "Night" in weather_2:
                                total_count[0][index] += len(bbox_data)
                            if "Rain" in weather_2:
                                total_count[1][index] += len(bbox_data)
                            if random_actor == "low":
                                total_count[5][index] += len(bbox_data)
                            elif random_actor == "mid":
                                total_count[6][index] += len(bbox_data)
                            else:
                                total_count[7][index] += len(bbox_data)
                            if road[0] == 'i':
                                total_count[2][index] += len(bbox_data)
                            elif road[0] == 't':
                                total_count[3][index] += len(bbox_data)
                            elif road[0] == 's':
                                total_count[4][index] += len(bbox_data)


    for i, action in enumerate(action_type):
        tp, fp, tn, fn = result[i]
        print(f"{action}: {action_type_cnt[i]} --> tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
        pre = tp / (tp+fp) if (tp+fp)>0 else 0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0
        f1 = (2*rec*pre) / (pre+rec) if (pre+rec)>0 else 0
        print(f"\tPrecision: {pre*100:.2f}, Recall: {rec*100:.2f}, F1: {f1*100:.2f}\n")
    for i in range(len(ss)):
        tp, fp, tn, fn = total_count[i]
        print(f"{ss[i]} --> tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
        pre = tp / (tp+fp) if (tp+fp)>0 else 0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0
        f1 = (2*rec*pre) / (pre+rec) if (pre+rec)>0 else 0
        print(f"\tPrecision: {pre*100:.2f}, Recall: {rec*100:.2f}, F1: {f1*100:.2f}\n")


def risk_object(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):#for TTC
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    # net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, _, valloader = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time, validation=True)
    print("finish data loading.")

    gt_stype = ["collision", "interactive", "obstacle"]
    threshold = [0.3, 0.4, 0.5, 0.6 ,0.7, 0.8]
    accumulate = [1, 2, 3, 4]
    
    for accum in accumulate:
        for thres in threshold:
            out_p = {}
            out_n = {}

            with torch.no_grad():
                for data in valloader:
                    frame, roi, collision_labels, risk_labels, path = data
                    frame, roi = frame.to(device), roi.to(device)

                    cur_len = roi.shape[1]
                    # print(cur_len, end=' ')
                    net.n_frame = cur_len
                    pred_c, _, pred_r, _ = net(roi, frame)

                    for i, label in enumerate(collision_labels):#batch=1, for dif frame len
                        temp_1 = {}
                        p_c = pred_c[i].cpu().numpy()#tx2
                        # p_r = (pred_r[i] > 0.5).float().cpu().numpy()#txn
                        s_r = pred_r[i].cpu().numpy()#txn

                        file_name = path[i]
                        s_type, s_name, _, weather = file_name.split('/')
                        # print(file_name)
                        
                        track_file = open(os.path.join(local_path, file_name, 'tracker_inverse.json'))
                        track_data = json.load(track_file)
                        track_file.close()

                        if s_type in gt_stype:
                            gt_frame, agent_id = GT_loader.getGTframe(s_type, s_name, weather)
                        else:
                            gt_frame = None

                        bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                        first_frame = int(bbox_list[0].split('.')[0])
                        if gt_frame is None:
                            bbox_list = bbox_list[:clip_time]

                        for j in range(cur_len):#test each frame
                            frame_num = int(bbox_list[j].split('.')[0])
                            if (s_type in gt_stype) and (frame_num == gt_frame):
                                break

                            temp_2 = {}
                            risk_list = []
                            flag = True #for accumulate condition check

                            if j < accum-1:
                                flag = False
                            else:
                                for cnt in range(accum):#check accumulate condition
                                    if p_c[j-cnt, 1] < 0.5:
                                        flag = False
                                        break
                            
                            if flag:
                                for n in range(len(track_data)):#num of actual actors
                                    if n >= tracking:
                                        break
                                    if s_r[j, n] > thres:
                                        risk_list.append(track_data[str(n)])

                            if len(risk_list) > 0:
                                temp_2['True'] = risk_list
                            else:
                                temp_2['False'] = risk_list
                            
                            temp_1[frame_num] = temp_2

                        if s_type not in gt_stype:
                            out_n[file_name] = temp_1
                        else:
                            out_p[file_name] = temp_1 

            with open(f'baseline3_last/{accum}-{thres}/RA_baseline3_out_pos.json', 'w') as f:
                json.dump(out_p, f)
                print(f"\nT={accum}, Thres={thres}, positive, done.")
            with open(f'baseline3_last/{accum}-{thres}/RA_baseline3_out_neg.json', 'w') as f:
                json.dump(out_n, f)
                print(f"T={accum}, Thres={thres}, negative, done.")
    print("finish.")


def TTC_test(batch_size, model_path, local_path, nas_path, tracking, clip_time=60):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = baseline3_model.Supervised(device, n_obj=tracking, n_frame=clip_time, features_size=256*7*7).to(device)
    # net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    _, testloader, _ = carla_dataset.get_dataset_loader(3, local_path, nas_path, tracking, batch_size, detection_collate, clip_time, validation=True)
    print("finish data loading.")

    gt_stype = ["collision", "interactive", "obstacle"]
    colli_thres = 0.5
    instance_thres = 0.8
    accum = 1
    out_p = {}
    out_n = {}

    with torch.no_grad():
        for data in testloader:
            frame, roi, collision_labels, _, path = data
            frame, roi = frame.to(device), roi.to(device)

            cur_len = roi.shape[1]
            net.n_frame = cur_len
            pred_c, _, pred_r, _ = net(roi, frame)

            for i, label in enumerate(collision_labels):#batch=1, for dif frame len
                temp_1 = {}
                p_c = pred_c[i].cpu().numpy()#tx2
                # p_r = (pred_r[i] > 0.5).float().cpu().numpy()#txn
                s_r = pred_r[i].cpu().numpy()#txn

                file_name = path[i]
                s_type, s_name, _, weather = file_name.split('/')
                print(file_name)
                
                track_file = open(os.path.join(local_path, file_name, 'tracker_inverse.json'))
                track_data = json.load(track_file)
                track_file.close()

                if s_type in gt_stype:
                    gt_frame, agent_id = GT_loader.getGTframe(s_type, s_name, weather)
                else:
                    gt_frame = None

                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])
                if gt_frame is None:
                    bbox_list = bbox_list[:clip_time]

                for j in range(cur_len):#test each frame
                    frame_num = int(bbox_list[j].split('.')[0])
                    if (s_type in gt_stype) and (frame_num == gt_frame):
                        break

                    temp_2 = {}
                    risk_list = []
                    flag = True #for accumulate condition check

                    if j < accum-1:
                        flag = False
                    else:
                        for cnt in range(accum):#check accumulate condition
                            if p_c[j-cnt, 1] < colli_thres:
                                flag = False
                                break
                    
                    if flag:
                        for n in range(len(track_data)):#num of actual actors
                            if n >= tracking:
                                break
                            if s_r[j, n] > instance_thres:
                                risk_list.append(track_data[str(n)])

                    if len(risk_list) > 0:
                        temp_2['True'] = risk_list
                    else:
                        temp_2['False'] = risk_list
                    
                    temp_1[frame_num] = temp_2

                if s_type not in gt_stype:
                    out_n[file_name] = temp_1
                else:
                    out_p[file_name] = temp_1 

    with open(f'baseline3_last/TTC_test/RA_baseline3_out_pos.json', 'w') as f:
        json.dump(out_p, f)
        print(f"\npositive, done.")
    with open(f'baseline3_last/TTC_test/RA_baseline3_out_neg.json', 'w') as f:
        json.dump(out_n, f)
        print(f"negative, done.")
    print("finish.")


if __name__ == '__main__':
    args = carla_dataset.get_parser()
    args = args.parse_args()
    learning_rate = args.lr
    batch_size = args.batch
    n_epoch = args.epoch
    model_path = args.model
    local = args.localpath
    nas = args.naspath
    # seed = args.seed

    if args.mode == "training":
        training(batch_size, n_epoch, learning_rate, local, nas, 40)
    elif args.mode == "test":
        testing(batch_size, model_path, local, nas, 40)
    elif args.mode == "TTC":
        TTC_test(batch_size, model_path, local, nas, 40)
        # risk_object(batch_size, model_path, local, nas, 40)
    elif args.mode == "ROI":
        # ROI_sweep(batch_size, model_path, local, nas, 40)
        # ROI_test(batch_size, model_path, local, nas, 40)
        ROI_test_scenario(batch_size, model_path, local, nas, 40)


## --localpath /data/carla_dataset/data_collection --naspath /mnt/Final_Dataset/dataset/