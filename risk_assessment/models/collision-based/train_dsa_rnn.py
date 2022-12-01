import dsa_rnn
import torch
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time
import zipfile
import os
import carla_dataset
import json
import GT_loader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    frames = None
    rois = None
    labels = []
    file_name = []
    for sample in batch:
        frame_features,roi,collision_label,path = sample
        if frames is not None:
            frames = torch.cat((frames,frame_features.unsqueeze(0)))
            rois = torch.cat((rois,roi.unsqueeze(0)))
        else:
            frames = frame_features.unsqueeze(0)
            rois = roi.unsqueeze(0)
        labels.append(collision_label)
        file_name.append(path)
    labels = np.array(labels)
    return frames,rois,torch.from_numpy(labels), file_name

def training(batch_size,n_epoch,learning_rate,local_path,nas_path,seed,ablation,clip_time,tracking=None):
    name = "baseline2_{}_{}_{}_{}".format(str(seed),str(batch_size),str(ablation),str(clip_time))
    single = False
    if not os.path.isdir(name):
        os.mkdir(name)
    else:
        name = name+"_2"
        os.mkdir(name)
    if tracking is not None:
        object_perframe = tracking
    else:
        object_perframe = 20
    if not single:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cuda:1"
    net = baseline2_model.Baseline_SA(object_perframe,device,ablation,n_frame=clip_time,features_size=256*7*7)  # detectron: 1024 or 256*7*7, maskformer: 256*7*7                                                           # maskformer2: 256x7x7 
    print(net)
    # if single:
    #     net = net.to(device)
    # else:
    #     net= torch.nn.DataParallel(net).to(device)\
    net= torch.nn.DataParallel(net)
    net = net.to(device)
    weights = [1.0, 4]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = baseline2_model.custom_loss(int(clip_time*0.8),class_weights)  
    if single:  
        criterion = criterion.to(device)
    else:
        criterion = torch.nn.DataParallel(criterion).to(device)                                                  
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainloader,testloader,validationloader = carla_dataset.get_dataset_loader(2,local_path,nas_path,tracking,batch_size,detection_collate,clip_time,random_seed=seed)

    total_batch = len(trainloader)
    test_total_batch = len(testloader)
    val_total_batch = len(validationloader)
    best_vloss = 100000.0
    train_list = []
    test_list = []
    val_list = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("Epoch %d/%d"%(epoch+1,n_epoch))
        net.train() 
        start_t = time.time()
        running_loss = 0.0
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            frame,roi,labels,_ = data
            frame = frame.to(device)
            roi = roi.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pred, _, pred_loss= net(roi,frame)
            pred = pred.argmax(dim=2)[:,int(clip_time*0.8)]
            labels = labels.long()
            # Calculate acc, mAP
            tp,fp,tn,fn = carla_dataset.calculate_matric(pred,labels)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            loss = criterion(pred_loss, labels,device).mean()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            print("\tBatch : %d/%d, loss: %f" % (i+1,total_batch,loss.item()),end='\r')
        try:
            recall = (TP/(TP+FN))*100
        except:
            recall = 0
        try:
            precision = (TP/(TP+FP))*100
        except:
            precision = 0
        try:
            F1 = (2*recall*precision)/(precision+recall)
        except:
            F1 = 0
        train_list.append([running_loss/total_batch,recall,precision,F1])
        print('\n\tloss: %f'% train_list[epoch][0])
        print('\tRecall: %f%%' % train_list[epoch][1])
        print('\tPrecision: %f%%' % train_list[epoch][2])
        print('\tF1 score: %f%%' % train_list[epoch][3])
        print("Saving model..")
        model_path = name+ '/model_{}'.format(epoch+1)
        torch.save(net.state_dict(), model_path)
        # Testing
        print("testing...")
        net.eval()
        running_loss = 0.0
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # get the inputs; data is a list of [inputs, labels]
                frame,roi,labels,_ = data
                frame = frame.to(device)
                roi = roi.to(device)
                labels = labels.to(device)
                pred, _, pred_loss= net(roi,frame)
                labels = labels.long()
                loss = criterion(pred_loss, labels,device).mean()
                pred = pred.argmax(dim=2)[:,int(clip_time*0.8)]
                # Calculate acc, mAP
                tp,fp,tn,fn = carla_dataset.calculate_matric(pred,labels)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                # print statistics
                running_loss += loss.item()
                print("\tBatch : %d/%d, loss: %f" % (i+1,test_total_batch,loss.item()),end='\r')
        try:
            recall = (TP/(TP+FN))*100
        except:
            recall = 0
        try:
            precision = (TP/(TP+FP))*100
        except:
            precision = 0
        try:
            F1 = (2*recall*precision)/(precision+recall)
        except:
            F1 = 0
        test_list.append([running_loss/test_total_batch,recall,precision,F1])
        print('\n\tloss: %f'% test_list[epoch][0])
        print('\tRecall: %f%%' % test_list[epoch][1])
        print('\tPrecision: %f%%' % test_list[epoch][2])
        print('\tF1 score: %f%%' % test_list[epoch][3])

        # Valiadating
        if epoch%5==0:
            print("validating...")
            running_loss = 0.0
            TP = 0.0
            FP = 0.0
            TN = 0.0
            FN = 0.0
            with torch.no_grad():
                for i, data in enumerate(validationloader):
                    # get the inputs; data is a list of [inputs, labels]
                    frame,roi,labels,_ = data
                    frame = frame.to(device)
                    roi = roi.to(device)
                    labels = labels.to(device)
                    pred, _, pred_loss= net(roi,frame)
                    labels = labels.long()
                    loss = criterion(pred_loss, labels,device).mean()
                    pred = pred.argmax(dim=2)[:,int(clip_time*0.8)]
                    # Calculate acc, mAP
                    tp,fp,tn,fn = carla_dataset.calculate_matric(pred,labels)
                    TP += tp
                    FP += fp
                    TN += tn
                    FN += fn
                    # print statistics
                    running_loss += loss.item()
                    print("\tBatch : %d/%d, loss: %f" % (i+1,val_total_batch,loss.item()),end='\r')
            try:
                recall = (TP/(TP+FN))*100
            except:
                recall = 0
            try:
                precision = (TP/(TP+FP))*100
            except:
                precision = 0
            try:
                F1 = (2*recall*precision)/(precision+recall)
            except:
                F1 = 0
            val_list.append([running_loss/val_total_batch,recall,precision,F1])
            print('\n\tloss: %f'% val_list[-1][0])
            print('\tRecall: %f%%' % val_list[-1][1])
            print('\tPrecision: %f%%' % val_list[-1][2])
            print('\tF1 score: %f%%' % val_list[-1][3])
        
        print('\tTime taken: ',time.time()-start_t,' seconds')
    train_list = np.array(train_list)
    test_list = np.array(test_list)
    val_list = np.array(val_list)
    print("Training finish.")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(1,n_epoch+1))
    plt.plot(range(1,n_epoch+1),train_list[:,0],color = 'r')
    plt.plot(range(1,n_epoch+1),test_list[:,0], color = 'g')
    plt.plot(range(1,n_epoch+1,5),val_list[:,0], color = 'b')
    plt.legend(["training","testing","validation"])
    plt.savefig(name +'/Loss.png')

    plt.clf()
    plt.ylabel('Recall')
    plt.xlabel('epoch')
    plt.xticks(range(1,n_epoch+1))
    plt.plot(range(1,n_epoch+1),train_list[:,1],color = 'r')
    plt.plot(range(1,n_epoch+1),test_list[:,1], color = 'g')
    plt.plot(range(1,n_epoch+1,5),val_list[:,1], color = 'b')
    plt.legend(["training","testing","validation"])
    plt.savefig(name +'/Recall.png')

    plt.clf()
    plt.ylabel('Precision')
    plt.xlabel('epoch')
    plt.xticks(range(1,n_epoch+1))
    plt.plot(range(1,n_epoch+1),train_list[:,2],color = 'r')
    plt.plot(range(1,n_epoch+1),test_list[:,2], color = 'g')
    plt.plot(range(1,n_epoch+1,5),val_list[:,2], color = 'b')
    plt.legend(["training","testing","validation"])
    plt.savefig(name +'/Precision.png')

    plt.clf()
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.plot(range(1,n_epoch+1),train_list[:,3],color = 'r')
    plt.plot(range(1,n_epoch+1),test_list[:,3], color = 'g')
    plt.plot(range(1,n_epoch+1,5),val_list[:,3], color = 'b')
    plt.legend(["training","testing","validation"])
    plt.savefig(name +'/F-score.png')
    return

def testing(batch_size, model_path, local_path,nas_path,ablation,tracking=None,clip_time=60):
    if tracking is not None:
        object_perframe = tracking
    else:
        object_perframe = 20
    device = "cuda:0"#torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    net = baseline2_model.Baseline_SA(object_perframe,device,ablation,n_frame=60,features_size=256*7*7) 
    net= torch.nn.DataParallel(net,device_ids=[0]).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    weights = [1.0, 4.0]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = baseline2_model.custom_loss(int(clip_time*0.8),class_weights)    
    criterion = torch.nn.DataParallel(criterion,device_ids=[0]).to(device)    
    trainloader,testloader,validationloader = carla_dataset.get_dataset_loader(2,local_path,nas_path,tracking,batch_size,detection_collate,clip_time)
    total_batch = len(trainloader)
    test_total_batch = len(testloader)
    val_total_batch = len(validationloader)

    with torch.no_grad():
        running_loss = 0.0
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        for i,data in enumerate(validationloader):
            frame,roi,labels,_ = data
            frame = frame.to(device)
            roi = roi.to(device)
            labels = labels.to(device)
            pred, _, pred_loss= net(roi,frame)
            labels = labels.long()
            loss = criterion(pred_loss, labels,device).mean()
            pred = pred.argmax(dim=2)[:,int(clip_time*0.8)]
            # Calculate acc, mAP
            tp,fp,tn,fn = carla_dataset.calculate_matric(pred,labels)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            # print statistics
            running_loss += loss.item()
            print(tp,fp,tn,fn,"\tBatch : %d/%d, loss: %f" % (i+1,val_total_batch,loss.item()),end='\r')
        print("Validation: ")
        print("\tTP:",TP)
        print("\tFP:",FP)
        print("\tTN:",TN)
        print("\tFN:",FN)
        




def load_CARLA_scenario(path):
    img_archive = zipfile.ZipFile(os.path.join(path,'rgb','front.zip'), 'r')
    zip_file_list = img_archive.namelist()
    img_file_list = sorted(zip_file_list)[1:] # the first element is a folder
    # Read bbox
    bbox_archive = zipfile.ZipFile(os.path.join(path,'bbox','front.zip'), 'r')
    zip_file_list = bbox_archive.namelist()
    bbox_file_list = sorted(zip_file_list)[2:]
    # Align frame number
    index = -1
    while img_file_list[index][-7:-4]!=bbox_file_list[-1][-8:-5]:
        index -= 1
    if index!=-1:
        img_file_list = img_file_list[:index+1]
    img_file_list = img_file_list[-100:]
    return img_archive, img_file_list

def demo(batch_size, model_path, local_path,nas_path,ablation,tracking=None,clip_time=60,write_video=False):
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if tracking is not None:
        object_perframe = tracking
    else:
        object_perframe = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = baseline2_model.Baseline_SA(object_perframe,device,ablation,n_frame=60,features_size=256*7*7) 
    net= torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    _,_,validationloader = carla_dataset.get_dataset_loader(2,local_path,nas_path,tracking,batch_size,detection_collate,clip_time,validation=True)
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    with torch.no_grad():
        for data in validationloader:
            frame,roi,labels,path = data
            preds, probs, _= net(roi,frame)
            for i,label in enumerate(labels):
                if label:
                    out = None
                    pred = preds[i].cpu().numpy()
                    weight = probs[i].cpu().numpy()
                    file_name = path[i]
                    plt.plot(pred[:60,1],linewidth=3.0)
                    plt.ylim(0, 1)
                    plt.ylabel('Probability')
                    plt.xlabel('Frame')
                    plt.savefig(os.path.join(local_path,file_name,'collision_prob.png'))
                    plt.clf()
                    new_weight = weight* 255
                    counter = 0 
                    print(file_name)
                    if write_video:
                        out = cv2.VideoWriter(os.path.join(local_path,file_name,'baseline2_demo.mp4'), fourcc, 20.0, (640,  360))
                    # img files
                    img_archive = zipfile.ZipFile(os.path.join(nas_path,file_name,'rgb','front.zip'), 'r')
                    zip_file_list = img_archive.namelist()
                    img_file_list = sorted(zip_file_list)[1:]
                    # bbox files
                    bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                    # crop
                    history = open(os.path.join(nas_path,file_name,'collision_history.json'))
                    collision_data = json.load(history)[0]
                    collision_frame = collision_data['frame']
                    history.close()
                    first_frame = int(bbox_list[0].split('.')[0])
                    start,end = collision_frame-first_frame-int(clip_time*0.8),collision_frame-first_frame+int(clip_time*0.2)
                    
                    img_file_list = img_file_list[start:end]
                    bbox_list = bbox_list[start:end]
                    # read first img
                    frame = img_archive.read(img_file_list[counter])
                    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                    frame = cv2.resize(frame, (640, 360))
                    while True:
                        # frame = cv2.resize(frame,(640,320),interpolation=cv2.INTER_LINEAR)
                        attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                        now_weight = new_weight[counter,:]
                        json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[counter]))
                        datas = json.load(json_file)
                        for i,data in enumerate(datas):
                            box = data['box']
                            box = np.round(np.array(box)*0.5)
                            actor_id = data['actor_id']
                            if now_weight[i]/255.0>0.4:
                                cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
                            else:
                                cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
                            cv2.putText(frame,str(round(now_weight[i]/255.0*10000)/10000),(int(box[2]),int(box[3])), font, 0.4,(0,0,255),1,cv2.LINE_AA)
                            attention_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = now_weight[i]
                            cv2.putText(frame,str(actor_id),(int(box[0]),int(box[1])), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                        attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                        dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                        cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
                        out.write(dst)
                        # cv2.imshow('result',dst)
                        # time.sleep(0.05)
                        # c = cv2.waitKey(50)
                        # if c == ord('q') and c == 27:
                        #     break
                        counter += 1
                        if counter == clip_time:
                            break
                        frame = img_archive.read(img_file_list[counter])
                        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                        frame = cv2.resize(frame, (640, 360))
            
                    if out is not None:
                        out.release()

                # if True:   
                #     frame = cv2.imread(os.path.join(file_name,'rgb','front', png_list[70]))
                #     new_bboxes = bboxes[70]
                #     now_weight = new_weight[70,:]
                #     actor = []
                #     risk_score = []
                #     bar_color = []
                #     for num_box in range(20):
                #         try:
                #             color = colors[num_box % len(colors)]
                #             color_2 = [i  for i in color]
                #             color = [i * 255 for i in color]
                #             box = new_bboxes[num_box]['box']
                #             bar_color.append(tuple(color_2[::-1]))
                #             actor.append(new_bboxes[num_box]['actor_id'])
                #             risk_score.append(round(now_weight[num_box]/255.0*10000)/10000)
                #         except :
                #             break
                #         cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
                #         cv2.putText(frame,str(new_bboxes[num_box]['actor_id']),(int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,200),1,cv2.LINE_AA)
                # cv2.imshow('acc',frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # df = pd.DataFrame(actor, columns=['Object_id', 'Risk_score'])
                # ax = sns.barplot(x='Object_id', y='Risk_score', data=df,palette=bar_color)
                # ax.set_title('Risks')
                # x_pos = np.arange(len(actor))
                # plt.bar(x_pos, risk_score, color=bar_color)
                # plt.xticks(x_pos, actor)
                # plt.show()

def TTC(batch_size, model_path, local_path,nas_path,ablation,th,accumulate_frame,tracking=None,clip_time=60):
    '''
        out: List[List[Dict[Dict[Dict[bool]]]]]
            index:
                first list: 0~3 (accumulate = 1,2,3,4)
                second list: 0~12 (collision threshold = 0.3,0.35,0.40,...,0.9)
    '''
    
    temppp = int(input("Testing input 1, Validating input 2: "))
    testing = True if temppp == 1 else 2
    single = True
    if tracking is not None:
        object_perframe = tracking
    else:
        object_perframe = 20
    if not single:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cuda:0"
    net = baseline2_model.Baseline_SA(object_perframe,device,ablation,n_frame=clip_time,features_size=256*7*7) 
    # if single:
    #     net = net.to(device)
    # else:
    #     net= torch.nn.DataParallel(net).to(device)\
    net= torch.nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    # print(dict(net.named_parameters()))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    _,testingloader,validationloader = carla_dataset.get_dataset_loader(2,local_path,nas_path,tracking,batch_size,detection_collate,clip_time,validation=True)
    loader = testingloader if testing else validationloader
    out = {}
    thresh = np.linspace(0.3,0.9,num=13)
    temp = [{} for i in range(13)] # 13 th
    temp2 = [copy.deepcopy(temp) for i in range(4)] # 4 T
    out = [copy.deepcopy(temp2) for i in range(4)] # 4 scenario type
    obj_th = 0.2
    with torch.no_grad():
        for data in loader:
            frame,roi,labels,path = data
            frame,roi,labels = frame.to(device), roi.to(device), labels.to(device)
            curr_length = roi.shape[1]
            print(curr_length,end=' ')
            net.module.n_frame = curr_length
            preds, probs, _= net(roi,frame)
            for i,label in enumerate(labels):
                # print(i,end='\r')
                pred = preds[i].cpu().numpy()
                weight = probs[i].cpu().numpy()
                file_name = path[i]

                temp = file_name.split('/')
                if temp[0] == "interactive" :
                    s_type = 0
                elif temp[0] == "collision" :
                    s_type = 1 
                elif temp[0] == "non-interactive":
                    s_type = 2
                else:
                    s_type = 3 # obstacle

                print(file_name)
                if s_type != 2:
                    GT_start, GT_frame, agent_id  = GT_loader.getGTframe(temp[0],temp[1],temp[3])
                else:
                    GT_frame = curr_length
                    GT_start = 1000
                # bbox files
                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])

                # if GT_frame is None:
                #     bbox_list = bbox_list[:clip_time]

                for k,th in enumerate(thresh):
                    if k >0:
                        break
                    for accumulate in range(1,5):
                        if accumulate >1:
                            break
                        temp_1 = {}
                        if accumulate >1 :
                            for j in range(accumulate):
                                frame_num = str(int(bbox_list[j].split('.')[0]))
                                temp_1[frame_num] = False
                
                        for j in range(accumulate-1,GT_frame-first_frame):
                            flag = True
                            frame_num = str(int(bbox_list[j].split('.')[0]))
                            for index in range(accumulate):
                                if pred[j-index][1] <th:
                                    flag = False
                                    break
                            json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                            datas = json.load(json_file)
                            temp = {}
                            for jj,data in enumerate(datas):
                                # try:
                                #     if datas[np.argmax(weight[j])]['actor_id'] != agent_id:
                                #         flag = False
                                # except:
                                #     flag = False
                                if jj>=20:
                                    temp[data['actor_id']] = 0.0
                                    continue
                                if flag:
                                    # temp[data['actor_id']] = True if weight[j][jj]>obj_th else False
                                    temp[data['actor_id']] = float(weight[j][jj])
                                else:
                                    temp[data['actor_id']] = 0.0
                                    # temp[data['actor_id']] = False
                            temp_1[frame_num] = temp
                        out[s_type][accumulate-1][k][file_name] = temp_1
    print(out)
    with open('risk_assessment_collision_prediction_baseline2_interactive.json', 'w') as f:
        json.dump(out[0][0], f)
    with open('risk_assessment_collision_prediction_baseline2_collision.json', 'w') as f:
        json.dump(out[1][0], f)
    with open('risk_assessment_collision_prediction_baseline2_non-interactive.json', 'w') as f:
        json.dump(out[2][0], f)
    with open('risk_assessment_collision_prediction_baseline2_obstacle.json', 'w') as f:
        json.dump(out[3][0], f)

def risky_object(batch_size, model_path, local_path,nas_path,ablation,tracking=None,clip_time=60):
    temppp = int(input("Testing input 1, Validating input 2: "))
    testing = True if temppp == 1 else 2
    def count(branch,flag):
        if branch == 1:
            if flag:
                index = 0
            else:
                index = 3

        elif branch == 2:
            if flag:
                index = 1
            else:
                index = 2
                
        else:
            if flag:
                index = 3
            else:
                index = 2
        total_count[s_type][index] += 1
        # scenario type
        if "Night" in weather:
            night[index] += 1
        if "Rain" in weather:
            rain[index] += 1
        if random_actor == "low":
            low[index] += 1
        elif random_actor == "mid":
            mid[index] += 1
        else:
            high[index] += 1
        if road[0] == 'i':
            four_way[index] += 1
        elif road[0] == 't':
            three_way[index] += 1
        elif road[0] == 's':
            straight[index] += 1
        # action type
        if s_type == 1 or s_type == 0:
            action_count[action][index] += 1


    single = True
    if tracking is not None:
        object_perframe = tracking
    else:
        object_perframe = 20
    if not single:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cuda:0"
    net = baseline2_model.Baseline_SA(object_perframe,device,ablation,n_frame=clip_time,features_size=256*7*7) 
    # if single:
    #     net = net.to(device)
    # else:
    #     net= torch.nn.DataParallel(net).to(device)\
    net= torch.nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    # print(dict(net.named_parameters()))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    _,testingloader,validationloader = carla_dataset.get_dataset_loader(2,local_path,nas_path,tracking,batch_size,detection_collate,clip_time,validation=True)
    loader = testingloader if testing else validationloader
    out = {}
    if testing:
        collision_th = 0.5
        instance_th = 0.3
        accumulate = 1
        night = [0.0]*4
        rain = [0.0]*4
        four_way = [0.0]*4
        three_way = [0.0]*4
        straight = [0.0]*4
        low = [0.0]*4
        mid = [0.0]*4
        high = [0.0]*4
        total_count = np.zeros((4,4))
        # action_type = ['c_l', 'c_r', 'c_f', 'c_u', 'f_f', 'f_l', 'f_r', 'f_sl', 'f_sr', 'f_u', 'j_f', 'l_f', 'l_l',
        #            'l_r', 'l_u', 'r_f', 'r_r', 'r_l', 'sl_f', 'sr_f', 'l_r', 'u_f', 'u_l', 'u_r', 'sl_sr', 'sr_sl']
        action_type = ["c_l","c_r","c_f","c_u","f_f","f_l","f_r","f_sl","f_sr","f_u","j_f",
        "l_f","l_l","l_r","l_u","r_f","r_r","r_l","sl_f","sr_f","sr_r","u_f","u_l","u_r","sl_sr","sr_sl"]
        action_count = {}
        for action in action_type:
            action_count[action] = np.zeros((4))
        ss = ["night","rain","four_way","three_way","straight","low","mid","high"]
        ss_2 = ["interactive","collision","non-interactive","obstacle"]
    else:
        thresh = np.linspace(0.3,0.9,num=13)
        # [TP,FP,TN,FN]
        T_0 = np.zeros((4,13,4)) # interactive
        T_1 = np.zeros((4,13,4)) # collision
        T_2 = np.zeros((4,13,4)) # non-interactive
        T_3 = np.zeros((4,13,4)) # obstacle
    
    s_type = None
    t_time = time.time()
    with torch.no_grad():
        for data in loader:
            time1 = time.time()
            frame,roi,labels,path = data
            frame,roi,labels = frame.to(device), roi.to(device), labels.to(device)
            curr_length = roi.shape[1]
            # print(curr_length,end=' ')
            net.module.n_frame = curr_length
            preds, probs, _= net(roi,frame)
            # print('Time1 = : ',time.time()-time1)
            for i,label in enumerate(labels):
                pred = preds[i].cpu().numpy()
                weight = probs[i].cpu().numpy()
                file_name = path[i]
                print(file_name)
                temp = file_name.split('/')
                road = temp[1].split('_')[1]
                weather = temp[3].split('_')[0]
                random_actor = temp[3].split('_')[1]
                temp_s_type = temp[0]
                if temp_s_type == "interactive" :
                    s_type = 0
                elif temp_s_type == "collision" :
                    s_type = 1 # collision
                elif temp_s_type == "non-interactive":
                    s_type = 2
                else:
                    s_type = 3
                if s_type == 1 or s_type == 0:
                    action = temp[1].split('_')[4:6]
                    action = action[0]+'_'+action[1]
                if temp[0] != "non-interactive":
                    GT_start, GT_frame, agent_id  = GT_loader.getGTframe(temp[0],temp[1],temp[3])
                else:
                    GT_frame = curr_length
                    GT_start = 1000
                # bbox files
                bbox_list = sorted(os.listdir(os.path.join(nas_path,file_name, 'bbox/front')))
                first_frame = int(bbox_list[0].split('.')[0])
                print("GT_start: %d, GT_end: %d, first_frame: %d"% (GT_start,GT_frame,first_frame))
                if GT_frame is None:
                    bbox_list = bbox_list[:clip_time]
                if testing:
                    for j in range(accumulate-1,GT_frame-first_frame):
                        flag = True
                        frame_num = (int(bbox_list[j].split('.')[0]))
                        print(frame_num,end='\r')
                        for index in range(accumulate):
                            if pred[j-index][1] <collision_th:
                                flag = False
                                break
                        json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                        datas = json.load(json_file)

                        for jj,data in enumerate(datas):
                            if not flag:
                                if s_type == 2 :#or frame_num<GT_start:
                                    count(3,False)
                                else:
                                    if data['actor_id'] == agent_id:
                                        count(3,True)
                                    else:
                                        count(3,False)
                                continue
                            if s_type == 2 :#or frame_num<GT_start:
                                if jj>= 20:
                                    count(2,False)
                                else:
                                    if weight[j][jj]>instance_th:
                                        count(2,True)
                                    else:
                                        count(2,False)
                            else:
                                if jj>= 20:
                                    if data['actor_id'] == agent_id:
                                        count(1,False)
                                    else:
                                        count(2,False)
                                else:
                                    if data['actor_id'] == agent_id:
                                        if weight[j][jj]>instance_th:
                                            count(1,True)
                                        else:
                                            count(1,False)
                                    else:
                                        if weight[j][jj]>instance_th:
                                            count(2,True)
                                        else:
                                            count(2,False)
                    print('')
                else:
                    for k,th in enumerate(thresh):
                        for accumulate in range(1,5):
                            flag = False
                            # count = 0
                            for j in range(accumulate-1,GT_frame-first_frame):
                                flag = True
                                frame_num = str(int(bbox_list[j].split('.')[0]))
                                
                                for index in range(accumulate):
                                    if pred[j-index][1] <0.5:
                                        flag = False
                                        break
                                if not flag:
                                    if s_type == 0:
                                        T_0[accumulate-1][k][3] += 1
                                    elif s_type == 1:
                                        T_1[accumulate-1][k][3] += 1
                                    elif s_type == 2:
                                        T_2[accumulate-1][k][2] += 1
                                    else:
                                        T_3[accumulate-1][k][3] += 1
                                    continue
                                json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[j]))
                                datas = json.load(json_file)
                                for jj,data in enumerate(datas):
                                    if s_type == 2:
                                        if jj>= 20:
                                            T_2[accumulate-1][k][2] += 1
                                        else:
                                            if weight[j][jj]>th:
                                                T_2[accumulate-1][k][1] += 1
                                            else:
                                                T_2[accumulate-1][k][2] += 1
                                    else:
                                        if jj>= 20:
                                            if data['actor_id'] == agent_id:
                                                if s_type == 0:
                                                    T_0[accumulate-1][k][3] += 1
                                                elif s_type == 1:
                                                    T_1[accumulate-1][k][3] += 1
                                                else:
                                                    T_3[accumulate-1][k][3] += 1
                                            else:
                                                if s_type == 0:
                                                    T_0[accumulate-1][k][2] += 1
                                                elif s_type == 1:
                                                    T_1[accumulate-1][k][2] += 1
                                                else:
                                                    T_3[accumulate-1][k][2] += 1
                                        else:
                                            if data['actor_id'] == agent_id:
                                                if weight[j][jj]>th:
                                                    if s_type == 0:
                                                        T_0[accumulate-1][k][0] += 1
                                                    elif s_type == 1:
                                                        T_1[accumulate-1][k][0] += 1
                                                    else:
                                                        T_3[accumulate-1][k][0] += 1
                                                else:
                                                    if s_type == 0:
                                                        T_0[accumulate-1][k][3] += 1
                                                    elif s_type == 1:
                                                        T_1[accumulate-1][k][3] += 1
                                                    else:
                                                        T_3[accumulate-1][k][3] += 1
                                            else:
                                                if weight[j][jj]>th:
                                                    if s_type == 0:
                                                        T_0[accumulate-1][k][1] += 1
                                                    elif s_type == 1:
                                                        T_1[accumulate-1][k][1] += 1
                                                    else:
                                                        T_3[accumulate-1][k][1] += 1
                                                else:
                                                    if s_type == 0:
                                                        T_0[accumulate-1][k][2] += 1
                                                    elif s_type == 1:
                                                        T_1[accumulate-1][k][2] += 1
                                                    else:
                                                        T_3[accumulate-1][k][2] += 1

    if testing:
        ll = [night,rain,four_way,three_way,straight,low,mid,high]
        for l,text in zip(ll,ss):
            TP, FP, TN, FN = l
            recall_temp = (TP/(TP+FN))*100
            precision_temp = (TP/(TP+FP))*100
            print(text,end=":\n")
            print("\tPrecision: %f\n\tRecall: %f\n\tF1: %f"%(precision_temp,recall_temp,(2*recall_temp*precision_temp)/(precision_temp+recall_temp)))
        for action in action_type:
            TP, FP, TN, FN = action_count[action]
            recall_temp = (TP/(TP+FN))*100
            precision_temp = (TP/(TP+FP))*100
            print(action,end=":\n")
            print("\tPrecision: %f\n\tRecall: %f\n\tF1: %f"%(precision_temp,recall_temp,(2*recall_temp*precision_temp)/(precision_temp+recall_temp)))
        total_tp = 0.0
        total_fp = 0.0
        total_tn = 0.0
        total_fn = 0.0
        for l,text in zip(total_count,ss_2):
            TP, FP, TN, FN = l
            total_tp += TP
            total_fp += FP
            total_tn += TN
            total_fn += FN
            if text == "non-interactive":
                continue
            recall_temp = (TP/(TP+FN))*100
            precision_temp = (TP/(TP+FP))*100
            print(text,end=":\n")
            print("\tPrecision: %f\n\tRecall: %f\n\tF1: %f"%(precision_temp,recall_temp,(2*recall_temp*precision_temp)/(precision_temp+recall_temp)))
        recall_total = (total_tp/(total_tp+total_fn))*100
        precision_total = (total_tp/(total_tp+total_fp))*100
        print("Precision: %f\nRecall: %f\nF1: %f"%(precision_total,recall_total,(2*recall_total*precision_total)/(precision_total+recall_total)))
        print("Time elapsed: ",time.time()-t_time)
        return
    precision, recall, F1 = [],[],[]
    for i in range(4):
        TP = T_0[i][:,0] + T_1[i][:,0] +T_2[i][:,0] + T_3[i][:,0] 
        FP = T_0[i][:,1] + T_1[i][:,1] +T_2[i][:,1] + T_3[i][:,1] 
        TN = T_0[i][:,2] + T_1[i][:,2] +T_2[i][:,2] + T_3[i][:,2] 
        FN = T_0[i][:,3] + T_1[i][:,3] +T_2[i][:,3] + T_3[i][:,3] 
        recall_temp = (TP/(TP+FN))*100
        precision_temp = (TP/(TP+FP))*100
        precision.append(precision_temp)
        recall.append(recall_temp)
        F1.append((2*recall_temp*precision_temp)/(precision_temp+recall_temp))
    # print('Total: \n\tPrecision: %f\n\tRecall: %f\n\tF1: %f'%(precision[0][0],recall[0][0],F1[0][0]))
    # return
    plt.clf()
    # plt.title("non-Interactive")
    plt.ylabel('Precision(%)')
    plt.xlabel('Threshold')
    plt.xticks(thresh)
    plt.plot(thresh,precision[0],color = 'r')
    plt.plot(thresh,precision[1], color = 'g')
    plt.plot(thresh,precision[2], color = 'b')
    plt.plot(thresh,precision[3], color = 'y')
    plt.legend(["Accumulate_frame = 1","Accumulate_frame = 2","Accumulate_frame = 3","Accumulate_frame = 4"])
    plt.savefig(model_path+'_Precision_ROI.png')

    plt.clf()
    # plt.title("non-Interactive")
    plt.ylabel('Recall(%)')
    plt.xlabel('Threshold')
    plt.xticks(thresh)
    plt.plot(thresh,recall[0],color = 'r')
    plt.plot(thresh,recall[1], color = 'g')
    plt.plot(thresh,recall[2], color = 'b')
    plt.plot(thresh,recall[3], color = 'y')
    plt.legend(["Accumulate_frame = 1","Accumulate_frame = 2","Accumulate_frame = 3","Accumulate_frame = 4"])
    plt.savefig(model_path+'_Recall_ROI.png')

    plt.clf()
    # plt.title("non-Interactive")
    plt.ylabel('F1(%)')
    plt.xlabel('Threshold')
    plt.xticks(thresh)
    plt.plot(thresh,F1[0],color = 'r')
    plt.plot(thresh,F1[1], color = 'g')
    plt.plot(thresh,F1[2], color = 'b')
    plt.plot(thresh,F1[3], color = 'y')
    plt.legend(["Accumulate_frame = 1","Accumulate_frame = 2","Accumulate_frame = 3","Accumulate_frame = 4"])
    plt.savefig(model_path+'_F1_ROI.png')


    # plt.clf()
    # fig = plt.figure(figsize =(20, 8))
    # plt.title("Collision prediction statistc")
    # plt.ylabel('Probability')
    # plt.xlabel('Frame')
    # # plt.xticks(range(1,clip_time))
    # plt.boxplot(stat)
    # plt.savefig('Statistic.png')

                        

                # frame_num = str(int(bbox_list[48].split('.')[0]))
                # json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_list[40]))
                # datas = json.load(json_file)
                # for data,risk in zip(datas,weight[40]):
                #     if data['actor_id'] == agent_id:
                #         if float(risk) >=0.3:
                #             T[0] += 1
                #         else:
                #             F[0] += 1
                #         if float(risk) >=0.4:
                #             T[1] += 1
                #         else:
                #             F[1] += 1
                #         if float(risk) >=0.5:
                #             T[2] += 1
                #         else:
                #             F[2] += 1
                    # if float(risk) >= 0.4:
                    #     if data['actor_id'] == agent_id:
                    #         T[0] += 1
                    #     else:
                    #         F[0] += 1
                    # if float(risk) >= 0.3:
                    #     if data['actor_id'] == agent_id:
                    #         T[1] += 1
                    #     else:
                    #         F[1] += 1
                    # if float(risk) >= 0.5:
                    #     if data['actor_id'] == agent_id:
                    #         T[2] += 1
                    #     else:
                    #         F[2] += 1
    #             temp_1 = {}
    #             for j,bbox_file in enumerate(bbox_list):
    #                 frame_num = str(int(bbox_file.split('.')[0]))
    #                 json_file = open(os.path.join(nas_path,file_name, 'bbox/front', bbox_file))
    #                 datas = json.load(json_file)
    #                 temp_2 = {}
    #                 for data,risk in zip(datas,weight[j]):
    #                     temp_2[str(data['actor_id'])] = float(risk)
    #                 temp_1[frame_num] = temp_2
    #             out[file_name] = temp_1
    # with open('risk_assessment_collision_prediction_baseline2.json', 'w') as f:
    #     json.dump(out, f)

if __name__ == '__main__':
    args = carla_dataset.get_parser()
    args.add_argument(
        '--ablation',
        default=0,
        help="ablation study",
        type=int
    )
    args = args.parse_args()
    learning_rate = args.lr
    batch_size = args.batch
    n_epoch = args.epoch
    model_path = args.model
    local = args.localpath
    nas = args.naspath
    seed = args.seed
    ablation = args.ablation
    clip_time = args.time
    if args.mode == "training":
        training(batch_size, n_epoch, learning_rate, local,nas,seed,ablation,clip_time)
    # elif args.mode == "demo":
    #     demo(batch_size, model_path, local,nas,ablation,write_video=True)
    elif args.mode == "ROI":
        risky_object(batch_size, model_path, local,nas,ablation,clip_time=clip_time)
    elif args.mode == "testing":
        testing(batch_size, model_path, local,nas,ablation)
    elif args.mode == "TTC":
        th = 0.8
        accumulate_frame = 2
        TTC(batch_size, model_path, local,nas,ablation,th,accumulate_frame,clip_time=clip_time)
    


## --localpath /data/carla_dataset/data_collection --naspath /mnt/Final_Dataset/dataset/
## python baseline2.py --localpath /data/carla_dataset/data_collection --naspath /mnt/Final_Dataset/dataset/ --batch 1 --mode guan --model baseline2_40_16_0_loss_2/model_19