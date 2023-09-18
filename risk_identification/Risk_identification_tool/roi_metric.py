import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from transpose import ROI_transpose
from collections import OrderedDict


PRED_SEC = 20
FRAME_PER_SEC = 20
TOWN = 10
OPTION = 2

roi_root = "./model"
dataset_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:]

Method = ["random", "nearest", "kalman-filter", "social_gan",
          "mantra", "DSA", "RRL", "single-stage",
          "two-stage", "two-stage+intention", "two-stage+intention+filter", "two-stage+filter",
          "LSTM+GAT", "LSTM+GAT+intention", "LSTM+GAT+state"][:]


attributes = [["", "Night", "Rain", f"{int(TOWN)}_i", f"{int(TOWN)}_t", f"{int(TOWN)}_s", "low", "mid", "high"],
              ["", 'p_c_l', 'p_c_r', 'p_c_f', 'f_f', 'f_l', 'f_r', 'f_sl', 'f_sr', 'f_u', 'j_f', 'l_f', 'l_l',
              'l_r', 'l_u', 'r_f', 'r_r', 'r_l', 'sl_f', 'sr_f', 'l_r', 'u_f', 'u_l', 'u_r', 'sl_sr', 'sr_sl'],
              ["", "c", "t", "m", "b", "p", "4", "2"],
              [""]][OPTION]


def read_data(data_type, method, attr=""):

    json_path = os.path.join(roi_root, method, f"{data_type}.json")

    roi_file = open(json_path)
    roi_result = json.load(roi_file)
    roi_file.close()

    # filter scenario
    all_scnarios = list(roi_result.keys())
    for scenario_weather in all_scnarios:

        is_del = False

        if OPTION != 2:
            if not attr in scenario_weather:
                is_del = True
        elif attr != "":
            actor_type = scenario_weather.split('_')[3]
            if attr == "4":
                if not actor_type in ["c", "t"]:
                    is_del = True
            elif attr == "2":
                if not actor_type in ["m", "b"]:
                    is_del = True
            else:
                if attr != actor_type:
                    is_del = True

        if is_del:
            del roi_result[scenario_weather]

    return roi_result


def cal_confusion_matrix(data_type, roi_result, risky_dict, behavior_dict=None):

    TP, FN, FP, TN = 0, 0, 0, 0
    TOTAL_FRAME = FRAME_PER_SEC*PRED_SEC
    f1_sec = np.zeros((PRED_SEC+1, 4))
    cnt = 0

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        risky_id = risky_dict[scenario_weather][0]

        for frame_id in roi_result[scenario_weather]:

            _TP, _FN, _FP, _TN = 0, 0, 0, 0
            all_actor_id = list(
                roi_result[scenario_weather][str(frame_id)].keys())

            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][str(
                    frame_id)][actor_id]

                if behavior_stop and (str(int(actor_id) % 65536) == risky_id
                                      or str(int(actor_id) % 65536+65536) == risky_id):

                    if is_risky:
                        _TP += 1
                    else:
                        _FN += 1
                else:
                    if is_risky:
                        _FP += 1
                    else:
                        _TN += 1

            # if end_frame - int(frame_id) < TOTAL_FRAME:
            #     f1_sec[(end_frame - int(frame_id))//FRAME_PER_SEC +
            #            1, :] += np.array([_TP, _FN, _FP, _TN])

            TP += _TP
            FN += _FN
            FP += _FP
            TN += _TN

    return np.array([TP, FN, FP, TN]).astype(int), f1_sec


def cal_IDsw(roi_result):

    IDcnt = 0
    IDsw = 0

    for scenario_weather in roi_result.keys():

        pre_frame_info = None

        for frame_id in roi_result[scenario_weather]:

            cur_frame_info = roi_result[scenario_weather][str(frame_id)]
            all_actor_id = list(cur_frame_info.keys())

            for actor_id in all_actor_id:

                IDcnt += 1
                if not pre_frame_info is None:
                    if actor_id in pre_frame_info and cur_frame_info[actor_id] != pre_frame_info[actor_id]:
                        IDsw += 1

            pre_frame_info = cur_frame_info

    return IDcnt, IDsw


def cal_MOTA(cur_confusion_matrix, IDsw, IDcnt):

    FN, FP = cur_confusion_matrix[1:3]
    MOTA = 1-(FN+FP+IDsw)/IDcnt

    return MOTA


def cal_PIC(data_type, roi_result, critical_dict, EPS=1e-8):

    assert data_type != "non-interactive", "non-interactive can not calculate PIC!!!"

    PIC = 0

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        end_frame = critical_dict[scenario_weather]
        risky_id = risky_dict[scenario_weather]

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > int(end_frame):
                break

            all_actor_id = list(roi_result[scenario_weather][frame_id].keys())

            if len(all_actor_id) == 0:
                continue

            TP, FN, FP, TN = 0, 0, 0, 0
            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][frame_id][actor_id]

                if behavior_stop and (str(int(actor_id) % 65536) in risky_id
                                      or str(int(actor_id) % 65536+65536) in risky_id):
                    if is_risky:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if is_risky:
                        FP += 1
                    else:
                        TN += 1

            recall, precision, f1 = compute_f1(
                np.array([TP, FN, FP, TN]).astype(int))

            # exponential F1 loss
            PIC += -(np.exp(-int(end_frame)+int(frame_id))
                     * np.log(f1 + EPS))

    PIC = PIC/len(roi_result.keys())
    return PIC


def cal_consistency(data_type, roi_result, risky_dict, critical_dict, EPS=1e-05):

    FRAME_PER_SEC = 20
    TOTAL_FRAME = FRAME_PER_SEC*3

    # consistency in 0~3 seconds
    consistency_sec = np.ones(4)*EPS
    consistency_sec_cnt = np.ones(4)*EPS

    for scenario_weather in roi_result.keys():

        end_frame = critical_dict[scenario_weather]
        risky_id = str(risky_dict[scenario_weather][0])
        is_risky = [None]*TOTAL_FRAME

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > end_frame:
                break

            if end_frame - int(frame_id) >= TOTAL_FRAME:
                continue

            cur_frame_info = roi_result[scenario_weather][str(frame_id)]
            all_actor_id = list(cur_frame_info.keys())

            if not risky_id in all_actor_id:
                continue

            is_risky[end_frame - int(frame_id)] = cur_frame_info[risky_id]

        for i in range(0, TOTAL_FRAME, FRAME_PER_SEC):

            if np.any(is_risky[i:i+FRAME_PER_SEC] != None):
                consistency_sec_cnt[i//FRAME_PER_SEC+1] += 1

                if not False in is_risky[:i+FRAME_PER_SEC]:
                    consistency_sec[i//FRAME_PER_SEC+1] += 1
                # else:
                #     if i//FRAME_PER_SEC+1 == 1:
                #         print(scenario_weather, end_frame)
            else:
                print(is_risky[i:i+FRAME_PER_SEC])

    return consistency_sec, consistency_sec_cnt


def cal_FA(roi_result):

    FA = 0
    n_frame = 0

    for scenario_weather in roi_result:

        for frame_id in roi_result[scenario_weather]:

            if len(roi_result[scenario_weather][frame_id]) == 0:
                continue

            if True in roi_result[scenario_weather][frame_id].values():
                FA += 1

            n_frame += 1

    return FA, n_frame


def compute_f1(confusion_matrix, EPS=1e-5):

    TP, FN, FP, TN = confusion_matrix

    recall = TP / (TP+FN+EPS)
    precision = TP / (TP+FP+EPS)
    f1_score = 2*precision*recall / (precision+recall+EPS)

    return recall, precision, f1_score


def show_result(_type, method, confusion_matrix, IDcnt, IDsw, MOTA, PIC=-1, consistency_sec=-1, consistency_sec_cnt=-1, FA=-1, n_frame=-1, attribute=None):

    TP, FN, FP, TN = confusion_matrix
    recall, precision, f1_score = compute_f1(confusion_matrix)

    print(
        f"Method: {method}\tAttribute: {attribute}, type: {_type}")
    print(f"TP: {TP},  FN: {FN},  FP: {FP},  TN: {TN}")
    print(
        f"Recall: {recall*100:.2f}%  Precision: {precision*100:.2f}%  F1-Score: {f1_score*100:.2f}%")
    # print(f"N: {int(TP+FN+FP+TN)}")
    # print(f"Accuracy: {(TP+TN)*100/(TP+FN+FP+TN):.2f}%")
    print(f"IDcnt: {IDcnt}, IDsw: {IDsw}, IDsw rate:{IDsw/IDcnt*100:.2f}%")
    print(f"MOTA: {MOTA*100:.2f}%   PIC: {PIC:.1f}")
    print(f"FA rate: {FA/n_frame*100:.2f}%")

    for i in range(1, 4):
        print(
            f"Consistency in {i}s: {consistency_sec[i]/consistency_sec_cnt[i]*100:.2f}%")

    # f1_save = {"f1": f1_score, "Current": {}, "Accumulation": {}}

    # # confusion matrix
    # f1_sec_sum = np.zeros(4)

    # for i in range(1, PRED_SEC+1):

    #     f1_sec_sum += f1_sec[i]
    #     r, p, f1 = compute_f1(f1_sec_sum)
    #     f1_save["Accumulation"][i] = f1

    #     # print(
    #     #     f"F1 in {i}s: {f1*100:.2f}%,\tRecall: {r*100:.2f},\tPrecision: {p*100:.2f},\tID: {int(np.sum(f1_sec_sum))}")

    #     r, p, f1 = compute_f1(f1_sec[i])
    #     acc = f1_sec[i][0]/(f1_sec[i][0]+f1_sec[i][1])
    #     f1_save["Current"][i] = f1

    #     print(
    #         # f"F1 in {i}s: {f1*100:.2f}%,\tRecall: {r*100:.2f},\tPrecision: {p*100:.2f}")
    #         f"F1 in {i}s: {f1*100:.2f}%,\tRecall: {r*100:.2f},\tPrecision: {p*100:.2f},\tPositive Accuracy: {acc*100:.2f}")

    print()

    result = {"Method": method, "Attribute": attribute, "type": _type, "confusion matrix": confusion_matrix.tolist(),
              "recall": f"{recall:.4f}", "precision": f"{precision:.4f}",
              "accuracy": f"{(TP+TN)/(TP+FN+FP+TN):.4f}", "f1-Score": f"{f1_score:.4f}",
              "IDcnt": f"{IDcnt}", "IDsw": f"{IDsw}", "IDsw rate": f"{IDsw/IDcnt:.4f}",
              "MOTA": f"{MOTA:.4f}", "PIC": f"{PIC:.1f}", "FA": f"{FA/n_frame:.4f}",
              "Consistency_1s": f"{consistency_sec[1]/consistency_sec_cnt[1]:.4f}",
              "Consistency_2s": f"{consistency_sec[2]/consistency_sec_cnt[2]:.4f}",
              "Consistency_3s": f"{consistency_sec[3]/consistency_sec_cnt[3]:.4f}"}

    return result, recall, precision, f1_score


def ROI_evaluation(_type, method, roi_result, risky_dict, critical_dict, behavior_dict=None, attribute=None):

    EPS = 1e-05
    PIC = -1
    FA, n_frame = 0, -1
    consistency_sec = np.zeros(4)
    consistency_sec_cnt = np.ones(4)*EPS

    confusion_matrix, f1_sec = cal_confusion_matrix(
        _type, roi_result, risky_dict, behavior_dict)
    IDcnt, IDsw = cal_IDsw(roi_result)
    MOTA = cal_MOTA(confusion_matrix, IDsw, IDcnt)

    if _type != "non-interactive":
        PIC = cal_PIC(_type, roi_result, critical_dict)
        consistency_sec, consistency_sec_cnt = cal_consistency(
            _type, roi_result, risky_dict, critical_dict)
    else:
        FA, n_frame = cal_FA(roi_result)

    if args.verbose:
        metric_result, recall, precision, f1_score = show_result(
            _type, method, confusion_matrix, IDcnt, IDsw, MOTA, PIC, consistency_sec, consistency_sec_cnt, FA, n_frame, attribute)

    if args.save_result:
        save_folder = f"./result/{method}"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        with open(os.path.join(save_folder, f"{_type}_attr={attribute}_result.json"), 'w') as f:
            json.dump(metric_result, f, indent=4)
    return confusion_matrix


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--method', default="all", required=True, type=str)
#     parser.add_argument('--data_type', default='all', type=str)
#     parser.add_argument('--transpose', action='store_true', default=False)
#     parser.add_argument('--threshold', default=None, type=float)
#     parser.add_argument('--topk', default=None, type=int)
#     parser.add_argument('--save_result', action='store_true', default=False)
#     parser.add_argument('--mode', type=str)
    
#     args = parser.parse_args()

#     if args.method != 'all':
#         Method = [args.method]

#     if args.data_type != 'all':
#         data_types = [args.data_type]

#     for _type in data_types:

#         risky_dict = json.load(open(f"GT_risk/{args.mode}/{_type}.json"))
#         critical_dict = json.load(open(f"GT_critical_point/{args.mode}/{_type}.json"))
#         behavior_dict = None

#         if _type in ["interactive", "obstacle"]:
#             behavior_dict = json.load(
#                 open(f"./behavior/{_type}.json"))

#         for method in Method:

#             for attr in attributes:

#                 roi_result = read_data(_type, method, attr=attr)

#                 if len(roi_result) == 0:
#                     continue

#                 if args.transpose:
#                     roi_result = ROI_transpose(method,
#                                             roi_result, args.threshold, args.topk)

#                 ROI_evaluation(_type, method, roi_result,
#                             risky_dict, critical_dict, behavior_dict, attribute=attr)
#                 print("#"*40)

#         print()
        
if __name__ == '__main__':
    EPS = 1e-6
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="all", required=True, type=str)
    parser.add_argument('--data_type', default='all', type=str)
    parser.add_argument('--transpose', action='store_true', default=False)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--topk', default=None, type=int)
    parser.add_argument('--save_result', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--test_thresholds', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.method != 'all':
        Method = [args.method]

    if args.data_type != 'all':
        data_types = [args.data_type]
    else:
        data_types = ['interactive','collision','obstacle','non-interactive']
    
    if args.test_thresholds:
        method = args.method
        recall = []
        F1 = []
        precision = []
        thresholds = np.linspace(0.0,0.95,20)
        for threshold in thresholds:
            result = np.array([0.0]*4)
            for _type in data_types:

                risky_dict = json.load(open(f"GT_risk/{args.mode}/{_type}.json"))
                critical_dict = json.load(open(f"GT_critical_point/{args.mode}/{_type}.json"))
                behavior_dict = None

                if _type in ["interactive", "obstacle"]:
                    behavior_dict = json.load(
                        open(f"./behavior/{_type}.json"))

                for method in Method:

                    roi_result = read_data(_type, method)
                    if args.transpose:
                        roi_result = ROI_transpose(method,
                                                roi_result, threshold, args.topk)

                    res = ROI_evaluation(_type, method, roi_result,
                                risky_dict, critical_dict, behavior_dict)
                    result += np.array(res)
            TP, FN, FP, TN = result
            recall_ = TP / (TP+FN+EPS)
            precision_ = TP / (TP+FP+EPS)
            f1_score = 2*precision_*recall_ / (precision_+recall_+EPS)
            recall.append(recall_)
            F1.append(f1_score)
            precision.append(precision_)
        plt.plot(thresholds,recall)
        plt.plot(thresholds,precision)
        plt.plot(thresholds,F1)
        plt.xlabel("Threshold")
        plt.ylabel("Percantage")
        plt.ylim(0.0,1.0)
        plt.legend(["recall","precision","F1"])
        plt.title(f'Risk identification ({args.mode}set)')
        plt.show()
    else:
        result = np.array([0.0]*4)
        for _type in data_types:

            risky_dict = json.load(open(f"GT_risk/{args.mode}/{_type}.json"))
            critical_dict = json.load(open(f"GT_critical_point/{args.mode}/{_type}.json"))
            behavior_dict = None

            if _type in ["interactive", "obstacle"]:
                behavior_dict = json.load(
                    open(f"./behavior/{_type}.json"))

            for method in Method:

                roi_result = read_data(_type, method)
                if args.transpose:
                    roi_result = ROI_transpose(method,
                                            roi_result, args.threshold, args.topk)

                res = ROI_evaluation(_type, method, roi_result,
                            risky_dict, critical_dict, behavior_dict)
                result += np.array(res)
                print("#"*20)
        TP, FN, FP, TN = result
        recall_ = TP / (TP+FN+EPS)
        precision_ = TP / (TP+FP+EPS)
        f1_score = 2*precision_*recall_ / (precision_+recall_+EPS)
        print("F1:",f1_score)