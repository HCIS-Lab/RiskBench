import numpy as np

PRED_SEC = 20
FRAME_PER_SEC = 20


def cal_confusion_matrix(data_type, roi_result, risky_dict, behavior_dict=None):

    TP, FN, FP, TN = 0, 0, 0, 0
    f1_sec = np.zeros((PRED_SEC+1, 4))

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        if data_type != "non-interactive":
            risky_id = risky_dict[scenario_weather][0]
        else:
            risky_id = None

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


def cal_PIC(data_type, roi_result, behavior_dict, risky_dict, critical_dict, EPS=1e-8):

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


def show_result(_type, method, confusion_matrix, IDcnt, IDsw, MOTA, PIC=-1,
                consistency_sec=-1, consistency_sec_cnt=-1, FA=-1, n_frame=-1, attribute=None):

    TP, FN, FP, TN = confusion_matrix
    recall, precision, f1_score = compute_f1(confusion_matrix)

    metric_result = {"Method": method, "Attribute": attribute, "type": _type, "confusion matrix": confusion_matrix.tolist(),
              "recall": f"{recall:.4f}", "precision": f"{precision:.4f}",
              "accuracy": f"{(TP+TN)/(TP+FN+FP+TN):.4f}", "f1-Score": f"{f1_score:.4f}",
              "IDcnt": f"{IDcnt}", "IDsw": f"{IDsw}", "IDsw rate": f"{IDsw/IDcnt:.4f}",
              "MOTA": f"{MOTA:.4f}", "PIC": f"{PIC:.1f}", "FA": f"{FA/n_frame:.4f}",
              "Consistency_1s": f"{consistency_sec[1]/consistency_sec_cnt[1]:.4f}",
              "Consistency_2s": f"{consistency_sec[2]/consistency_sec_cnt[2]:.4f}",
              "Consistency_3s": f"{consistency_sec[3]/consistency_sec_cnt[3]:.4f}"}

    return metric_result


def ROI_evaluation(_type, method, roi_result, behavior_dict=None, risky_dict=None, critical_dict=None, attribute=None):

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
        PIC = cal_PIC(_type, roi_result, behavior_dict, risky_dict, critical_dict)
        consistency_sec, consistency_sec_cnt = cal_consistency(
            _type, roi_result, risky_dict, critical_dict)
    else:
        FA, n_frame = cal_FA(roi_result)

    metric_result = show_result(
        _type, method, confusion_matrix, IDcnt, IDsw, MOTA, PIC, consistency_sec, consistency_sec_cnt, FA, n_frame, attribute)

    return metric_result

