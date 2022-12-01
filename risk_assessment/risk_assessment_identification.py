import argparse
import json
import numpy as np
import os

data_type = ['interactive', 'obstacle', 'non-interactive', 'collision']
dataset_town = '10'


def read_data(root, data_type, method):

    json_path = f'{root}/{method}/RA_{data_type}.json'

    RA_file = open(json_path)
    RA = json.load(RA_file)
    RA_file.close()

    return RA


def getGTframe(root, scenario_type, scenario_weather):

    temp = scenario_weather.split('_')
    scenario_id = '_'.join(temp[:-3])
    weather = '_'.join(temp[-3:-1])+'_'

    def jsonToDict(file_path):
        with open(file_path) as f:
            data = json.load(f)
            f.close()
            return data

    interactive_data = f'{root}/TTC_GT_loader/new_interactive_GT.json'
    collision_data = f'{root}/TTC_GT_loader/new_collision_GT.json'
    obstacle_data = f'{root}/TTC_GT_loader/new_obstacle_GT.json'

    if scenario_type == 'interactive':
        data = jsonToDict(interactive_data)

        start = data[scenario_id][weather]["gt_start_frame"]
        end = data[scenario_id][weather]["gt_end_frame"]
        return [int(data[scenario_id][weather][str(end)])], start, end

    if scenario_type == 'non-interactive':
        return [], None, None

    if scenario_type == 'collision':
        data = jsonToDict(collision_data)

        if scenario_id == "10_t1-2_0_p_j_r_j":
            return None, None, None

        start = data[scenario_id][weather]["gt_start_frame"]
        end = data[scenario_id][weather]["gt_end_frame"]
        return [int(data[scenario_id][weather][str(end)])], start, end

    if scenario_type == 'obstacle':
        data = jsonToDict(obstacle_data)
        start = data[scenario_id][weather]["gt_start_frame"]
        end = data[scenario_id][weather]["gt_end_frame"]
        return [int(data[scenario_id][weather]["nearest_obstacle_id"])], start, end


def roi_testing(root, data_type, RA, attr):

    TP, FN, FP, TN = 0, 0, 0, 0

    for scenario_weather in RA.keys():

        if not attr in scenario_weather:
            continue

        # due to TTC_GT_loader no collision gt "10_t1-2_0_p_j_r_j"
        if '_'.join(scenario_weather.split('_')[:-3]) == "10_t1-2_0_p_j_r_j":
            continue

        gt_cause_id, start_frame, end_frame = getGTframe(root,
                                                         data_type, scenario_weather)

        for frame_id in RA[scenario_weather]:

            if str(int(frame_id)) not in RA[scenario_weather].keys():
                continue

            if not end_frame is None and int(frame_id) > int(end_frame):
                break

            all_actor_id = list(RA[scenario_weather]
                                [str(frame_id)].keys())

            for actor_id in all_actor_id:

                is_risky = RA[scenario_weather][str(frame_id)][actor_id]
                if int(actor_id) in gt_cause_id:
                    if is_risky:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if is_risky:
                        FP += 1
                    else:
                        TN += 1

    return np.array([TP, FN, FP, TN]).astype(int)


def show_result(_type, confusion_matrix, method):
    TP, FN, FP, TN = confusion_matrix
    recall, precision, f1_score = compute_f1(confusion_matrix)

    print(f"Type: {_type}")
    print(f"TP: {TP},  FN: {FN},  FP: {FP},  TN: {TN}")
    print(
        f"Recall: {recall*100:.2f}%  Precision: {precision*100:.2f}%  F1-Score: {f1_score*100:.2f}%")
    # print(f"N: {int(TP+FN+FP+TN)}")
    print(f"Accuracy: {(TP+TN)*100/(TP+FN+FP+TN):.2f}%")
    print("=========================================")

    result = {"Method": method, "type": _type, "confusion matrix": confusion_matrix.tolist(),
              "recall": f"{recall:.4f}", "precision": f"{precision:.4f}",
              "accuracy": f"{(TP+TN)/(TP+FN+FP+TN):.4f}", "f1-Score": f"{f1_score:.4f}"}

    return result, recall, precision, f1_score


def compute_f1(confusion_matrix):

    TP, FN, FP, TN = confusion_matrix

    if TP+FN == 0:
        recall = 0
    else:
        recall = TP / (TP+FN)

    if TP+FP == 0:
        precision = 0
    else:
        precision = TP / (TP+FP)

    if precision+recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall / (precision+recall)

    return recall, precision, f1_score


def F1(root, method, save, attr):
    all_result = []
    confusion_matrix = np.zeros((4)).astype(int)

    for _type in data_type:
        RA = read_data(root, _type, method)

        # confusion_matrix: 1*4 ndarray, [TP, FN, FP, TN]
        cur_confusion_matrix = roi_testing(root, _type, RA, attr)
        confusion_matrix += cur_confusion_matrix

        result, recall, precision, f1_score = show_result(
            _type, cur_confusion_matrix, method)
        all_result.append(result)

    result, recall, precision, f1_score = show_result(
        "all", confusion_matrix, method)
    all_result.append(result)

    if save:
        save_path = f"{root}/result/{method}.json"

    with open(save_path, 'w') as f:
        json.dump(all_result, f, indent=4)


def retrieve_prediction(predictions, id, weather):
    '''
    parse prediction result
    return scneario raw_data
    '''

    raw_data = predictions[id + '_' + weather]
    # print(id + '_'+ weather)
    # assert(len(raw_data) != 0)
    return raw_data


def jsonToDict(file_path):
    with open(file_path) as f:
        data = json.load(f)
        f.close()
        return data


def parse_collision_data(root, method):
    print('====parsing collision data====')
    preds = jsonToDict(os.path.join(
        os.getcwd(), root, method, 'RA_collision.json'))
    collision_GT = jsonToDict(os.path.join(
        os.getcwd(), root, 'TTC_GT_loader', 'new_collision_GT.json'))

    before_GTframe_sample_result = dict()

    scenario_counter = 0
    missing_list = list()

    for s_id, weathers in collision_GT.items():
        if s_id.startswith(dataset_town):
            for weather, gts in weathers.items():
                scenario_counter += 1
                GT_frame_num = gts['gt_end_frame']
                id = gts[str(GT_frame_num)]

                id_str = str(id)

                try:
                    a_pred = retrieve_prediction(preds, s_id, weather)

                except KeyError:
                    missing_list.append([s_id, weather])
                    continue
                frame_prediction_per_scenario = np.zeros(
                    (400, 3), dtype=int)  # 400 frame: [{TP}, {FP}, {FN}]
                for frame, data in a_pred.items():
                    frame_num = int(frame)
                    if frame_num < GT_frame_num:
                        for id, tf in data.items():
                            try:
                                frame_index_before_GT_frame = GT_frame_num-frame_num-1
                                if id == id_str:
                                    if tf == True:  # TP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][0] += 1
                                    else:  # FN
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][2] += 1
                                else:
                                    if tf == True:  # FP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][1] += 1
                            except KeyError:  # no GT id
                                continue
                before_GTframe_sample_result['collision/'+s_id +
                                             '/' + weather] = frame_prediction_per_scenario

    effective_total = scenario_counter-len(missing_list)

    # print('Missing list', missing_list)
    # print('#. of missing_list:',len(missing_list))

    # print('Total_scneario:', scenario_counter)
    # print('Effective scenario amount:', effective_total)
    # print ('====Done====')

    return before_GTframe_sample_result


def parse_interactive_data(root, method):
    print('====parsing interactive data====')
    preds = jsonToDict(os.path.join(os.getcwd(), root,
                       method, 'RA_interactive.json'))
    interactive_GT = jsonToDict(os.path.join(
        os.getcwd(), root, 'TTC_GT_loader', 'new_interactive_GT.json'))

    before_GTframe_sample_result = dict()

    scenario_counter = 0
    missing_list = list()

    for s_id, weathers in interactive_GT.items():
        if s_id.startswith(dataset_town):
            for weather, gts in weathers.items():
                scenario_counter += 1
                GT_frame_num = gts['gt_end_frame']
                id = gts[str(GT_frame_num)]

                id_str = str(id)

                try:
                    a_pred = retrieve_prediction(preds, s_id, weather)

                except KeyError:
                    missing_list.append([s_id, weather])
                    continue
                frame_prediction_per_scenario = np.zeros(
                    (400, 3), dtype=int)  # 400 frame: [{TP}, {FP}, {FN}]
                for frame, data in a_pred.items():
                    frame_num = int(frame)
                    if frame_num < GT_frame_num:
                        for id, tf in data.items():
                            try:
                                frame_index_before_GT_frame = GT_frame_num-frame_num-1
                                if id == id_str:
                                    if tf == True:  # TP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][0] += 1
                                    else:  # FN
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][2] += 1
                                else:
                                    if tf == True:  # FP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][1] += 1
                            except KeyError:  # no GT id
                                continue
                before_GTframe_sample_result['interactive/'+s_id +
                                             '/' + weather] = frame_prediction_per_scenario

    effective_total = scenario_counter-len(missing_list)

    # print('Missing list', missing_list)
    # print('#. of missing_list:',len(missing_list))

    # print('Total_scneario:', scenario_counter)
    # print('Effective scenario amount:', effective_total)
    # print ('====Done====')

    return before_GTframe_sample_result


def parse_obstacle_data(root, method):
    print('====parsing obstacle data====')
    preds = jsonToDict(os.path.join(
        os.getcwd(), root, method, 'RA_obstacle.json'))
    obstacle_GT = jsonToDict(os.path.join(
        os.getcwd(), root, 'TTC_GT_loader', 'new_obstacle_GT.json'))

    before_GTframe_sample_result = dict()

    scenario_counter = 0
    missing_list = list()

    for s_id, weathers in obstacle_GT.items():
        if s_id.startswith(dataset_town):
            for weather, gts in weathers.items():
                scenario_counter += 1

                GT_frame_num = gts['gt_end_frame']
                id_str = gts['nearest_obstacle_id']

                try:
                    a_pred = retrieve_prediction(preds, s_id, weather)

                except KeyError:
                    missing_list.append([s_id, weather])
                    continue

                frame_prediction_per_scenario = np.zeros(
                    (400, 3), dtype=int)  # 400 frame: [{TP}, {FP}, {FN}]
                for frame, data in a_pred.items():
                    frame_num = int(frame)
                    if frame_num < GT_frame_num:
                        for id, tf in data.items():
                            try:
                                frame_index_before_GT_frame = GT_frame_num-frame_num - 1
                                if id == id_str:
                                    if tf == True:  # TP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][0] += 1
                                    else:  # FN
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][2] += 1
                                else:
                                    if tf == True:  # FP
                                        frame_prediction_per_scenario[frame_index_before_GT_frame][1] += 1
                            except KeyError:  # no GT id
                                continue
                before_GTframe_sample_result['obstacle/' + s_id +
                                             '/' + weather] = frame_prediction_per_scenario

    effective_total = scenario_counter-len(missing_list)

    # print('Missing list', missing_list)
    # print('#. of missing_list:',len(missing_list))

    # print('Total_scneario:', scenario_counter)
    # print('Effective scenario amount:', effective_total)
    # print ('====Done====')

    return before_GTframe_sample_result


def FalseAlarm(root, method):
    print('====parsing non-interactive data====')
    preds = jsonToDict(os.path.join(
        os.getcwd(), root, method, 'RA_collision.json'))
    non_interactive_GT = jsonToDict(os.path.join(
        os.getcwd(), root, 'TTC_GT_loader', 'new_collision_GT.json'))

    scenario_counter = 0
    frame_counter = 0

    FP_frame = 0
    FP_scenario = 0

    missing_list = list()
    for scenario_id, variant_scenarios in non_interactive_GT.items():
        if scenario_id.startswith(dataset_town):
            for weather in variant_scenarios:
                scenario_counter += 1
                try:
                    FP_init = FP_frame
                    a_pred = retrieve_prediction(preds, scenario_id, weather)

                    for frame, data in a_pred.items():
                        frame_counter += 1

                        for id in data.keys():
                            if data[id] == True:
                                FP_frame += 1
                                break

                    if FP_init != FP_frame:  # FP_frame increased
                        FP_scenario += 1

                except KeyError:
                    missing_list.append([scenario_id, weather])
                    continue
                except AssertionError:
                    missing_list.append([scenario_id, weather])
                    continue

    FA_rate_per_scenario = float(FP_scenario) / \
        float(scenario_counter-len(missing_list))
    FA_rate_per_frame = float(FP_frame)/float(frame_counter)
    # print('missing_list:', missing_list)
    # print('#. of missing scenarios:', len(missing_list))
    # print('#. of non-interactive scenario:', scenario_counter)

    # print('FA per scenario:', FP_scenario)
    # print('FA rate per scenario:', FA_rate_per_scenario)

    # print('frame_counter:', frame_counter)
    print('FP_frame:', FP_frame)
    print('FA rate per frame:', FA_rate_per_frame)
    print('==========non-interactive end=============')

    return FA_rate_per_scenario, FA_rate_per_frame


def FalseAlarm(root, method, save):
    # print('====False Alarm====')
    preds = jsonToDict(os.path.join(
        os.getcwd(), root, method, 'RA_collision.json'))
    non_interactive_GT = jsonToDict(os.path.join(
        os.getcwd(), root, 'TTC_GT_loader', 'new_collision_GT.json'))

    scenario_counter = 0
    frame_counter = 0

    FP_frame = 0
    FP_scenario = 0

    missing_list = list()
    for scenario_id, variant_scenarios in non_interactive_GT.items():
        if scenario_id.startswith(dataset_town):
            for weather in variant_scenarios:
                scenario_counter += 1
                try:
                    FP_init = FP_frame
                    a_pred = retrieve_prediction(preds, scenario_id, weather)

                    for frame, data in a_pred.items():
                        frame_counter += 1

                        for id in data.keys():
                            if data[id] == True:
                                FP_frame += 1
                                break

                    if FP_init != FP_frame:  # FP_frame increased
                        FP_scenario += 1

                except KeyError:
                    missing_list.append([scenario_id, weather])
                    continue
                except AssertionError:
                    missing_list.append([scenario_id, weather])
                    continue

    FA_rate_per_scenario = float(FP_scenario) / \
        float(scenario_counter-len(missing_list))
    FA_rate_per_frame = float(FP_frame)/float(frame_counter)
    # print('missing_list:', missing_list)
    # print('#. of missing scenarios:', len(missing_list))
    # print('#. of non-interactive scenario:', scenario_counter)

    # print('FA per scenario:', FP_scenario)
    print('FA rate per scenario:', FA_rate_per_scenario)

    # print('frame_counter:', frame_counter)
    # print('FP_frame:', FP_frame)
    print('FA rate per frame:', FA_rate_per_frame)
    # print('==========non-interactive end=============')

    return FA_rate_per_scenario, FA_rate_per_frame


def PIC(root, method, save, attr):
    '''
    Progressive Increasing Cost (PIC):
        Merge results of 3 scenarios including collision, interactive and obstacle.
        lowest_value = 1e-7 (if f-1 score == 0)
    '''
    collision_result = parse_collision_data(root, method)
    interactive_result = parse_interactive_data(root, method)
    obstacle_result = parse_obstacle_data(root, method)
    result_dict = {**collision_result, **interactive_result, **obstacle_result}

    result = np.zeros((400, 3))

    for s_id, pred in result_dict.items():
        if 'non-interactive' not in s_id:
            result += pred

    lowest_value = 1e-7
    # precision
    Y_TP = result[:, 0]
    Y_FP = result[:, 1]
    T_P = Y_TP + Y_FP
    index = np.where(T_P > 0)
    Y_precision = np.zeros(400)
    Y_precision[index] = Y_TP[index] / T_P[index]
    index = np.where(Y_precision == 0)
    Y_precision[index] = lowest_value

    # recall
    Y_TP = result[:, 0]
    Y_FN = result[:, 2]
    P = Y_TP + Y_FN
    index = np.where(P > 0)
    Y_recall = np.zeros(400)
    Y_recall[index] = Y_TP[index] / P[index]
    index = np.where(Y_recall == 0)
    Y_recall[index] = lowest_value

    # F1
    index = np.where((Y_precision + Y_recall) > 0)
    Y_F1 = np.zeros(400)
    Y_F1[index] = 2 * Y_precision[index] * Y_recall[index] / \
        (Y_precision[index] + Y_recall[index])
    index = np.where(Y_F1 == 0)
    Y_F1[index] = lowest_value

    # exponential F1 loss in 3 s
    loss = 0
    for i in range(60):
        loss += -(np.exp(-((i+1)*0.05))*np.log(Y_F1[i]))
    print('F1 loss in 3s:', loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--metric', required=True, choices=["F1", "PIC", "FA"])
    parser.add_argument('--path', default="prediction")
    parser.add_argument('--scenario', required=False, default="",
                        choices=["Sunset", "Rain", "Noon", "Night", "low", "mid", "high"])
    parser.add_argument('--model', required=True, choices=["random", "nearest", "rss", "kalman-filter", "social_gan",
                                                           "mantra", "dsa_rnn", "dsa_rnn_supervised", "single-stage", "two-stage"])

    args = parser.parse_args()

    save = args.save_result
    root = args.path
    attr = args.scenario
    method = args.model

    if args.metric == "F1":
        F1(root, method, save, attr)
    if args.metric == "PIC":
        PIC(root, method, save, attr)
    if args.metric == 'FA':
        FalseAlarm(root, method, save)
