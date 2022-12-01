import numpy as np
import json
import argparse

data_type = ['interactive', 'obstacle', 'non-interactive', 'collision']


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


def PIC(root, method, save, attr):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--metric', required=True, choices=["F1", "PIC"])
    parser.add_argument('--path', default="prediction")
    parser.add_argument('--scenario', required=False, default="",
                        choices=["Sunset", "Rain", "Noon", "Night", "low", "mid", "high"])
    parser.add_argument('--model', required=True, choices=["random", "nearest", "kalman-filter", "social_gan",
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
