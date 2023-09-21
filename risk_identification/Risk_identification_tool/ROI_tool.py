import argparse
import json
import os

from utils.utils import read_metadata, filter_roi_scenario
from utils.cal_metric import ROI_evaluation


data_types = ['interactive', 'non-interactive', 'collision', 'obstacle']
Method = ["Random", "Range", "Kalman filter", "Social-GAN",
          "MANTRA", "QCNet", "DSA", "RRL", "BP", "BCP"]


def show_result(metric_result):

    method = metric_result['Method']
    attribute = metric_result['Attribute']
    data_type = metric_result['type']

    TP, FN, FP, TN = metric_result["confusion matrix"].values()
    recall, precision, accuracy, f1_score = metric_result["recall"], metric_result[
        "precision"], metric_result["accuracy"], metric_result["f1-Score"]

    # IDcnt, IDsw, IDsw_rate = metric_result['IDcnt'], metric_result['IDsw'], metric_result['IDsw rate']
    # MOTA = metric_result['MOTA']
    PIC, FA_rate = metric_result['PIC'], metric_result['FA']

    print()

    print(f"Method: {method}\tAttribute: {attribute}, type: {data_type}")
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    print(f"Recall: {recall}, Precision: {precision}, F1-Score: {f1_score}")

    # print(f"Accuracy: {accuracy}")
    # print(f"N: {TP+FN+FP+TN}")
    # print(f"IDcnt: {IDcnt}, IDsw: {IDsw}, IDsw rate: {IDsw_rate}")
    # print(f"MOTA: {MOTA}")

    if data_type == "non-interactive":
        print(f"FA rate: {FA_rate}")
    else:
        print(f"PIC: {PIC}")

    for i in range(1, 4):
        print(f"Consistency in {i}s: {metric_result[f'Consistency_{i}s']}")

    print()
    print("="*60)


def save_result(result_path, data_type, method, metric_result):

    save_folder = os.path.join(result_path, method)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, f"{data_type}.json"), 'w') as f:
        json.dump(metric_result, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="all", required=True, type=str)
    parser.add_argument('--data_type', default='all', required=True, type=str)
    parser.add_argument('--metadata_root', default="./metadata", type=str)
    parser.add_argument('--model_root', default="./model", type=str)
    parser.add_argument('--result_path', default="./ROI_result", type=str)
    parser.add_argument('--save_result', action='store_true', default=False)

    args = parser.parse_args()

    if args.method != 'all':
        Method = [args.method]

    if args.data_type != 'all':
        data_types = [args.data_type]

    for method in Method:

        for data_type in data_types:

            behavior_dict, risky_dict, critical_dict = read_metadata(
                data_type=data_type, metadata_root=args.metadata_root)

            roi_result = filter_roi_scenario(
                data_type, method, attr_list="All", model_root=args.model_root)

            _, metric_result = ROI_evaluation(data_type, method, roi_result,
                                              behavior_dict, risky_dict, critical_dict, attribute="All")

            if args.save_result:
                save_result(args.result_path, data_type, method, metric_result)

            show_result(metric_result)
