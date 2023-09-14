import os
import json
from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

__all__ = ['compute_result',
           'topk_accuracy',
           'topk_recall',
           'tta']


def compute_result(class_index, score_metrics, target_metrics, result_path, result_name,
                   ignore_class=[], save=False, verbose=False):

    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    print(pred_metrics)
    print(target_metrics)

    # Compute ACC (stop)
    correct_stop = np.sum((target_metrics != 0) &
                          (target_metrics == pred_metrics))
    total_stop = np.sum(target_metrics != 0)

    correct_go = np.sum((target_metrics != 1) & (
        target_metrics == pred_metrics))
    total_go = np.sum(target_metrics != 1)

    result['ACC_stop'] = correct_stop / total_stop
    result['ACC_go'] = correct_go / total_go

    if verbose:
        print('ACC_stop: {:.5f}'.format(result['ACC_stop']))
        print('ACC_go: {:.5f}'.format(result['ACC_go']))

    # Compute confusion matrix
    # [ [gt0_pre0, gt0_pre1],
    #   [gt1_pre0, gt1_pre1] ]
    result['confusion_matrix'] = \
        confusion_matrix(target_metrics, pred_metrics).tolist()

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics[target_metrics != 24] == cls).astype(np.int),
                score_metrics[target_metrics != 24, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(
                    class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        with open(os.path.join(result_path, result_name), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(
                os.path.join(result_path, result_name)))

    return result['mAP'], result


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers

    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    cls_accuracy = {}
    for c in classes:
        if c != 0:
            recalls += topk_accuracy(scores, labels,
                                     ks=(k,), selected_class=c)[0]
            cls_accuracy[str(c)] = topk_accuracy(
                scores, labels, ks=(k,), selected_class=c)[0]
    return recalls/(len(classes)-1), cls_accuracy


def tta(scores, labels, k):
    """Implementation of time to action curve"""
    rankings = scores.argsort()[..., ::-1]
    comparisons = rankings == labels.reshape(rankings.shape[0], 1, 1)
    cum_comparisons = np.cumsum(comparisons, 2)
    cum_comparisons = np.concatenate([cum_comparisons, np.ones(
        (cum_comparisons.shape[0], 1, cum_comparisons.shape[2]))], 1)
    time_stamps = np.array([1.0, 0.67, 0.33, 0.0])
    return np.nanmean(time_stamps[np.argmax(cum_comparisons, 1)], 0)[k-1]
