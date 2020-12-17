import numpy as np
import torch

from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve


def compute_batch_accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the accuracy of the predictions over the items in a single batch
    :param o: the logit output of datum in the batch
    :param y: the correct class index of each datum
    :return the percentage of correct predictions as a value in [0,1]
    """
    corrects = (torch.max(o, 1)[1].view(y.size()).data == y.data).sum()
    accuracy = corrects / y.size()[0]
    return accuracy.item()


def compute_metrics(y_true: np.array, y_pred: np.array, y_1_scores: np.array) -> dict:
    """
    Computes the metrics for the given predictions and labels
    :param y_true: the ground-truth labels
    :param y_pred: the predictions of the model
    :param y_1_scores: the probabilities for the positive class
    :return: the following metrics in a dict:
        * Sensitivity (TP rate) / Specificity (FP rate) / Combined
        * Accuracy / F1 / AUC
    """
    metrics = {}

    if len(np.unique(y_true)) > 1:
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = recall_score(y_true, y_pred, pos_label=0)

        metrics = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "combined": (sensitivity + specificity) / 2,
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_1_scores)
        }

    return metrics


def compute_optimal_roc_threshold(y_true: np.array, y_1_scores: np.array) -> float:
    """
    Computes the optimal ROC threshold
    :param y_true: the ground truth
    :param y_1_scores: the predicted scores for the positive class
    :return: the optimal ROC threshold (defined for more than one sample, else 0.5)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fp_rates, tp_rates, thresholds = roc_curve(y_true, y_1_scores)
    best_threshold, dist = 0.5, 100

    for i, threshold in enumerate(thresholds):
        current_dist = np.sqrt((np.power(1 - tp_rates[i], 2)) + (np.power(fp_rates[i], 2)))
        if current_dist <= dist:
            best_threshold, dist = threshold, current_dist

    return best_threshold


def pprint_metrics(metrics: dict):
    for metric, value in metrics.items():
        print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {}").format(metric, value))


def average_metrics(metrics: list) -> dict:
    return {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
