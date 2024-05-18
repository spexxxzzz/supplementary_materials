import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def compute_ari(true_labels, pred_labels):
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    return adjusted_rand_score(true_np, pred_np)

def compute_level_accuracy(true_labels, pred_labels, num_levels=3):
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    correct = (true_np == pred_np).sum()
    total = len(true_np)
    return correct / total

def compute_nmi(true_labels, pred_labels):
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    return normalized_mutual_info_score(true_np, pred_np)

def compute_confusion_matrix(true_labels, pred_labels, num_classes=3):
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(true_np)):
        matrix[true_np[i], pred_np[i]] += 1
    return matrix
