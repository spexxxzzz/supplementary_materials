import torch
import torch.nn as nn

def spearman_correlation(x, y):
    x_rank = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    y_rank = torch.argsort(torch.argsort(y, dim=-1), dim=-1).float()
    x_mean = x_rank.mean(dim=-1, keepdim=True)
    y_mean = y_rank.mean(dim=-1, keepdim=True)
    numerator = ((x_rank - x_mean) * (y_rank - y_mean)).sum(dim=-1)
    x_std = torch.sqrt(((x_rank - x_mean) ** 2).sum(dim=-1) + 1e-8)
    y_std = torch.sqrt(((y_rank - y_mean) ** 2).sum(dim=-1) + 1e-8)
    return numerator / (x_std * y_std + 1e-8)

def compute_density_knn(features, k=5):
    from scipy.spatial.distance import cdist
    import numpy as np
    features_np = features.detach().cpu().numpy()
    distances = cdist(features_np, features_np)
    kth_distances = np.partition(distances, k, axis=1)[:, k]
    density = 1.0 / (kth_distances + 1e-8)
    return torch.tensor(density, device=features.device)

def hungarian_matching(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return torch.tensor(row_ind), torch.tensor(col_ind)

def adjusted_rand_index(true_labels, pred_labels):
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    return adjusted_rand_score(true_np, pred_np)

def normalized_mutual_info(true_labels, pred_labels):
    from sklearn.metrics import normalized_mutual_info_score
    import numpy as np
    true_np = true_labels.detach().cpu().numpy()
    pred_np = pred_labels.detach().cpu().numpy()
    return normalized_mutual_info_score(true_np, pred_np)
