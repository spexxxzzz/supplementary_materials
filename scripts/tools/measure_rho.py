#!/usr/bin/env python3

import argparse
import torch
from src.data import ToyDataset, KITTILoader
from src.core.math_utils import compute_density_knn, spearman_correlation

def measure_rho(dataset, k=5):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    all_densities = []
    all_labels = []
    
    for features, labels in loader:
        features = features.squeeze(0)
        labels = labels.squeeze(0)
        densities = compute_density_knn(features, k=k)
        all_densities.append(densities)
        all_labels.append(labels)
    
    all_densities = torch.cat(all_densities)
    all_labels = torch.cat(all_labels)
    
    rho = spearman_correlation(all_densities, all_labels.float())
    return rho.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    
    if args.dataset == 'toy':
        dataset = ToyDataset()
    elif args.dataset == 'kitti':
        dataset = KITTILoader('/data/shared/kitti')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    rho = measure_rho(dataset, k=args.k)
    print(f"ρ = {rho:.3f}")
