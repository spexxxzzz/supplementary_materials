import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

KITTI_ROOT = '/mnt/raid0_data/neel/kitti_processed_v4/'
CACHE_DIR = '/mnt/raid0_data/neel/kitti_cache/'
METADATA_CACHE = os.path.join(CACHE_DIR, 'kitti_metadata_v4.pkl')
FEATURE_CACHE_DIR = os.path.join(CACHE_DIR, 'precomputed_features')

class KITTILoader(Dataset):
    def __init__(self, root_dir=None, split='train', use_cache=True, precompute_features=False):
        if root_dir is None:
            root_dir = KITTI_ROOT
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.use_cache = use_cache
        self.precompute_features = precompute_features
        
        self.scan_dir = self.root_dir / 'scans'
        self.annotation_dir = self.root_dir / 'annotations'
        self.hierarchy_dir = self.root_dir / 'hierarchy_annotations'
        
        try:
            with open(METADATA_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
                self.metadata = cache_data['metadata']
                self.scan_indices = cache_data[f'{split}_indices']
                self.density_map = cache_data['density_map']
                print(f"[INFO] Loaded cached metadata from {METADATA_CACHE}")
        except FileNotFoundError:
            print(f"Warning: Fast cache not found at {METADATA_CACHE}. Recomputing (this will take 4 hours)...")
            self.metadata = None
            self.scan_indices = None
            self.density_map = None
            if not os.path.exists(self.scan_dir):
                raise FileNotFoundError(f"KITTI scan directory not found: {self.scan_dir}. Please ensure data is mounted at {KITTI_ROOT}")
        
        self.feature_cache_dir = Path(FEATURE_CACHE_DIR) if use_cache else None
        if self.feature_cache_dir and not self.feature_cache_dir.exists():
            os.makedirs(self.feature_cache_dir, exist_ok=True)
        
        self._load_scan_list()
    
    def _load_scan_list(self):
        if self.scan_indices is not None:
            return
        
        scan_list_file = self.root_dir / f'{self.split}_scans.txt'
        if not scan_list_file.exists():
            raise FileNotFoundError(f"Scan list file not found: {scan_list_file}")
        
        with open(scan_list_file, 'r') as f:
            self.scan_indices = [int(line.strip()) for line in f.readlines()]
    
    def _load_scan(self, scan_idx):
        scan_file = self.scan_dir / f'{scan_idx:06d}.npy'
        if not scan_file.exists():
            raise FileNotFoundError(f"Scan file not found: {scan_file}")
        
        scan_data = np.load(scan_file)
        return torch.from_numpy(scan_data).float()
    
    def _load_annotation(self, scan_idx):
        annotation_file = self.annotation_dir / f'{scan_idx:06d}_parts.npy'
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        annotation = np.load(annotation_file)
        return torch.from_numpy(annotation).long()
    
    def _load_hierarchy(self, scan_idx):
        hierarchy_file = self.hierarchy_dir / f'{scan_idx:06d}_hierarchy.npy'
        if not hierarchy_file.exists():
            return None
        
        hierarchy = np.load(hierarchy_file)
        return torch.from_numpy(hierarchy).long()
    
    def _get_cached_features(self, scan_idx):
        if self.feature_cache_dir is None:
            return None
        
        cache_file = self.feature_cache_dir / f'{scan_idx:06d}_features.pt'
        if cache_file.exists():
            return torch.load(cache_file)
        return None
    
    def _save_cached_features(self, scan_idx, features):
        if self.feature_cache_dir is None:
            return
        
        cache_file = self.feature_cache_dir / f'{scan_idx:06d}_features.pt'
        torch.save(features, cache_file)
    
    def __len__(self):
        if self.scan_indices is not None:
            return len(self.scan_indices)
        return 1842 if self.split == 'train' else 200
    
    def __getitem__(self, idx):
        if self.scan_indices is not None:
            scan_idx = self.scan_indices[idx]
        else:
            scan_idx = idx
        
        cached_features = self._get_cached_features(scan_idx)
        if cached_features is not None:
            features = cached_features
        else:
            scan = self._load_scan(scan_idx)
            if self.precompute_features:
                features = self._extract_features(scan)
                self._save_cached_features(scan_idx, features)
            else:
                features = scan
        
        annotation = self._load_annotation(scan_idx)
        hierarchy = self._load_hierarchy(scan_idx)
        
        if hierarchy is None:
            hierarchy = annotation
        
        density = None
        if self.density_map is not None and scan_idx in self.density_map:
            density = torch.tensor(self.density_map[scan_idx], dtype=torch.float32)
        
        return features, annotation, hierarchy, density
    
    def _extract_features(self, scan):
        from scipy.spatial.distance import cdist
        scan_np = scan.detach().cpu().numpy()
        distances = cdist(scan_np, scan_np)
        k = 5
        kth_distances = np.partition(distances, k, axis=1)[:, k]
        density = 1.0 / (kth_distances + 1e-8)
        density_tensor = torch.from_numpy(density).float().unsqueeze(-1)
        return torch.cat([scan, density_tensor], dim=-1)
