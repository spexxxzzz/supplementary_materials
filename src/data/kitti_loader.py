import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class KITTILoader(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.meta_path = '/data/shared/kitti_meta_v4.csv'
        
        try:
            self.metadata = pd.read_csv(self.meta_path)
        except FileNotFoundError:
            print(f"Warning: {self.meta_path} not found. Using placeholder data.")
            self.metadata = None
        
        self.scans = []
        self.annotations = []
    
    def __len__(self):
        return 1842 if self.split == 'train' else 200
    
    def __getitem__(self, idx):
        if self.metadata is None:
            scan = torch.randn(100, 3)
            annotation = torch.randint(0, 3, (100,))
            return scan, annotation
        
        scan_path = f"{self.root_dir}/scans/{idx:06d}.pcd"
        annotation_path = f"{self.root_dir}/annotations/{idx:06d}.npy"
        
        scan = torch.from_numpy(np.load(scan_path.replace('.pcd', '.npy')))
        annotation = torch.from_numpy(np.load(annotation_path))
        
        return scan, annotation
