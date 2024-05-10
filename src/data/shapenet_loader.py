import torch
from torch.utils.data import Dataset
import numpy as np

class ShapeNetLoader(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
    
    def __len__(self):
        return 2400 if self.split == 'train' else 300
    
    def __getitem__(self, idx):
        mesh_path = f"{self.root_dir}/meshes/{idx:06d}.obj"
        annotation_path = f"{self.root_dir}/annotations/{idx:06d}.npy"
        
        points = torch.randn(500, 3)
        labels = torch.randint(0, 3, (500,))
        
        return points, labels
