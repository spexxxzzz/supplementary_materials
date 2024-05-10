import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class COCOLoader(Dataset):
    def __init__(self, root_dir, split='train', use_dinov2=True):
        self.root_dir = root_dir
        self.split = split
        self.use_dinov2 = use_dinov2
    
    def __len__(self):
        return 5000 if self.split == 'train' else 500
    
    def __getitem__(self, idx):
        image_path = f"{self.root_dir}/images/{idx:06d}.jpg"
        annotation_path = f"{self.root_dir}/annotations/{idx:06d}.npy"
        
        if self.use_dinov2:
            features = torch.randn(100, 768)
        else:
            features = torch.randn(100, 512)
        
        labels = torch.randint(0, 3, (100,))
        
        return features, labels
