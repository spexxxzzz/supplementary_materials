import torch
from torch.utils.data import Dataset
import numpy as np

class SpritesDataset(Dataset):
    def __init__(self, num_scenes=1000, seed=42):
        self.num_scenes = num_scenes
        self.rng = np.random.RandomState(seed)
    
    def __len__(self):
        return self.num_scenes
    
    def __getitem__(self, idx):
        points = []
        labels = []
        
        for obj_idx in range(3):
            center = torch.tensor(self.rng.uniform(-2, 2, 2))
            center_point = center + torch.tensor(self.rng.normal(0, 0.1, 2))
            points.append(center_point)
            labels.append(0)
            
            num_parts = self.rng.randint(4, 6)
            for _ in range(num_parts):
                r = self.rng.uniform(0.8, 1.2)
                theta = self.rng.uniform(0, 2 * np.pi)
                part_point = center + torch.tensor([r * np.cos(theta), r * np.sin(theta)]) + torch.tensor(self.rng.normal(0, 0.15, 2))
                points.append(part_point)
                labels.append(1)
                
                num_subparts = self.rng.randint(2, 4)
                for _ in range(num_subparts):
                    subpart_point = part_point + torch.tensor(self.rng.normal(0, 0.08, 2))
                    points.append(subpart_point)
                    labels.append(2)
        
        points = torch.stack(points)
        labels = torch.tensor(labels)
        
        return points, labels
