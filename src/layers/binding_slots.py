import torch
import torch.nn as nn
from src.core.constants import LEVEL_TIMES

class WorldlineBinding(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False):
        super().__init__()
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.learnable_times = learnable_times
        
        self.object_centers = nn.Parameter(torch.randn(num_objects, hidden_dim) * 0.1)
        
        if learnable_times:
            self.level_times = nn.Parameter(LEVEL_TIMES.clone())
        else:
            self.register_buffer('level_times', LEVEL_TIMES.clone())
    
    def forward(self):
        batch_size = 1
        slots = []
        for i in range(self.num_objects):
            mu = self.object_centers[i:i+1]
            for j in range(self.num_levels):
                t = self.level_times[j:j+1]
                slot = torch.cat([t, mu], dim=-1)
                slots.append(slot)
        slots = torch.stack(slots, dim=1)
        return slots
