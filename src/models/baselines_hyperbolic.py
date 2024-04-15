import torch
import torch.nn as nn
from src.core.poincare_ops import poincare_distance

class HyperbolicWorldline(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32):
        super().__init__()
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.object_directions = nn.Parameter(torch.randn(num_objects, hidden_dim))
        self.level_radii = nn.Parameter(torch.tensor([0.2, 0.5, 0.8]))
    
    def forward(self, features):
        slots = []
        for i in range(self.num_objects):
            theta = self.object_directions[i] / (torch.norm(self.object_directions[i]) + 1e-8)
            for j in range(self.num_levels):
                r = self.level_radii[j]
                slot = r * theta
                slots.append(slot)
        slots = torch.stack(slots, dim=0).unsqueeze(0)
        
        distances = []
        for i in range(slots.shape[1]):
            slot = slots[:, i:i+1, :]
            dist = poincare_distance(features, slot)
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        attention = torch.softmax(-distances / 0.1, dim=1)
        return slots, attention

class HyperbolicEntailment(nn.Module):
    def __init__(self, num_slots=9, hidden_dim=32, curvature=-1.0):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.1)
    
    def forward(self, features):
        distances = []
        for i in range(self.num_slots):
            slot = self.slots[i:i+1].unsqueeze(0)
            dist = poincare_distance(features, slot) * torch.sqrt(torch.abs(self.curvature))
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        attention = torch.softmax(-distances / 0.1, dim=1)
        return self.slots.unsqueeze(0), attention
