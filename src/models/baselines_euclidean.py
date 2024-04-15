import torch
import torch.nn as nn
from src.layers.binding_slots import WorldlineBinding

class EuclideanWorldline(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False):
        super().__init__()
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.learnable_times = learnable_times
        self.binding = WorldlineBinding(num_objects, num_levels, hidden_dim, learnable_times)
    
    def euclidean_distance(self, x, y):
        return torch.norm(x - y, dim=-1, keepdim=True)
    
    def forward(self, features):
        slots = self.binding()
        distances = []
        for i in range(slots.shape[1]):
            slot = slots[:, i:i+1, :]
            dist = self.euclidean_distance(features, slot)
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        attention = torch.softmax(-distances / 0.1, dim=1)
        return slots, attention

class EuclideanStandard(nn.Module):
    def __init__(self, num_slots=9, hidden_dim=32):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
    
    def forward(self, features):
        distances = torch.cdist(features, self.slots.unsqueeze(0))
        attention = torch.softmax(-distances / 0.1, dim=-1)
        return self.slots.unsqueeze(0), attention
