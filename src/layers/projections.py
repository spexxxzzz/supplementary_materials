import torch
import torch.nn as nn

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        return self.norm(self.projection(x))

class TemporalProjection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.projection(x)
