# DEPRECATED: Do not use
# Early Euclidean baseline without proper worldline structure

import torch
import torch.nn as nn

class EuclideanV0(nn.Module):
    def __init__(self, num_slots=9, hidden_dim=32):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
    
    def forward(self, features):
        return self.slots.unsqueeze(0), None
