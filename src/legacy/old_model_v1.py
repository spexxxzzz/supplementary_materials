# DEPRECATED: Do not use
# This was an early version of the model before worldline binding was implemented

import torch
import torch.nn as nn

class OldLoCoV1(nn.Module):
    def __init__(self, num_slots=9, hidden_dim=32):
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
    
    def forward(self, features):
        distances = torch.cdist(features, self.slots.unsqueeze(0))
        attention = torch.softmax(-distances / 0.1, dim=-1)
        return self.slots.unsqueeze(0), attention
