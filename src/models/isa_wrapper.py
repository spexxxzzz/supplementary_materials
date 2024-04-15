import torch
import torch.nn as nn

class ISAWrapper(nn.Module):
    def __init__(self, hidden_dim=32, num_slots=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, features, num_iterations=3):
        slots = self.slots.unsqueeze(0)
        for _ in range(num_iterations):
            distances = torch.cdist(features, slots)
            attention = torch.softmax(-distances / 0.1, dim=-1)
            updates = torch.bmm(attention.transpose(1, 2), features)
            slots = self.gru(updates.squeeze(0), slots.squeeze(0)).unsqueeze(0)
        return slots, attention
