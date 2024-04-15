import torch
import torch.nn as nn

class COCANetWrapper(nn.Module):
    def __init__(self, hidden_dim=32, num_slots=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
    
    def forward(self, features):
        slots = self.slots.unsqueeze(0)
        attn_out, attention = self.attention(slots, features, features)
        return attn_out, attention
