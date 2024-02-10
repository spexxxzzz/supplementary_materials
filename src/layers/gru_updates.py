import torch
import torch.nn as nn

class MultiScaleGRU(nn.Module):
    def __init__(self, hidden_dim=32, num_levels=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, attention_weights, current_centers):
        batch_size, num_slots, num_features = attention_weights.shape
        num_objects = num_slots // self.num_levels
        
        aggregated_features = torch.bmm(attention_weights, current_centers.unsqueeze(0).expand(batch_size, -1, -1))
        
        updates = []
        for i in range(num_objects):
            object_updates = []
            for j in range(self.num_levels):
                slot_idx = i * self.num_levels + j
                attn = attention_weights[:, slot_idx:slot_idx+1, :]
                feat = aggregated_features[:, slot_idx:slot_idx+1, :]
                update = self.gru(feat.squeeze(1), current_centers[i])
                object_updates.append(update)
            combined_update = torch.stack(object_updates, dim=0).mean(dim=0)
            updates.append(combined_update)
        
        return torch.stack(updates, dim=0)
