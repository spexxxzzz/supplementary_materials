import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.minkowski_ops import proper_time_distance
from src.core.light_cone import cone_score
from src.core.constants import LAMBDA_CONE, TAU_TEMP

class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, hidden_dim, num_levels=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.lambda_cone = LAMBDA_CONE
        self.tau_temp = TAU_TEMP
    
    def forward(self, features, slots, horizons):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        
        distances = []
        cone_scores = []
        
        for i in range(num_slots):
            slot = slots[:, i:i+1, :]
            horizon = horizons[:, i:i+1]
            dist = proper_time_distance(features, slot)
            cone = cone_score(features, slot, horizon)
            distances.append(dist)
            cone_scores.append(cone)
        
        distances = torch.stack(distances, dim=1)
        cone_scores = torch.stack(cone_scores, dim=1)
        
        abs_distances = torch.abs(distances)
        cone_contrib = torch.tanh(cone_scores) * self.lambda_cone
        
        logits = (-abs_distances + cone_contrib) / self.tau_temp
        attention = F.softmax(logits, dim=1)
        
        return attention
