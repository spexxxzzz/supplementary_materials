import torch
import torch.nn as nn
from src.layers.binding_slots import WorldlineBinding
from src.layers.attention import ScaleAdaptiveAttention
from src.layers.scale_adaptive import AdaptiveHorizon
from src.layers.gru_updates import MultiScaleGRU
from src.layers.projections import FeatureProjection

class LoCo(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False):
        super().__init__()
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        self.binding = WorldlineBinding(num_objects, num_levels, hidden_dim, learnable_times)
        self.attention = ScaleAdaptiveAttention(hidden_dim, num_levels)
        self.horizon = AdaptiveHorizon(num_levels)
        self.gru = MultiScaleGRU(hidden_dim, num_levels)
        self.projection = FeatureProjection(hidden_dim, hidden_dim)
    
    def forward(self, features, rho, num_iterations=3):
        batch_size, num_features, _ = features.shape
        slots = self.binding()
        horizons = self.horizon(rho)
        
        for _ in range(num_iterations):
            attention_weights = self.attention(features, slots, horizons)
            centers = self.binding.object_centers
            updated_centers = self.gru(attention_weights, centers)
            self.binding.object_centers.data = updated_centers
            slots = self.binding()
        
        return slots, attention_weights
