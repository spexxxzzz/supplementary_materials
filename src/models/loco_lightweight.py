import torch
import torch.nn as nn
from src.models.loco import LoCo
from src.layers.projections import FeatureProjection

class LoCoLightweight(LoCo):
    def __init__(self, input_dim=2, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False):
        super().__init__(num_objects, num_levels, hidden_dim, learnable_times)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, num_iterations=3):
        features = self.encoder(x)
        rho = self.scale_predictor(features.mean(dim=1))
        return super().forward(features, rho, num_iterations)
