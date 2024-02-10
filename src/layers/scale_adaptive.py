import torch
import torch.nn as nn
from src.core.constants import BASE_HORIZONS, HORIZON_SCALE
from src.core.light_cone import adaptive_horizon

class AdaptiveHorizon(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.base_horizons = BASE_HORIZONS
        self.horizon_scale = HORIZON_SCALE
    
    def forward(self, rho):
        batch_size = rho.shape[0]
        horizons = []
        for i in range(self.num_levels):
            base_h = self.base_horizons[i]
            h = adaptive_horizon(base_h, rho, self.horizon_scale)
            horizons.append(h)
        return torch.stack(horizons, dim=1)
