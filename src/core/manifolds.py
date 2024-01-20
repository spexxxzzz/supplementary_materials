import torch
import torch.nn as nn
from .minkowski_ops import minkowski_inner_product, proper_time_distance
from .poincare_ops import poincare_distance

class LorentzianManifold(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def distance(self, x, y):
        return proper_time_distance(x, y)
    
    def inner_product(self, x, y):
        return minkowski_inner_product(x, y)
    
    def exp_map(self, x, v):
        return x + v
    
    def log_map(self, x, y):
        return y - x

class HyperbolicManifold(nn.Module):
    def __init__(self, dim, curvature=-1.0):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
    
    def distance(self, x, y):
        return poincare_distance(x, y) * torch.sqrt(torch.abs(self.curvature))
    
    def inner_product(self, x, y):
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        return 4 * xy_dot / ((1 - x_norm_sq) * (1 - y_norm_sq) + 1e-8)
