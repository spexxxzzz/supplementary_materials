# DEPRECATED: Do not use
# Experimental manifold implementation that was not used in final paper

import torch
import torch.nn as nn

class ExperimentalManifold(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def distance(self, x, y):
        return torch.norm(x - y, dim=-1)
