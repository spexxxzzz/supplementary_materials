import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class RandomJitter(nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise
