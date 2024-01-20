import torch
import torch.nn as nn
from .constants import EPSILON

def smooth_relu(x, beta=1.0):
    return torch.log(1 + torch.exp(beta * x)) / beta

def smooth_abs(x, beta=1.0):
    return torch.sqrt(x * x + beta * beta)

def gradient_clip_norm(parameters, max_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-8)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

def check_gradient_stability(grad_norm, threshold=10.0):
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        return False
    if grad_norm > threshold:
        return False
    return True
