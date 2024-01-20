import torch
import torch.nn as nn
from .constants import EPSILON

def minkowski_inner_product(x, y):
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    y0 = y[..., 0:1]
    y_spatial = y[..., 1:]
    return x0 * y0 - torch.sum(x_spatial * y_spatial, dim=-1, keepdim=True)

def proper_time_distance(x, y):
    delta = x - y
    inner = minkowski_inner_product(delta, delta)
    sign = torch.sign(inner)
    abs_inner = torch.abs(inner)
    sqrt_inner = torch.sqrt(abs_inner + EPSILON)
    return sign * sqrt_inner

def minkowski_norm_squared(x):
    return minkowski_inner_product(x, x)

def is_timelike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return inner > 0

def is_spacelike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return inner < 0

def is_lightlike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return torch.abs(inner) < EPSILON
