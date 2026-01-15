import torch
import torch.nn as nn
from .constants import EPSILON

def minkowski_inner_product(x, y):
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    y0 = y[..., 0:1]
    y_spatial = y[..., 1:]
    return x0 * y0 - torch.sum(x_spatial * y_spatial, dim=-1, keepdim=True)

def minkowski_inner_product_einsum(x, y):
    batch_dims = x.shape[:-1]
    d = x.shape[-1] - 1
    x_reshaped = x.view(-1, d + 1)
    y_reshaped = y.view(-1, d + 1)
    metric = torch.zeros(d + 1, d + 1, device=x.device, dtype=x.dtype)
    metric[0, 0] = 1.0
    metric[1:, 1:] = -torch.eye(d, device=x.device, dtype=x.dtype)
    result = torch.einsum('bi,ij,bj->b', x_reshaped, metric, y_reshaped)
    return result.view(*batch_dims, 1)

def proper_time_distance(x, y):
    delta = x - y
    inner = minkowski_inner_product(delta, delta)
    sign = torch.sign(inner)
    abs_inner = torch.abs(inner)
    sqrt_inner = torch.sqrt(abs_inner + EPSILON)
    return sign * sqrt_inner

def proper_time_distance_safe(x, y):
    delta = x - y
    inner = minkowski_inner_product(delta, delta)
    sign = torch.sign(inner + EPSILON)
    abs_inner = torch.abs(inner)
    sqrt_inner = torch.sqrt(torch.clamp(abs_inner, min=EPSILON))
    result = sign * sqrt_inner
    mask_timelike = inner > EPSILON
    mask_spacelike = inner < -EPSILON
    result = torch.where(mask_timelike, sqrt_inner, result)
    result = torch.where(mask_spacelike, -sqrt_inner, result)
    return result

def minkowski_norm_squared(x):
    return minkowski_inner_product(x, x)

def is_timelike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return inner > EPSILON

def is_spacelike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return inner < -EPSILON

def is_lightlike(x, y):
    inner = minkowski_inner_product(x - y, x - y)
    return torch.abs(inner) < EPSILON

def minkowski_exp_map(x, v):
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    v0 = v[..., 0:1]
    v_spatial = v[..., 1:]
    inner_v = minkowski_inner_product(v, v)
    sqrt_inner = torch.sqrt(torch.clamp(torch.abs(inner_v), min=EPSILON))
    if torch.all(inner_v > EPSILON):
        cosh_term = torch.cosh(sqrt_inner)
        sinh_term = torch.sinh(sqrt_inner)
        new_x0 = x0 * cosh_term + v0 * sinh_term / sqrt_inner
        new_x_spatial = x_spatial * cosh_term + v_spatial * sinh_term / sqrt_inner
    else:
        new_x0 = x0 + v0
        new_x_spatial = x_spatial + v_spatial
    return torch.cat([new_x0, new_x_spatial], dim=-1)

def minkowski_log_map(x, y):
    delta = y - x
    inner_delta = minkowski_inner_product(delta, delta)
    if torch.all(inner_delta > EPSILON):
        sqrt_delta = torch.sqrt(inner_delta)
        arcosh_arg = minkowski_inner_product(x, y)
        arcosh_val = torch.acosh(torch.clamp(arcosh_arg, min=1.0 + EPSILON))
        scale = arcosh_val / sqrt_delta
        return delta * scale
    else:
        return delta
