import torch
import torch.nn as nn
from .minkowski_ops import minkowski_inner_product, minkowski_norm_squared
from .constants import EPSILON, LAMBDA_P, LAMBDA_S

def future_light_cone(s, x):
    tau = x[..., 0:1] - s[..., 0:1]
    delta = x - s
    inner = minkowski_inner_product(delta, delta)
    time_condition = tau > 0
    causal_condition = inner >= 0
    return time_condition & causal_condition

def cone_score(f, s, h):
    tau = f[..., 0:1] - s[..., 0:1]
    f_spatial = f[..., 1:]
    s_spatial = s[..., 1:]
    r = torch.norm(f_spatial - s_spatial, dim=-1, keepdim=True)
    abs_tau = torch.abs(tau) + EPSILON
    past_penalty = nn.functional.relu(-tau) * LAMBDA_P
    spacelike_penalty = nn.functional.relu(r - torch.abs(tau)) * LAMBDA_S
    score = h - r / abs_tau - past_penalty - spacelike_penalty
    return score

def cone_membership(f, s, h):
    score = cone_score(f, s, h)
    return score > 0

def adaptive_horizon(base_horizon, rho, horizon_scale):
    return base_horizon + horizon_scale * (rho - 0.5)
