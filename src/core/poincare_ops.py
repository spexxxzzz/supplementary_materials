import torch
import torch.nn as nn

def poincare_distance(x, y):
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    denom = (1 - x_norm_sq) * (1 - y_norm_sq) + 1e-8
    arcosh_arg = 1 + 2 * diff_norm_sq / denom
    return torch.acosh(torch.clamp(arcosh_arg, min=1.0 + 1e-8))

def poincare_exp_map(x, v):
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - x_norm_sq + 1e-8)
    coeff = torch.tanh(lambda_x * v_norm / 2) / v_norm
    return poincare_mobius_add(x, coeff * v)

def poincare_mobius_add(x, y):
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy_dot + y_norm_sq) * x + (1 - x_norm_sq) * y
    denom = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq + 1e-8
    return num / denom

def poincare_log_map(x, y):
    diff = poincare_mobius_add(-x, y)
    diff_norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - x_norm_sq + 1e-8)
    return 2 / lambda_x * torch.atanh(diff_norm) * diff / diff_norm
