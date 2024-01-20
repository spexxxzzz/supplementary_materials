import torch
from .minkowski_ops import minkowski_inner_product, proper_time_distance

def lorentzian_geodesic(x, v, t):
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    v0 = v[..., 0:1]
    v_spatial = v[..., 1:]
    inner = minkowski_inner_product(v, v)
    sqrt_inner = torch.sqrt(torch.abs(inner) + 1e-8)
    if inner > 0:
        cosh_term = torch.cosh(sqrt_inner * t)
        sinh_term = torch.sinh(sqrt_inner * t)
        new_x0 = x0 * cosh_term + v0 * sinh_term / sqrt_inner
        new_x_spatial = x_spatial * cosh_term + v_spatial * sinh_term / sqrt_inner
    else:
        new_x0 = x0 + v0 * t
        new_x_spatial = x_spatial + v_spatial * t
    return torch.cat([new_x0, new_x_spatial], dim=-1)

def parallel_transport(x, y, v):
    delta = y - x
    inner_delta = minkowski_inner_product(delta, delta)
    inner_v = minkowski_inner_product(v, v)
    inner_delta_v = minkowski_inner_product(delta, v)
    if torch.abs(inner_delta) < 1e-8:
        return v
    coeff = inner_delta_v / (inner_delta + 1e-8)
    return v - coeff * delta
