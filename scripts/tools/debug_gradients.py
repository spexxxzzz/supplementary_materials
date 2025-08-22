#!/usr/bin/env python3

import torch
import argparse
from src.models.loco import LoCo
from src.core.gradients import check_gradient_stability

def debug_gradients(model, loss):
    model.zero_grad()
    loss.backward()
    
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            is_stable = check_gradient_stability(param_norm)
            print(f"{name}: norm={param_norm:.4f}, stable={is_stable}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()
    
    model = LoCo()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    
    dummy_input = torch.randn(1, 60, 32)
    dummy_rho = torch.tensor([[0.5]])
    output = model(dummy_input, dummy_rho)
    dummy_loss = output[0].sum()
    
    debug_gradients(model, dummy_loss)
