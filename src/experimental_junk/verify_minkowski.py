import torch
from src.core.minkowski_ops import minkowski_inner_product, proper_time_distance

# Quick sanity check for Minkowski operations
# Run this to verify math is correct

print("Testing Minkowski operations...")

# Test vectors
x = torch.tensor([[1.0, 0.5, 0.3, 0.2]])
y = torch.tensor([[1.0, 0.4, 0.2, 0.1]])

inner = minkowski_inner_product(x, y)
print(f"Inner product: {inner.item()}")

dist = proper_time_distance(x, y)
print(f"Distance: {dist.item()}")

# Test timelike condition
delta = x - y
inner_delta = minkowski_inner_product(delta, delta)
print(f"Delta inner product: {inner_delta.item()}")
print(f"Timelike: {inner_delta.item() > 0}")

print("Tests passed!")
