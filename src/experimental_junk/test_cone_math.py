import torch
from src.core.light_cone import cone_score, future_light_cone

# Quick test to verify light cone math
# This was giving weird results earlier

print("Testing light cone calculations...")

# Test feature and slot
f = torch.tensor([[2.0, 1.0, 0.5, 0.3]])
s = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
h = torch.tensor([[0.5]])

score = cone_score(f, s, h)
print(f"Cone score: {score.item()}")

in_cone = future_light_cone(s, f)
print(f"In future light cone: {in_cone.item()}")

# Test edge case
f_edge = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
score_edge = cone_score(f_edge, s, h)
print(f"Edge case score: {score_edge.item()}")

print("Done testing")
