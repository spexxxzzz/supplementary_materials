# DEPRECATED: Do not use
# Old attention mechanism without adaptive horizons

import torch
import torch.nn.functional as F

def old_attention(features, slots, temperature=0.1):
    distances = torch.cdist(features, slots)
    return F.softmax(-distances / temperature, dim=-1)
