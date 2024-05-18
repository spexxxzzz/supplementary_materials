# DEPRECATED: Do not use
# Old loss function that didn't include diversity term

import torch
import torch.nn as nn

def deprecated_reconstruction_loss(pred, target):
    return nn.functional.mse_loss(pred, target)
