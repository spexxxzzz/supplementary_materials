# Temporary debug script for dataloader issues
# DELETE THIS AFTER FIXING

from src.data.kitti_loader import KITTILoader
import torch

print("Testing KITTI dataloader...")

try:
    dataset = KITTILoader(split='train')
    print(f"Dataset length: {len(dataset)}")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    
    for i, batch in enumerate(loader):
        features, annotation, hierarchy, density = batch
        print(f"Batch {i}: features shape {features.shape}, annotation shape {annotation.shape}")
        if i >= 2:
            break
    
    print("Dataloader works!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
