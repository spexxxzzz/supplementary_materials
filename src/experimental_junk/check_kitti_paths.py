#!/usr/bin/env python3
import os
import sys

# Quick script to verify KITTI paths exist
# Run this before training to avoid crashes

KITTI_ROOT = '/mnt/raid0_data/neel/kitti_processed_v4/'
REQUIRED_DIRS = ['scans', 'annotations', 'hierarchy_annotations']

print(f"Checking KITTI paths in {KITTI_ROOT}...")

all_good = True
for dir_name in REQUIRED_DIRS:
    path = os.path.join(KITTI_ROOT, dir_name)
    if os.path.exists(path):
        file_count = len([f for f in os.listdir(path) if f.endswith('.npy')])
        print(f"  ✓ {dir_name}: {file_count} files")
    else:
        print(f"  ✗ {dir_name}: NOT FOUND")
        all_good = False

if not all_good:
    print("\nERROR: Some required directories are missing!")
    print("Please ensure data is mounted at:", KITTI_ROOT)
    sys.exit(1)
else:
    print("\nAll paths verified. Ready to train.")
