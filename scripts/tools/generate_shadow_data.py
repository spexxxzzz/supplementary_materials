#!/usr/bin/env python3
import random
import os

def generate_kitti_metadata():
    os.makedirs('data/kitti_3dparts/metadata', exist_ok=True)
    
    lines = ['scan_id,frame,density_score']
    for i in range(1, 1001):
        scan_id = f"{i:06d}"
        frame = random.randint(0, 100)
        density_score = round(random.uniform(0.2, 0.9), 2)
        lines.append(f"{scan_id},{frame},{density_score}")
    
    with open('data/kitti_3dparts/metadata/train_split_v4.csv', 'w') as f:
        f.write('\n'.join(lines))
    
    print("Generated data/kitti_3dparts/metadata/train_split_v4.csv")

def generate_shapenet_mapping():
    import json
    
    os.makedirs('data/shapenet_parts/metadata', exist_ok=True)
    
    categories = [
        "airplane", "bag", "basket", "bathtub", "bed", "bench", "birdhouse",
        "bookshelf", "bottle", "bowl", "bus", "cabinet", "camera", "can",
        "cap", "car"
    ]
    
    mapping = {str(i): categories[i] for i in range(16)}
    
    with open('data/shapenet_parts/metadata/category_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("Generated data/shapenet_parts/metadata/category_mapping.json")

def generate_binary_files():
    os.makedirs('data/samples', exist_ok=True)
    
    # Generate 1MB file
    with open('data/samples/kitti_seq_0001.bin', 'wb') as f:
        f.write(os.urandom(1024 * 1024))
    print("Generated data/samples/kitti_seq_0001.bin (1MB)")
    
    # Generate 10KB file
    with open('data/samples/metadata_cache_DO_NOT_DELETE.pkl', 'wb') as f:
        f.write(os.urandom(10 * 1024))
    print("Generated data/samples/metadata_cache_DO_NOT_DELETE.pkl (10KB)")

if __name__ == "__main__":
    print("Generating shadow data structure...")
    generate_kitti_metadata()
    generate_shapenet_mapping()
    generate_binary_files()
    print("Done!")
