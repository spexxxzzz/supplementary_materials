#!/usr/bin/env python3
import os

base_config = """model:
  name: LoCo
  num_objects: 3
  num_levels: 3
  hidden_dim: 32
  learnable_times: false

training:
  batch_size: 32
  num_epochs: 250
  optimizer: Adam
  learning_rate: {lr}
  weight_decay: 1e-5
  gradient_clip: 1.0

loss:
  reconstruction_weight: 1.0
  diversity_weight: 0.3
  lambda_cone: 0.5
  cone_penalty: {penalty}

data:
  dataset: KITTI-3DParts
  root_dir: /mnt/raid0_data/neel/kitti_processed_v4/
  train_split: 0.9

logging:
  log_dir: logs_archive/grid_search_lr
  checkpoint_dir: checkpoints/grid_search_lr
"""

def generate_lr_configs():
    lr_values = [
        1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
        8e-4, 3e-4, 1.5e-4, 7e-5, 3e-5, 1.2e-5,
        9e-4, 4e-4, 1.8e-4, 8e-5, 4e-5, 1.5e-5, 6e-6
    ]
    
    os.makedirs('configs/grid_searches/learning_rate_sweep', exist_ok=True)
    
    for i, lr in enumerate(lr_values):
        filename = f"configs/grid_searches/learning_rate_sweep/lr_{lr:.0e}.yaml"
        if 'e-0' in filename:
            filename = filename.replace('e-0', 'e-')
        content = base_config.format(lr=lr, penalty=0.5)
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Generated: {filename}")

def generate_penalty_configs():
    penalty_values = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0
    ]
    
    os.makedirs('configs/grid_searches/cone_penalty_sweep', exist_ok=True)
    
    for i, penalty in enumerate(penalty_values):
        filename = f"configs/grid_searches/cone_penalty_sweep/penalty_{penalty:.1f}.yaml"
        content = base_config.format(lr=1e-4, penalty=penalty)
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Generated: {filename}")

if __name__ == "__main__":
    print("Generating learning rate sweep configs...")
    generate_lr_configs()
    print("\nGenerating cone penalty sweep configs...")
    generate_penalty_configs()
    print("\nDone!")
