#!/usr/bin/env python3
import random
import datetime

crash_types = [
    ("CUDA out of memory", "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 32.00 GiB."),
    ("NaN loss", "RuntimeError: Function 'MseLossBackward' returned nan values in its 0th output."),
    ("Empty tensor", "RuntimeError: max(): Expected reduction dim to be >= 0 and < 1, but got dim=1"),
    ("Index out of bounds", "IndexError: index 1842 is out of bounds for dimension 0 with size 1842"),
    ("Gradient explosion", "RuntimeError: Function 'AddmmBackward' returned an infinity value."),
    ("File not found", "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/raid0_data/neel/kitti_processed_v4/scans/000000.npy'"),
    ("Division by zero", "ZeroDivisionError: float division by zero"),
    ("Shape mismatch", "RuntimeError: shape '[32, 3, 32]' is invalid for input of size 3072"),
]

def generate_crash_log(run_id, crash_type_idx=None):
    if crash_type_idx is None:
        crash_type_idx = run_id % len(crash_types)
    
    crash_name, crash_error = crash_types[crash_type_idx]
    
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append(f"Training Run {run_id:02d} - {datetime.datetime.now()}")
    log_lines.append("=" * 80)
    log_lines.append("[INFO] Cluster Node: g8-node-04 | GPU: Tesla V100-SXM2-32GB")
    log_lines.append("[INFO] Model: LoCo | Dataset: KITTI-3DParts")
    log_lines.append("[INFO] Config: configs/grid_searches/learning_rate_sweep/lr_5e-4.yaml")
    log_lines.append("")
    log_lines.append("Initializing model...")
    log_lines.append("Loading dataset...")
    log_lines.append("Starting training...")
    log_lines.append("")
    
    initial_loss = 0.95 + random.uniform(-0.1, 0.1)
    
    for epoch in range(1, 4):
        if epoch == 1:
            loss = initial_loss
        elif epoch == 2:
            if "NaN" in crash_name:
                loss = float('nan')
            elif "explosion" in crash_name.lower():
                loss = 1e10
            else:
                loss = initial_loss * 0.9 + random.uniform(-0.05, 0.05)
        else:
            loss = initial_loss * 0.85
        
        epoch_time = random.uniform(120, 180)
        log_lines.append(f"Epoch {epoch:3d}/250 | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")
        
        if epoch == 2:
            if "NaN" in crash_name or "explosion" in crash_name.lower():
                log_lines.append("")
                log_lines.append("WARNING: Loss is NaN or Inf. Checking gradients...")
                log_lines.append("WARNING: Gradient norm: inf")
                log_lines.append("")
    
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("ERROR: Training crashed!")
    log_lines.append("=" * 80)
    log_lines.append("")
    log_lines.append("Traceback (most recent call last):")
    log_lines.append('  File "train.py", line 245, in <module>')
    log_lines.append('    loss.backward()')
    log_lines.append('  File "/opt/conda/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward')
    log_lines.append('    Variable._execution_engine.run_backward(')
    log_lines.append('  File "train.py", line 189, in forward')
    log_lines.append('    output = self.model(features, rho)')
    log_lines.append('  File "src/models/loco.py", line 237, in forward')
    log_lines.append('    attention_weights = self.attention(features, slots, horizons)')
    log_lines.append(f"RuntimeError: {crash_error}")
    log_lines.append("")
    log_lines.append("Training terminated unexpectedly.")
    log_lines.append(f"Total runtime: {random.uniform(0.1, 0.5):.2f} hours")
    
    return "\n".join(log_lines)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: generate_crash_logs.py <run_id> <output_file> [crash_type_idx]")
        sys.exit(1)
    
    run_id = int(sys.argv[1])
    output_file = sys.argv[2]
    crash_type_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    log_content = generate_crash_log(run_id, crash_type_idx)
    
    with open(output_file, 'w') as f:
        f.write(log_content)
    
    print(f"Generated crash log: {output_file}")
