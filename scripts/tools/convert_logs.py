#!/usr/bin/env python3

import argparse
import json
import re

def parse_log_file(log_path):
    results = {
        'epochs': [],
        'losses': [],
        'accuracies': []
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss' in line:
                epoch_match = re.search(r'Epoch (\d+)', line)
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                acc_match = re.search(r'Acc: ([\d.]+)', line)
                
                if epoch_match:
                    results['epochs'].append(int(epoch_match.group(1)))
                if loss_match:
                    results['losses'].append(float(loss_match.group(1)))
                if acc_match:
                    results['accuracies'].append(float(acc_match.group(1)))
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    results = parse_log_file(args.log_file)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Converted {args.log_file} to {args.output}")
