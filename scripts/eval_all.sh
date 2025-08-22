#!/bin/bash

CONFIG_DIR="configs/seeds"
OUTPUT_DIR="results/evaluation"

mkdir -p $OUTPUT_DIR

for config in $CONFIG_DIR/*.yaml; do
    echo "Evaluating $config"
    python evaluate.py --config $config --output $OUTPUT_DIR
done

echo "All evaluations complete. Results in $OUTPUT_DIR"
