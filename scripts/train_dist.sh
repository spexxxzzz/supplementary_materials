#!/bin/bash

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config $CONFIG_FILE \
    --distributed
