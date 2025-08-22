#!/bin/bash
#SBATCH --job-name=loco_toy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_archive/toy_lightweight/slurm_%j.out
#SBATCH --error=logs_archive/toy_lightweight/slurm_%j.err

module load cuda/11.8
module load python/3.9

source venv/bin/activate

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/base_models/toy_lightweight.yaml"
fi

python train.py --config $CONFIG_FILE
