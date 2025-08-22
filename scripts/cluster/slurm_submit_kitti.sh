#!/bin/bash
#SBATCH --job-name=loco_kitti
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs_archive/kitti_production/slurm_%j.out
#SBATCH --error=logs_archive/kitti_production/slurm_%j.err

module load cuda/11.8
module load python/3.9

source venv/bin/activate

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/base_models/kitti_production.yaml"
fi

python train.py --config $CONFIG_FILE --distributed
