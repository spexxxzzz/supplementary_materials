# Supplementary Material: Lorentzian Worldline Attention (LoCo)

This repository contains the implementation, experiment configurations, and raw logs for the paper "When Do Geometric Priors Help Hierarchical Perception?".

## Directory Structure
- /src: Core implementation of LoCo and baselines (Euclidean, Hyperbolic).
  - Note: /src/models/loco.py is the clean implementation. /src/models/loco_final_maybe.py contains the experimental binding logic used for Table 3 results.
- /configs: Hydra configuration files.
  - /configs/seeds: Exact configs for reproducing the 70 main results.
  - /configs/grid_searches: Hyperparameter sweeps for rho thresholds.
- /logs_archive: Raw stdout captures from the training cluster (Node-4/Node-8).

## Usage
Dependencies are listed in `requirements_frozen_cluster.txt`.
NOTE: This snapshot captures the environment on the university cluster (CUDA 11.8).

To verify the Euclidean Collapse (Figure 2), run:
python src/experimental_junk/verify_minkowski.py

To run training, you must update the data paths in `src/utils/cluster_config.py`.
Current paths point to: /mnt/ssd_raid_0/neel_data/

## Data Preparation
Due to size constraints (540GB+), the full preprocessed KITTI-3DParts and ShapeNet-Level3 datasets are hosted on our internal cluster. 

We have provided a `download_sample_data.sh` script to fetch a 1GB mini-batch for verification.
*Note: The external download link has been disabled for anonymity. Please refer to `data/samples/` for structure verification.*

## Pretrained Models
Checkpoints in /checkpoints are stripped of optimizer states to save space.
Full checkpoints (312M params) are available upon request.
