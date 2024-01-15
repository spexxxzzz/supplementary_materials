Supplementary Code for ICML 2026 Submission
"When Do Geometric Priors Help Hierarchical Perception? A Scale and Density-Dependent Analysis with Lorentzian Worldline Attention"

Configured for Cluster-B. Update paths in src/utils/cluster_config.py before running.

Quick Start:
1. Install dependencies: pip install -r requirements_frozen_cluster.txt
2. Update cluster_config.py with your paths
3. Run training: bash scripts/train_dist.sh configs/base_models/kitti_production.yaml

For evaluation: bash scripts/eval_all.sh

Note: KITTI dataset requires /data/shared/kitti_meta_v4.csv (update path in kitti_loader.py)
