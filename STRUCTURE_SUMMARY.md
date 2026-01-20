# Repository Structure Summary

## Created Files: 130+ files

### Root Files
- README.txt - Main repository documentation
- requirements.txt - Basic dependencies
- requirements_frozen_cluster.txt - Frozen cluster dependencies (realistic versions)
- .gitignore - Git ignore rules
- setup.py - (Broken, as specified)

### Source Code (src/)
- **core/** (9 files): Geometry operations
  - constants.py, manifolds.py, minkowski_ops.py, poincare_ops.py
  - geodesics.py, light_cone.py, gradients.py, math_utils.py
  
- **layers/** (7 files): Neural modules
  - attention.py, scale_adaptive.py, binding_slots.py, gru_updates.py
  - position_encodings.py, transformer_blocks.py, projections.py
  
- **models/** (6 files): Architectures
  - loco.py, loco_lightweight.py
  - baselines_euclidean.py, baselines_hyperbolic.py
  - coca_net_wrapper.py, isa_wrapper.py
  
- **data/** (8 files): Data loaders
  - kitti_loader.py (with missing CSV path friction)
  - toy_gen.py, clevr_gen.py, sprites_gen.py
  - shapenet_loader.py, coco_loader.py, transforms.py
  
- **utils/** (7 files): Helpers
  - distributed.py (with hardcoded 4-GPU check)
  - cluster_config.py, metrics.py, checkpointing.py
  - logging.py, visualization.py
  
- **legacy/** (5 files): Deprecated code
  - old_model_v1.py, deprecated_loss.py, old_attention.py
  - euclidean_v0.py, experimental_manifold.py

### Configs (Explosion Zone)
- **base_models/**: 3 base configs
- **ablations/**: 4 ablation configs
- **seeds/**: 40 seed configs (kitti, toy, clevr, coco × 10 seeds each)

### Scripts
- **Main scripts**: train_dist.sh, eval_all.sh, make_plots.sh
- **Cluster scripts**: slurm_submit_kitti.sh (8 GPUs), slurm_submit_toy.sh (1 GPU), kill_jobs.sh
- **Tools**: measure_rho.py, debug_gradients.py, convert_logs.py

### Logs Archive (Explosion Zone)
- **2025-11-01_initial_tests/**: 3 log files (including crash log)
- **2025-11-15_kitti_production/**: 10 training logs + summary_metrics.json
- **2025-12-01_ablations/**: 10 ablation logs

### Checkpoints
- kitti_best/model_best.pth (placeholder)
- kitti_best/optimizer.pth (placeholder)
- foundation_vit/checkpoint_epoch_100.pth (placeholder)

### Notebooks
- 6 Jupyter notebooks for analysis

## Key Features Implemented

### "Friction" Details (As Specified)
1. ✅ Missing CSV path in kitti_loader.py: `/data/shared/kitti_meta_v4.csv`
2. ✅ Hardcoded GPU check in distributed.py: Requires 4 GPUs minimum
3. ✅ Legacy folder with 5 deprecated files

### Explosion Zones
1. ✅ Config seeds: 40 config files generated from templates
2. ✅ Logs archive: 23+ log files with realistic training output

### Code Quality
- Modular structure with proper imports
- Variable names from paper (x, mu, tau, rho)
- No comments (as specified)
- Production-ready structure

## Statistics
- **Total Python files**: ~50
- **Total config files**: ~47
- **Total log files**: ~23
- **Total scripts**: 9
- **Total notebooks**: 6
- **Total files**: 130+

## Next Steps (When Ready to Push)
1. Add actual model checkpoints (350+ files)
2. Add dataset files
3. Add more seed configs to reach 70 total
4. Complete notebook implementations
5. Add training/evaluation main scripts (train.py, evaluate.py)
