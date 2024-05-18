import os

CLUSTER_NAME = "Cluster-B"
DATA_ROOT = "/data/shared"
CHECKPOINT_ROOT = "/checkpoints/shared"
LOG_ROOT = "/logs/shared"

KITTI_META_PATH = "/data/shared/kitti_meta_v4.csv"
KITTI_SCANS_PATH = "/data/shared/kitti_scans"
KITTI_ANNOTATIONS_PATH = "/data/shared/kitti_annotations"

SHAPENET_ROOT = "/data/shared/shapenet"
COCO_ROOT = "/data/shared/coco"
PARTIMAGENET_ROOT = "/data/shared/partimagenet"

SLURM_PARTITION = "gpu"
SLURM_ACCOUNT = "research"
SLURM_TIME = "48:00:00"

def update_paths():
    global DATA_ROOT, CHECKPOINT_ROOT, LOG_ROOT
    if 'DATA_ROOT' in os.environ:
        DATA_ROOT = os.environ['DATA_ROOT']
    if 'CHECKPOINT_ROOT' in os.environ:
        CHECKPOINT_ROOT = os.environ['CHECKPOINT_ROOT']
    if 'LOG_ROOT' in os.environ:
        LOG_ROOT = os.environ['LOG_ROOT']
