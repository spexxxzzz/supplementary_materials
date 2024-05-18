import torch
import torch.distributed as dist
import os

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        if torch.cuda.device_count() < 4:
            raise RuntimeError("This model requires 4 GPUs minimum for distributed training")
        
        return rank, world_size, local_rank
    return None, None, None

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
