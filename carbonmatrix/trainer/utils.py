import os

import torch.distributed as dist

def get_local_rank():
    return int(os.environ['LOCAL_RANK'])

def get_world_rank():
    return dist.get_rank()

def get_world_size():
    return dist.get_world_size()

def setup_ddp():
    dist.init_process_group(backend='nccl')

def get_device(gpu_list=None):
    if gpu_list is None:
        return get_local_rank()
    else:
        return gpu_list[get_local_rank()]
