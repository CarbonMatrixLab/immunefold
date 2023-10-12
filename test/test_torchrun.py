import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist

@hydra.main(version_base=None, config_path="config", config_name="model/carbonfold")
def main(cfg : DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    print(cfg)
    print('local rank', local_rank)
    
    dist.init_process_group(backend='cncl')
    world_size = dist.get_world_size()
    print('world_size', world_size)

if __name__ == '__main__':
    main()
