import os
import argparse
import logging
import resource
import hydra
from omegaconf import DictConfig

import torch

from immunefold.trainer.train_func import train
from immunefold.trainer.train_func_bayesmvp import train as train_bayesmvp
from immunefold.trainer import utils

class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f'Rank {self._rank} | {record.msg}'
        return True

def setup(cfg):
    utils.setup_ddp()
    world_rank = utils.get_world_rank()
    local_rank = utils.get_local_rank()

    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'checkpoints')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'logs')), exist_ok=True)

    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'logs', f'rank{world_rank}.log'))

    level = logging.DEBUG if cfg.verbose else logging.INFO
    # fmt = f'%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) Rank {world_rank} | %(message)s'
    fmt = f'%(asctime)-15s [%(levelname)s] Rank {world_rank} | %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        #h.addFilter(WorkerLogFilter(world_rank),)
        return h

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file),
        ]

    handlers = list(map(_handler_apply, handlers))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    logging.info('-----------------')
    logging.info(f'Arguments: {cfg}')
    logging.info('-----------------')

    logging.info(f'torch.distributed.init_process_group: world_rank={world_rank}, local_rank={local_rank}, world_size=')

def cleanup(cfg):
    torch.distributed.destroy_process_group()

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg : DictConfig):
    setup(cfg)

    mode = cfg.get('mode', 'carbonfold')
    if mode == 'carbonfold':
        train(cfg)
    elif mode == 'bayesmvp':
        train_bayesmvp(cfg)
    else:
        raise NotImplementedError(f'train mode{mode} not implemented')
    
    cleanup(cfg)

if __name__ == '__main__':

    main()
