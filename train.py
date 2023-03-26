import os
import argparse
import logging
import resource

import torch

from abfold.trainer.train_func import train

class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f'Rank {self._rank} | {record.msg}'
        return True

def setup(args):
    if args.device == 'mlu':
        import torch_mlu
        import torch_mlu.core.mlu_model as ct
        torch.distributed.init_process_group(backend='cncl')
        args.world_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        ct.set_device(args.local_rank)
    elif args.device == 'gpu':
        torch.distributed.init_process_group(backend='nccl')


    os.makedirs(os.path.abspath(args.prefix), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(args.prefix, 'checkpoints')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(args.prefix, 'logs')), exist_ok=True)
    

    log_file = os.path.abspath(os.path.join(args.prefix, 'logs', f'rank{args.world_rank}.log'))

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
    
    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt))
        h.addFilter(WorkerLogFilter(args.world_rank))
        return h

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file)]

    handlers = list(map(_handler_apply, handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    logging.info('-----------------')
    logging.info(f'Arguments: {args}')
    logging.info('-----------------')

    logging.info(f'torch.distributed.init_process_group: local_rank={args.local_rank}, world_rank={args.world_rank}, world_size={args.world_size}')

def cleanup(args):
    torch.distributed.destroy_process_group()

def main(args):
    setup(args)
    train(args)
    cleanup(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mlu'], default='gpu')
    parser.add_argument('--prefix', type=str, default='.')
    
    parser.add_argument('--train_name_idx', type=str)
    parser.add_argument('--train_data', type=str)
    
    parser.add_argument('--general_data_gpu_ratio', type=float, default=0.0)
    
    parser.add_argument('--train_general_name_idx', type=str)
    parser.add_argument('--train_general_data', type=str)

    parser.add_argument('--eval_name_idx', type=str)
    parser.add_argument('--eval_data', type=str)
    parser.add_argument('--eval_general_name_idx', type=str)
    parser.add_argument('--eval_general_data', type=str)
    
    parser.add_argument('--num_epoch', type=int, default=1024)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--checkpoint_max_to_keep', type=int, default=5)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--num_gradient_accumulate_step', type=int, default=1)
    
    parser.add_argument('--max_seq_len', type=int, default=None)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_decay', type=str, choices=[None, 'half', 'poly'], default=None)
    parser.add_argument('--learning_rate', type=float, default='1e-3')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--flat_steps', type=int, default=0)
    
    parser.add_argument('--model_features', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)

    parser.add_argument("--local_rank", type=int, default=0)
    #parser.add_argument("--local_world_size", type=int, default=1)
    
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')


    args = parser.parse_args()
    
    # distributed training
    args.world_rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    #args.local_rank = int(os.environ['LOCAL_RANK'])
    print('world size', args.world_size, args.local_world_size)

    main(args)
