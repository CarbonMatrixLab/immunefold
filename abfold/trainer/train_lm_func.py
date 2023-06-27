import os
import functools
import json
import logging
import random
from collections import OrderedDict
import time

import numpy as np
import pandas as pd

import ml_collections
import torch
from torch import nn
from torch.optim import Adam

#from esm.pretrained import load_model_and_alphabet_local
from abfold.model.lm.pretrained import load_model_and_alphabet_local

from abfold.model import MetricDict
from abfold.trainer import dataset_lm as dataset
from abfold.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from abfold.trainer.loss_lm import loss_func

def get_device(args):
    if args.device == 'gpu':
        return torch.device(f'cuda:{args.local_rank}')
    elif args.device == 'mlu':
        return ct.mlu_device(args.local_rank)
    else:
        return torch.device('cpu')

def setup_model(model, args):
    device = get_device(args)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank], 
        output_device=args.local_rank,
        find_unused_parameters=True)
    #model._set_static_graph()
    
    logging.info('wrap model with nn.parallel.DistributedDataParallel class')
    return model
    
def setup_dataset(args):
    # feature
    device = get_device(args)
    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.loads(f.read())
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
                feats[i] = (feat_name, feat_args)
    
    logging.info('AbFold.feats: %s', feats)

    train_loader = dataset.load(
            fasta_file = args.train_data,
            feats=feats, max_seq_len=args.max_seq_len,
            is_cluster_idx = True,
            rank = args.world_rank, world_size = args.world_size,
            max_steps = args.decay_steps,
            batch_size = args.batch_size)

    logging.info(f'world_rank={args.world_rank} data_world_size={args.world_size}')

    eval_loader = None
    if eval_loader is not None:
        logging.info(f'eval_data_file= {eval_data_file} eval_name_idx_file= {eval_name_idx_file}')
    
    return feats, train_loader, eval_loader


def log_metric_dict(loss, epoch, it= '', prefix=''):
    if isinstance(loss, MetricDict):
        if prefix:
            prefix = f'{prefix}.'
        for k, v in loss.items():
            log_metric_dict(v, epoch, it, prefix=f'{prefix}{k}')
    elif isinstance(loss, torch.Tensor):
        loss = loss.item()
        logging.info(f'{epoch} {it} {prefix}: {loss}')

def train(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # dataset
    feats, train_loader, eval_loader = setup_dataset(args)

    # model
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
        config = ml_collections.ConfigDict(config)

    if args.restore_model_ckpt is not None:
        ckpt = torch.load(args.restore_model_ckpt)
        model, _, esm_cfg = load_model_and_alphabet_local(args.restore_model_ckpt)

        model.embed_tokens.requires_grad = False

        for k in range(config.finetuning.num_frozen_layers):
            for p in model.layers[k].parameters():
                p.requires_grad = False

        trainable_variables = [p for p in model.parameters() if p.requires_grad]
    
    model = setup_model(model, args)

    # optimizer
    optim = Optimizer(trainable_variables, args.learning_rate, decay_type='linear', decay_steps=args.decay_steps, min_lr=1e-5,
            betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    
    # checkpoint
    def _save_checkpoint(it):
        ckpt_dir = os.path.join(args.prefix, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)
        
        ckpt_file = os.path.join(ckpt_dir, f'step_{it}.ckpt')
        
        saved_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        torch.save(dict(
            model = saved_model.state_dict(),
            cfg = esm_cfg,
            feature_config = feats,
            args = args,
            train_steps = optim.cur_step), ckpt_file)

    # setup train
    model.train()

    running_loss = MetricDict()
    optim.zero_grad()
    batch_start_time = time.time()

    for it, batch in enumerate(train_loader):
        jt = (it + 1) % args.gradient_accumulation_it
        
        if jt == 0:
            r = model(tokens = batch['seq'])

            loss = loss_func(batch, r)
            loss = loss / args.gradient_accumulation_it
 
            loss.backward()
        else:
            with model.no_sync():
                r = model(tokens = batch['seq'])
 
                loss = loss_func(batch, r)
                loss = loss / args.gradient_accumulation_it

                loss.backward()

        running_loss += MetricDict({'loss': loss})

        if jt == 0:
            logging.info(f'optim step= {optim.cur_step} lr= {optim.get_values()}')

            optim.step()
            optim.zero_grad()
            
            for k, v in running_loss.items():
                #v = v / args.gradient_accumulation_it
                log_metric_dict(v, 0, optim.cur_step, prefix=f'Loss/train@{k}')

            running_loss = MetricDict()
        
            batch_end_time = time.time()
            logging.info(f'{optim.cur_step} batch time {batch_end_time - batch_start_time} s.')
            batch_start_time = time.time()

            if optim.cur_step % args.checkpoint_it == 0:
                    _save_checkpoint(optim.cur_step)
