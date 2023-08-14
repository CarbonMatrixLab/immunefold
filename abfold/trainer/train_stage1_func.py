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

from abfold.model import AbFold, MetricDict
from abfold.trainer import dataset_stage as dataset
from abfold.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from abfold.trainer.loss import Loss
from abfold.trainer import model_align

def get_device(args):
    if args.device == 'gpu':
        return torch.device(f'cuda:{args.gpu_list[args.local_rank]}')
    elif args.device == 'mlu':
        return ct.mlu_device(args.local_rank)
    else:
        return torch.device('cpu')

def setup_model(model, args):
    device = get_device(args)
    model.to(device)
    model.esm.to(device)
    print('setup model device', device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_list[args.local_rank]], 
        output_device=args.local_rank,)
    model._set_static_graph()
    #find_unused_parameters=True)
    
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
                print('setup dataset device', device)
                feat_args['device'] = device
                feats[i] = (feat_name, feat_args)

    logging.info('AbFold.feats: %s', feats)

    with open(args.train_name_idx) as f:
        name_idx = [i.strip() for i in f]

    train_loader = dataset.load(
            data_dir = args.train_data,
            name_idx = name_idx,
            feats=feats, max_seq_len=args.max_seq_len,
            is_cluster_idx = False,
            rank = args.world_rank, world_size = args.world_size,
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
        model_config = ckpt['model_config']
        model_config['esm2_model_file'] = args.restore_esm2_model
        model = AbFold(config = model_config)
        model.impl.load_state_dict(ckpt['model_state_dict'], strict=True)

        trainable_variables = model_align.setup_model(model, config.align)
        #trainable_variables = model.parameters()
    else:
        model = AbFold(config = config.model)
        trainable_variables = model.parameters()

    logging.info('AbFold.config: %s', config)

    model = setup_model(model, args)

    # optimizer

    print('trained number', len(trainable_variables))
    optim = Optimizer(trainable_variables,
            base_lr=args.learning_rate, warmup_steps=args.warmup_steps, flat_steps=args.flat_steps,
            decay_steps=args.decay_steps, decay_type='linear', min_lr=1e-5)
    
    # loss
    loss_object = Loss(config.loss)
   
    # checkpoint
    def _save_checkpoint(it):
        ckpt_dir = os.path.join(args.prefix, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)
        
        ckpt_file = os.path.join(ckpt_dir, f'epoch_{it}.ckpt')
        
        saved_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        torch.save(dict(
            model_state_dict = saved_model.state_dict(),
            optim_state_dict = optim.state_dict(),
            model_config = config.model,
            train_config = config,
            feature_config = feats,
            args = args,
            train_steps = optim.cur_step), ckpt_file)

    # setup train
    model.train()

    for epoch in range(args.num_epoch):
        running_loss = MetricDict()
        optim.zero_grad()

        for it, batch in enumerate(train_loader):
            jt = (it + 1) % args.gradient_accumulation_it
            
            batch_start_time = time.time()

            print('batch device', batch['seq'].device, batch['esm_seq'].device)

            r = model(batch=batch, compute_loss=True)
            loss_results = loss_object(r, batch)
            loss = loss_results['loss'] / args.gradient_accumulation_it
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            logging.info('traing examples ' + ','.join(batch['name']))
            running_loss += MetricDict({'all': loss_results['loss']})
            for k, v in loss_results.items():
                if k == 'loss':
                    continue
                for kk, vv in v.items():
                    if 'loss' not in kk:
                        continue
                    running_loss += MetricDict({f'{k}@{kk}': vv})
            
            for k, v in r['heads'].items():
                if k not in ['tmscore', 'metric']:
                    continue
                for kk, vv in v.items():
                    if 'loss' in kk:
                        running_loss += MetricDict({f'{k}@{kk}': vv})

            if jt == 0:
                logging.info(f'optim step= {optim.cur_step} lr= {optim.get_values()}')

                optim.step()
                optim.zero_grad()
                
                for k, v in running_loss.items():
                    v = v / args.gradient_accumulation_it
                    log_metric_dict(v, epoch, it, prefix=f'Loss/train@{k}')

                running_loss = MetricDict()
            
            batch_end_time = time.time()
            logging.info(f'{it} batch time {batch_end_time - batch_start_time} s.')

        # Save a checkpoint every epoch
        if args.world_rank == 0 and (epoch + 1) % args.checkpoint_it == 0:
            _save_checkpoint(epoch)
