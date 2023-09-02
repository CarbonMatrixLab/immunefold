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
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import Adam

from carbon.model import CarbonFold, MetricDict
from carbon.trainer import dataset_stage as dataset
from carbon.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from carbon.trainer.loss import Loss
from carbon.trainer import model_align

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
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_list[args.local_rank]], 
        output_device=args.gpu_list[args.local_rank],)
    #find_unused_parameters=True)
        #static_graph=False)
    #model._set_static_graph()
    
    logging.info('wrap model with nn.parallel.DistributedDataParallel class')
    return model, device

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

    logging.info('CarbonFold.feats: %s', feats)

    with open(args.train_name_idx) as f:
        name_idx = [i.strip() for i in f]
    
    real_batch_size = args.batch_size * args.world_size * args.gradient_accumulation_it
    reduced_num = len(name_idx) - len(name_idx) % real_batch_size
    name_idx = name_idx[:reduced_num]

    logging.info(f'traing sampels {len(name_idx)}')

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
        model = CarbonFold(config = model_config)
        model.impl.load_state_dict(ckpt['model_state_dict'], strict=True)

        trainable_variables = model_align.setup_model(model, config.align)
        #trainable_variables = model.parameters()
    else:
        model = CarbonFold(config = config.model)
        trainable_variables = model.parameters()

    #for n, p in model.named_parameters():
    #    print(n, p.requires_grad)

    logging.info('CarbonFold.config: %s', config)

    model, device = setup_model(model, args)

    # optimizer
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
        
        ckpt_file = os.path.join(ckpt_dir, f'step_{it}.ckpt')
        
        saved_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        torch.save(dict(
            model_state_dict = saved_model.esm.state_dict(),
            #optim_state_dict = optim.state_dict(),
            model_config = config.model,
            train_config = config,
            feature_config = feats,
            args = args,
            train_steps = optim.cur_step), ckpt_file)

    # setup train
    model.train()
    
    running_loss = MetricDict()
    optim.zero_grad()
    batch_start_time = time.time()


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epoch):
        running_loss = MetricDict()
        optim.zero_grad()

        for it, batch in enumerate(train_loader):
            jt = (it + 1) % args.gradient_accumulation_it

            ctx = nullcontext if jt == 0 else model.no_sync
            with ctx():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    r = model(batch=batch, compute_loss=True)
                    loss_results = loss_object(r, batch)
                    loss = loss_results['loss'] / args.gradient_accumulation_it
                
                scaler.scale(loss).backward()
                #loss.backward()
            
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

                #optim.step()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                
                for k, v in running_loss.items():
                    v = v / args.gradient_accumulation_it
                    log_metric_dict(v, epoch, it, prefix=f'Loss/train@{k}')

                running_loss = MetricDict()
            
                batch_end_time = time.time()
                logging.info(f'{it} batch time {batch_end_time - batch_start_time} s.')
                batch_start_time = time.time()

                if optim.cur_step % args.checkpoint_it == 0:
                    _save_checkpoint(optim.cur_step)
