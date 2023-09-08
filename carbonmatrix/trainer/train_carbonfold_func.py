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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from carbonmatrix.model import CarbonFold, MetricDict
from carbonmatrix.trainer import dataset_carbonfold as dataset
from carbonmatrix.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from carbonmatrix.trainer.loss_factory import LossFactory
from carbonmatrix.trainer import model_align
from carbonmatrix.trainer import utils


def setup_model(model, device):
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device], 
        output_device=cfg.gpu_list[l],
        find_unused_parameters=True,
        )
    
    # model._set_static_graph()

    logging.info('wrap model with nn.parallel.DistributedDataParallel class')
    
    return model

def setup_dataset(cfg):
    device = utils.get_device(cfg)
    
    logging.info('CarbonFold.feats: %s', feats)

    with open(cfg.train_name_idx) as f:
        name_idx = [i.strip() for i in f]

    real_batch_size = cfg.batch_size * utils.get_world_size() * cfg.gradient_accumulation_it
    reduced_num = len(name_idx) - len(name_idx) % real_batch_size

    name_idx = name_idx[:reduced_num]
    
    dataset = StructureDataset(cfg.train_data, name_idx, device=device, feats=cfg.features)
    sampler = DistributedSampler(dataset, drop_last=True)
    train_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            drop_last=True)
    
    logging.info(f'traing sampels {len(name_idx)}')

    return train_loader

def log_metric_dict(loss, epoch, it= '', prefix=''):
    if isinstance(loss, MetricDict):
        if prefix:
            prefix = f'{prefix}.'
        for k, v in loss.items():
            log_metric_dict(v, epoch, it, prefix=f'{prefix}{k}')
    elif isinstance(loss, torch.Tensor):
        loss = loss.item()
        logging.info(f'{epoch} {it} {prefix}: {loss}')

def train(cfg):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # dataset
    feats, train_loader, eval_loader = setup_dataset(cfg)

    # model
    with open(cfg.model_config, 'r', encoding='utf-8') as f:
        config = json.loads(f.read(), object_pairs_hook=OrderedDict)
        print('1', config)
        config = ml_collections.ConfigDict(config)
        print('2', config)

    if cfg.restore_model_ckpt is not None:
        ckpt = torch.load(cfg.restore_model_ckpt)
        model_config = ckpt['model_config']
        print('3', model_config)
        model_config['esm2_model_file'] = cfg.restore_esm2_model
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

    model, device = setup_model(model, cfg)

    # optimizer
    optim = Optimizer(trainable_variables,
            base_lr=cfg.learning_rate, warmup_steps=cfg.warmup_steps, flat_steps=cfg.flat_steps,
            decay_steps=cfg.decay_steps, decay_type='linear', min_lr=1e-5)

    # loss
    loss_object = LossFactory(config.loss)
   
    # checkpoint
    def _save_checkpoint(it):
        ckpt_dir = os.path.join(cfg.prefix, 'checkpoints')
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
            cfg = cfg,
            train_steps = optim.cur_step), ckpt_file)

    # setup train
    model.train()

    running_loss = MetricDict()
    optim.zero_grad()
    batch_start_time = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.num_epoch):
        running_loss = MetricDict()
        optim.zero_grad()

        for it, batch in enumerate(train_loader):
            jt = (it + 1) % cfg.gradient_accumulation_it

            ctx = nullcontext if jt == 0 else model.no_sync
            with ctx():
                with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                    r = model(batch=batch, compute_loss=True)
                    loss_results = loss_object(r, batch)
                    loss = loss_results['loss'] / cfg.gradient_accumulation_it

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
                    v = v / cfg.gradient_accumulation_it
                    log_metric_dict(v, epoch, it, prefix=f'Loss/train@{k}')

                running_loss = MetricDict()

                batch_end_time = time.time()
                logging.info(f'{it} batch time {batch_end_time - batch_start_time} s.')
                batch_start_time = time.time()

                if optim.cur_step % cfg.checkpoint_it == 0:
                    _save_checkpoint(optim.cur_step)
