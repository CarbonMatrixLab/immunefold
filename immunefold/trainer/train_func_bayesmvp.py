import os
import functools
import json
import logging
import random
from collections import OrderedDict
import time

import numpy as np
import pandas as pd
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler

from immunefold.model import BayesMVP
from immunefold.common import MetricDict
from immunefold.data.base_dataset import  TransformedDataLoader as DataLoader
from immunefold.data.base_dataset import collate_fn_seq
from immunefold.data.dataset import WeightedSeqDatasetFastaIO 
from immunefold.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from immunefold.trainer.losses.loss_factory import LossFactory
from immunefold.trainer import model_align
from immunefold.trainer import utils

def setup_model(model, device):
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        #find_unused_parameters=True,
        )

    #model._set_static_graph()

    logging.info('wrap model with nn.parallel.DistributedDataParallel class')

    return model

def setup_dataset(cfg):
    device = utils.get_device(cfg.gpu_list)

    logging.info('BayesMVP.feats: %s', cfg.transforms)

    dataset = WeightedSeqDatasetFastaIO(cfg.train_data, max_seq_len=cfg.max_seq_len)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    train_loader = DataLoader(
            dataset=dataset,
            feats=cfg.transforms,
            device = device,
            sampler=sampler,
            collate_fn=collate_fn_seq,
            batch_size=cfg.batch_size,
            drop_last=True,
            )

    return train_loader

def log_metric_dict(loss, it= '', prefix=''):
    if isinstance(loss, MetricDict):
        if prefix:
            prefix = f'{prefix}.'
        for k, v in loss.items():
            log_metric_dict(v, epoch, it, prefix=f'{prefix}{k}')
    elif isinstance(loss, torch.Tensor):
        loss = loss.item()
        logging.info(f'step= {it} {prefix}: {loss}')

def train(cfg):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # dataset
    train_loader = setup_dataset(cfg)

    # load model
    if cfg.get('restore_esm2_model', None) is not None:
        cfg.model.esm2_model_file = cfg.restore_esm2_model

    model_align.set_esm_lora_config(cfg.model, cfg.lora_r, cfg.lora_scaling)
    logging.info('final model config')
    logging.info(cfg.model)

    model = BayesMVP(config = cfg.model)
    
    trainable_variables = model_align.setup_esm_model(model, cfg.model_align)
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(f'trainable variable {n}')

    logging.info('BayesMVP.config: %s', cfg)

    device = utils.get_device(cfg.gpu_list)
    model = setup_model(model, device)

    # optimizer
    optim = Optimizer(trainable_variables,
            base_lr=cfg.learning_rate, warmup_steps=cfg.warmup_steps, flat_steps=cfg.flat_steps,
            decay_steps=cfg.decay_steps, decay_type='linear', min_lr=1e-5)
        
    # loss
    loss_object = LossFactory(cfg.loss)

    # checkpoint
    def _save_checkpoint(it):
        ckpt_dir = os.path.join(cfg.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)

        ckpt_file = os.path.join(ckpt_dir, f'step_{it}.ckpt')

        saved_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        torch.save(dict(
            model_state_dict = saved_model.impl.state_dict(),
            #optim_state_dict = optim.state_dict(),
            model_config = cfg.model,
            cfg = cfg,
            train_steps = optim.cur_step), ckpt_file)

    # setup train
    model.train()
    # save original model
    #_save_checkpoint(0)

    running_loss = MetricDict()
    optim.zero_grad()
    batch_start_time = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.num_epoch):
        optim.zero_grad()
        train_loader.set_epoch(epoch)
        running_loss = MetricDict()

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

            if jt == 0:
                logging.info(f'optim step= {optim.cur_step} lr= {optim.get_values()}')

                #optim.step()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                for k, v in running_loss.items():
                    v = v / cfg.gradient_accumulation_it
                    log_metric_dict(v, it=optim.cur_step, prefix=f'Loss/train@{k}')

                running_loss = MetricDict()

                batch_end_time = time.time()
                logging.info(f'{it} batch time {batch_end_time - batch_start_time} s.')
                batch_start_time = time.time()

                # if optim.cur_step % cfg.checkpoint_every_step == 0:
                #     _save_checkpoint(optim.cur_step)
