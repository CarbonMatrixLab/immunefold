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
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    pass

from abfold.model import AbFold, MetricDict
from abfold.trainer import dataset
from abfold.trainer.optimizer import OptipizerInverseSquarRootDecay as Optimizer
from abfold.trainer.loss import Loss
from abfold.common.ab_utils import calc_ab_metrics
from abfold.trainer import model_align

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
        output_device=args.local_rank,)
    model._set_static_graph()
    #find_unused_parameters=True)
    
    logging.info('wrap model with nn.parallel.DistributedDataParallel class')
    return model
    
def setup_dataset(args):
    world_size = (args.world_size)
    world_rank = args.world_rank
    local_rank = args.local_rank
    local_world_size = args.local_world_size
    assert world_size % local_world_size == 0
    num_nodes = world_size // local_world_size
    node_rank = (world_rank - world_rank % local_world_size) // local_world_size

    local_world_size_general = int(local_world_size * args.general_data_gpu_ratio)
    local_world_size_ig = local_world_size - local_world_size_general
    
    if local_rank < local_world_size_ig:
        data_type = 'ig'
        train_name_idx_file = args.train_name_idx
        train_data_file = args.train_data
        eval_name_idx_file = args.eval_name_idx
        eval_data_file = args.eval_data
        data_world_size = local_world_size_ig * num_nodes
        data_rank = node_rank * local_world_size_ig + local_rank
    else:
        data_type = 'general'
        train_name_idx_file = args.train_general_name_idx
        train_data_file = args.train_general_data
        eval_name_idx_file = args.eval_general_name_idx
        eval_data_file = args.eval_general_data
        data_world_size = local_world_size_general * num_nodes
        data_rank = node_rank * local_world_size_general + local_rank - local_world_size_ig

    train_name_idx = dataset.parse_cluster(train_name_idx_file)
    is_cluster_idx = True

    if data_type == 'general' and data_world_size < local_world_size:
        with open(args.train_name_idx) as f:
            num_ig = len(f.readlines)
        ig_batch_size = num_nodes * local_world_size_ig * args.batch_size * args.gradient_accumulation_it
        num_batch = num_ig // ig_batch_size
        train_reduce_num = num_batch * num_nodes * local_world_size_general * args.batch_size * args.gradient_accumulation_it
    else:
        full_batch_size = data_world_size * args.batch_size * args.gradient_accumulation_it
        train_reduce_num = len(train_name_idx) - len(train_name_idx) % full_batch_size
    
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
        data_dir=train_data_file, name_idx=train_name_idx,
        feats=feats, data_type=data_type,
        max_seq_len=args.max_seq_len, reduce_num = train_reduce_num,
        is_cluster_idx=is_cluster_idx,
        rank=data_rank, world_size=data_world_size,
        batch_size=args.batch_size)

    if eval_data_file is not None and eval_name_idx_file is not None and data_rank == 0:
        eval_name_idx = list(pd.read_csv(eval_name_idx_file, sep='\t')['name'])
        eval_loader = dataset.load(
            data_dir=eval_data_file, name_idx=eval_name_idx,
            feats=feats, data_type = data_type,
            is_training=False, batch_size=4 if data_type=='ig' else 1)
    else:
        eval_loader = None

    logging.info(f'data_type={data_type} node_rank={node_rank} world_rank={world_rank} local_rank={local_rank} data_rank={data_rank} data_world_size={data_world_size} train_reduce_num={train_reduce_num}')
    if eval_loader is not None:
        logging.info(f'eval_data_file= {eval_data_file} eval_name_idx_file= {eval_name_idx_file}')
    
    return data_type, feats, train_loader, eval_loader


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
    data_type, feats, train_loader, eval_loader = setup_dataset(args)

    # model
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
        config = ml_collections.ConfigDict(config)

    if args.restore_model_ckpt is not None:
        ckpt = torch.load(args.restore_model_ckpt)
        model = AbFold(config=ckpt['model_config'])
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

        trainable_variables = model_align.setup_model(model, config.align)
        #trainable_variables = model.parameters()
    else:
        model = AbFold(config = config.model)
        trainable_variables = model.parameters()

    logging.info('AbFold.config: %s', config)


    model = setup_model(model, args)

    # optimizer
    if args.lr_decay:
        optim = Optimizer(trainable_variables,
                base_lr=args.learning_rate, warmup_steps=args.warmup_steps, flat_steps=args.flat_steps, decay_type=args.lr_decay)
        if args.device == 'mlu':
            optim = ct.to(optim, torch.device('mlu'))
    else:
        optim = Adam(trainable_variables, lr=args.learning_rate)
    
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

        if eval_loader is not None and (epoch + 1) % args.eval_it == 0:
            model.eval()
            if data_type == 'general':
                evaluate_general_data(model, loss_object, eval_loader, epoch, args)
            elif data_type == 'ig':
                evaluate_ig_data(model, loss_object, eval_loader, epoch, args)
            model.train()

def evaluate_ig_data(model, loss_object, eval_loader, epoch, args):
    with torch.no_grad():
        eval_loss = MetricDict()
        count = 0

        metrics = []

        for batch in iter(eval_loader):
            logging.info(f'evaluate. len={batch["seq"].shape}')
            count += 1

            r = model(batch=batch, compute_loss=True)
            loss_results = loss_object(r, batch)
            for h, v in loss_results.items():
                if h == 'loss':
                    continue
                for kk, vv in v.items():
                    if 'loss' not in kk:
                        continue
                    eval_loss += MetricDict({f'{h}@{kk}': vv})

            for k, v in r['heads'].items():
                if k not in ['tmscore', 'metric']:
                    continue
                for kk, vv in v.items():
                    if 'loss' in kk:
                        eval_loss += MetricDict({f'{k}@{kk}': vv})

            for x, pid in enumerate(batch['name']):
                str_seq = batch['str_heavy_seq'][x] + batch['str_light_seq'][x]
                heavy_len = len(batch['str_heavy_seq'][x])
                N = len(str_seq)
                pred_ca = r['heads']['folding']['final_atom14_positions'][x,:,1].detach().cpu().numpy()  # (b l c d)
                gt_ca = batch['atom14_gt_positions'][x,:,1].detach().cpu().numpy()
                ca_mask = batch['atom14_gt_exists'][x,:,1].detach().cpu().numpy()
                cdr_def = batch['cdr_def'][x].detach().cpu().numpy()
                ca_mask = ca_mask * (cdr_def != -1)

                one_metric = OrderedDict({'name' : pid})
                rmsd = calc_ab_metrics(gt_ca, pred_ca, ca_mask, cdr_def)
                one_metric.update(rmsd)
                metrics.append(one_metric)

        for k, v in eval_loss.items():
            v /= count
            log_metric_dict(v, epoch, prefix=f'Loss/ig_eval@{k}')

        if len(metrics) > 0:
            columns = metrics[0].keys()
            metrics = zip(*map(lambda x:x.values(), metrics))
            df = pd.DataFrame(dict(zip(columns, metrics)))
            df.to_csv(os.path.join(args.prefix, 'checkpoints', f'rmsd_metric_epoch_{epoch}.csv'),
                sep='\t', index=False)

            head = [n for n in df.columns if n.endswith('_rmsd')]
            metric_mean = [np.mean(df[n].values) for n in head]
            with open(os.path.join(args.prefix, 'checkpoints', f'mean_rmsd_metric_epoch_{epoch}.csv'),
                'w') as fw:
                fw.write('\t'.join(head) + '\n')
                fw.write('\t'.join([f'{m:.2f}' for m in metric_mean]) + '\n')
            metric_str = ' '.join(f'{h}= {m:.2f}'for h, m in zip(head, metric_mean))
            logging.info(f'rmsds on evaluate data - epoch={epoch}: {metric_str}')

        return

def evaluate_general_data(model, loss_object, eval_loader, epoch, args):
    eval_loss = MetricDict()
    count = 0
    with torch.no_grad():
        for it, batch in enumerate(eval_loader):
            logging.info(f'it= {it} shape= {batch["seq"].shape}')
            if batch['seq'].shape[1] > 512:
                continue
            count += 1

            r = model(batch=batch, compute_loss=True)
            loss_results = loss_object(r, batch)
            #eval_loss += MetricDict({'all': r['loss']})
            for h, v in loss_results.items():
                if h == 'loss':
                    continue
                for kk, vv in v.items():
                    if 'loss' not in kk:
                        continue
                    eval_loss += MetricDict({f'{h}@{kk}': vv})
            
            for k, v in r['heads'].items():
                if k not in ['tmscore', 'metric']:
                    continue
                for kk, vv in v.items():
                    if 'loss' in kk:
                        eval_loss += MetricDict({f'{k}@{kk}': vv})


        for k, v in eval_loss.items():
            v /= count
            log_metric_dict(v, epoch, prefix=f'Loss/general_eval@{k}')
    return

