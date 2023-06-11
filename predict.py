import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import resource
import json

from collections import OrderedDict
import ml_collections
import torch
import torch.multiprocessing as mp
from einops import rearrange

#from abfold.data import dataset
from abfold.trainer import dataset
from abfold.data.utils import save_ig_pdb, save_general_pdb

from abfold.model.abfold import AbFold

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    pass

def worker_setup(rank, log_queue, args):  # pylint: disable=redefined-outer-name
    # logging
    logger = logging.getLogger()
    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)

    if (args.gpu_list or args.map_location) and torch.cuda.is_available():
        world_size = len(args.gpu_list) if args.gpu_list else 1
        if len(args.gpu_list) > 1:
            logging.info('torch.distributed.init_process_group: rank=%d@%d, world_size=%d', rank, args.gpu_list[rank] if args.gpu_list else 0, world_size)
            torch.distributed.init_process_group(
                    backend='nccl',
                    #init_method=f'file://{args.ipc_file}',
                    rank=rank, world_size=world_size)

def worker_cleanup(args):  # pylint: disable=redefined-outer-name
    if (args.gpu_list or args.map_location) and torch.cuda.is_available():
        if len(args.gpu_list) > 1:
            torch.distributed.destroy_process_group()

def worker_device(rank, args):  # pylint: disable=redefined-outer-name
    if args.device == 'gpu':
        return args.gpu_list[rank]
    elif args.device == 'mlu':
        ct.set_device(rank)
        return ct.mlu_device(rank)
    else:
        return torch.device('cpu')

def worker_load(rank, args):  # pylint: disable=redefined-outer-name
    def _feats_gen(feats, device):
        for fn, opts in feats:
            if 'device' in opts:
                opts['device'] = device
            yield fn, opts
    
    device = worker_device(rank, args)
    
    #with open(args.model_config, 'r', encoding='utf-8') as f:
    #    config = json.loads(f.read())
    #    config = ml_collections.ConfigDict(config)

    checkpoint = torch.load(args.model, map_location='cpu')
    #model = checkpoint['model']
    print(checkpoint.keys())
    model_config = checkpoint['model_config']
    model_state_dict = checkpoint['model_state_dict']
   
    #model_config.num_recycle = 0
    model = AbFold(config=model_config)
    model.load_state_dict(model_state_dict, strict=True)
    
    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.loads(f.read())
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device

    model = model.to(device=device)
    model.eval()

    return list(_feats_gen(feats, device)), model

def postprocess_one_ig(name, str_heavy_seq, str_light_seq, coord, args):
    pdb_file = f'{args.output_dir}/{name}.pdb'
    
    print(name, str_heavy_seq, str_light_seq)

    save_ig_pdb(str_heavy_seq, str_light_seq, coord, pdb_file)

def postprocess_predictions_ig(batch, coords, args):
    fields = ('name', 'str_heavy_seq', 'str_light_seq')
    names, str_heavy_seqs, str_light_seqs = map(batch.get, fields)

    for i, (name, str_heavy_seq, str_light_seq) in enumerate(zip(names, str_heavy_seqs, str_light_seqs)):
        postprocess_one_ig(name, str_heavy_seq, str_light_seq, coords[i, :len(str_heavy_seq)+len(str_light_seq)], args)

def postprocess_one_general(name, str_seq, coord, args):
    pdb_file = f'{args.output_dir}/{name}.pdb'
    
    save_general_pdb(str_seq, coord, pdb_file)

def postprocess_predictions_general(batch, coords, args):
    fields = ('name', 'str_seq')
    names, str_seqs = map(batch.get, fields)

    for i, (name, str_seq) in enumerate(zip(names, str_seqs)):
        postprocess_one_general(name, str_seq, coords[i, :len(str_seq)], args)

def postprocess_predictions(batch, coords, args):
    if args.mode == 'general':
        postprocess_predictions_general(batch, coords, args)
    elif args.mode == 'ig':
        postprocess_predictions_ig(batch, coords, args)

def evaluate(rank, log_queue, args):
    worker_setup(rank, log_queue, args)

    feats, model = worker_load(rank, args)
    logging.info('feats: %s', feats)
    # logging.info('model: %s', model)
    
    device = worker_device(rank, args)
    name_idx = []
    with open(args.name_idx) as f:
        name_idx = [x.strip() for x in f]
    
    test_loader = dataset.load(
        data_dir=args.data_dir,
        name_idx=name_idx,
        feats=feats,
        batch_size=args.batch_size,
        data_type=args.mode)

    for i, batch in enumerate(test_loader):
        try:
            logging.debug('name: %s', ','.join(batch['name']))
            logging.debug('len : %s', batch['seq'].shape[1])
            #logging.debug('seq : %s', batch['str_seq'][0])
            #if batch['seq'].shape[1] > 600:
            #    continue

            with torch.no_grad():
                r = model(batch=batch, compute_loss=False)

            assert 'folding' in r['heads'] and 'final_atom14_positions' in r['heads']['folding']
            coords = r['heads']['folding']['final_atom14_positions']  # (b l c d)
            
            postprocess_predictions(batch, coords.to('cpu').numpy(), args)
        except:
            logging.error('fails in predicting', batch['name'])

    worker_cleanup(args)

def main(args):
    #init_pyrosetta()
    
    mp.set_start_method('spawn', force=True)

    # logging
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                args.output_dir,
                f'{os.path.splitext(os.path.basename(__file__))[0]}.log'))]

    def handler_apply(h, f, *arg):
        f(*arg)
        return h
    level = logging.DEBUG if args.verbose else logging.INFO
    handlers = list(map(lambda x: handler_apply(
        x, x.setLevel, level), handlers))
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
    handlers = list(map(lambda x: handler_apply(
        x, x.setFormatter, logging.Formatter(fmt)), handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    log_queue = mp.Queue(-1)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    logging.info('-----------------')
    logging.info('Arguments: %s', args)
    logging.info('-----------------')

    if len(args.gpu_list) > 1:
        mp.spawn(evaluate, args=(log_queue, args),
                nprocs=len(args.gpu_list) if args.gpu_list else 1,
                join=True)
    else:
        evaluate(args.gpu_list[0], log_queue, args)

    logging.info('-----------------')
    logging.info('Resources(myself): %s',
                 resource.getrusage(resource.RUSAGE_SELF))
    logging.info('Resources(children): %s',
                 resource.getrusage(resource.RUSAGE_CHILDREN))
    logging.info('-----------------')

    listener.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu', 'mlu'], default='gpu')
    
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--mode', type=str, choices=['ig', 'general'], required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
