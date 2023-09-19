import os
import logging
from logging.handlers import QueueHandler, QueueListener

import torch
import hydra
from omegaconf import DictConfig

from carbonmatrix.data.pdbio import save_pdb
from carbonmatrix.model.carbonfold import CarbonFold
from carbonmatrix.data.dataset import SeqDatasetDirIO, SeqDatasetFastaIO
from carbonmatrix.data.base_dataset import TransformedDataLoader as DataLoader
from carbonmatrix.data.base_dataset import collate_fn_seq

class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f'Rank {self._rank} | {record.msg}'
        return True

def setup(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'predict.log'))

    level = logging.DEBUG if cfg.verbose else logging.INFO
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt))
        h.addFilter(WorkerLogFilter(rank=0))
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
    logging.info(f'Arguments: {cfg}')
    logging.info('-----------------')

def to_numpy(x):
    return x.detach().cpu().numpy()

def save_batch_pdb(values, batch, pdb_dir, data_type='general'):
    N = len(batch['str_seq'])
    pred_atom14_coords = to_numpy(values['heads']['structure_module']['final_atom14_positions'])
    names = batch['name']
    str_seqs = batch['str_seq']
    multimer_str_seqs = batch['multimer_str_seq']

    for i in range(N):
        pdb_file = os.path.join(pdb_dir, f'{names[i]}.pdb')
        multimer_str_seq = multimer_str_seqs[i]
        str_seq = str_seqs[i]
        chain_ids = ['H', 'L'] if data_type == 'ig' else None
        save_pdb(multimer_str_seq, pred_atom14_coords[i, :len(str_seq)], pdb_file, chain_ids)

    return

def predict(cfg):
    if cfg.data_io == 'dir':
        dataset = SeqDatasetDirIO(cfg.test_data, cfg.test_name_idx)
    elif cfg.data_io == 'fasta':
        dataset = SeqDatasetFastaIO(cfg.test_data)
    else:
        raise NotImplementedError(f'data io {cfg.data_io} not implemented')

    device = cfg.gpu

    test_loader = DataLoader(
            dataset=dataset,
            feats=cfg.transforms,
            device = device,
            collate_fn=collate_fn_seq,
            batch_size=cfg.batch_size,
            drop_last=False,
            )
    
    ckpt = torch.load(cfg.restore_model_ckpt)
    model = CarbonFold(config = cfg.model)
    model.impl.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
   
    data_type = cfg.get('data_type', 'general')

    for batch in test_loader:
        print(batch['str_seq'], 'str_seq')
        print(batch['multimer_str_seq'], 'multimer')
        print(batch['seq'].shape, 'seq')
        
        with torch.no_grad():
            ret = model(batch)
        
        save_batch_pdb(ret, batch, cfg.output_dir, data_type)

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg : DictConfig):
    setup(cfg)
    
    predict(cfg)

if __name__ == '__main__':
    main()
