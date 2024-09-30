import os
import logging
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from einops import rearrange

from carbonmatrix.data.pdbio import save_pdb
from carbonmatrix.model.immunefold import ImmuneFold
from carbonmatrix.data.dataset import SeqDatasetDirIO, SeqDatasetFastaIO, AbStructureDataNpzIO
from carbonmatrix.data.base_dataset import TransformedDataLoader as DataLoader
from carbonmatrix.data.base_dataset import collate_fn_seq, collate_fn_struc
from carbonmatrix.common.confidence import compute_plddt, compute_ptm

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

def save_batch_pdb(values, batch, pdb_dir, step=None, mode=None):
    N = len(batch['str_seq'])
    pred_atom14_coords = to_numpy(values['heads']['structure_module']['final_atom14_positions'])
    names = batch['name']
    str_seqs = batch['str_seq']
    multimer_str_seqs = batch['multimer_str_seq']
    plddt = None if 'plddt' not in values else to_numpy(values['plddt'])

    for i in range(N):
        if step is None:
            pdb_file = os.path.join(pdb_dir, f'{names[i]}.pdb')
        else:
            pdb_file = os.path.join(pdb_dir, f'{names[i]}.step{step}.pdb')

        chain_ids = names[i].split('_')[1:]
        if len(chain_ids) < 2:
            chain_ids = None
        else:
            chain_ids = list(filter(None, chain_ids))
        

        multimer_str_seq = multimer_str_seqs[i]
        str_seq = str_seqs[i]
        single_plddt = None if plddt is None else plddt[i]
        save_pdb(multimer_str_seq, pred_atom14_coords[i, :len(str_seq)], pdb_file, chain_ids, single_plddt)

    return


def _compute_plddt(values, mask):
    logits = values['heads']['predicted_lddt']['logits']
    plddt = compute_plddt(logits)
    full_plddt = torch.sum(plddt * mask, dim=1) / torch.sum(mask, dim=1)

    str_plddt = ','.join([str(x.item()) for x in full_plddt.to('cpu')])
    
    logging.info(f'plddt= {str_plddt}')

    return plddt, full_plddt

def _compute_ptm(values, mask, chain_id=None, interface=True, batch=None):
    logits = values['heads']['predicted_aligned_error']['logits']
    breaks = values['heads']['predicted_aligned_error']['breaks']
    ptm = compute_ptm(logits, breaks, mask)
    if interface and chain_id is not None:
        iptm = compute_ptm(logits, breaks, mask, chain_id, interface)
        ptm = 0.8 * iptm + 0.2 * ptm
    str_ptm = ','.join([str(x.item()) for x in ptm.to('cpu')])
    logging.info(f'ptm= {str_ptm}')

    return ptm

def immunefold(model, batch, cfg):

    with torch.no_grad():
        ret = model(batch, compute_loss=True)
        ptm = _compute_ptm(ret, batch['mask'], chain_id=batch['chain_id'], interface=True, batch=batch)
        plddt, full_plddt = _compute_plddt(ret, batch['mask'])
        ret.update(ptm=ptm, plddt=plddt)

    save_batch_pdb(ret, batch, cfg.output_dir)

def predict(cfg):
    if cfg.data_io == 'dir':
        dataset = SeqDatasetDirIO(cfg.test_data, cfg.test_name_idx)
        collate_fn = collate_fn_seq
    elif cfg.data_io == 'fasta':
        dataset = SeqDatasetFastaIO(cfg.test_data)
        collate_fn = collate_fn_seq
    elif cfg.data_io == 'abag':
        dataset = AbStructureDataNpzIO(cfg.fasta, cfg.ag, cfg.get('contact_idx', None))
        collate_fn = collate_fn_struc
    else:
        raise NotImplementedError(f'data io {cfg.data_io} not implemented')

    device = cfg.gpu

    test_loader = DataLoader(
            dataset=dataset,
            feats=cfg.transforms,
            device = device,
            collate_fn=collate_fn,
            batch_size=cfg.batch_size,
            drop_last=False,
            )

    ckpt = torch.load(cfg.restore_model_ckpt, map_location='cpu')

    if cfg.restore_esm2_model is not None:
        cfg.model.esm2_model_file = cfg.restore_esm2_model

    logging.info(f'esm2-model-{cfg.model.esm2_model_file}')

    model = ImmuneFold(config = cfg.model)
    model.impl.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()


    for batch in test_loader:
        logging.info('names= {}'.format(','.join(batch['name'])))
        logging.info('str_len= {}'.format(','.join([str(len(x)) for x in batch['str_seq']])))
        immunefold(model, batch, cfg)


@hydra.main(version_base=None, config_path="config", config_name="inference_tcr")
def main(cfg : DictConfig):
    setup(cfg)
    predict(cfg)

if __name__ == '__main__':
    main()
