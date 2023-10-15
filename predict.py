import os
import logging
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from carbonmatrix.data.pdbio import save_pdb
from carbonmatrix.model.carbonfold import CarbonFold
from carbonmatrix.data.dataset import SeqDatasetDirIO, SeqDatasetFastaIO
from carbonmatrix.trainer.dataset import StructureDatasetNpzIO
from carbonmatrix.data.base_dataset import TransformedDataLoader as DataLoader
from carbonmatrix.data.base_dataset import collate_fn_seq
from carbonmatrix.trainer.base_dataset import collate_fn_seq, collate_fn_struc
from carbonmatrix.sde.se3_diffuser import SE3Diffuser
from carbonmatrix.model import quat_affine 
from carbonmatrix.common.confidence import compute_plddt

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

def save_batch_pdb(values, batch, pdb_dir, data_type='general', step=None):
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
        multimer_str_seq = multimer_str_seqs[i]
        str_seq = str_seqs[i]
        chain_ids = ['H', 'L'] if data_type == 'ig' else None
        single_plddt = None if plddt is None else plddt[i]
        save_pdb(multimer_str_seq, pred_atom14_coords[i, :len(str_seq)], pdb_file, chain_ids, single_plddt)

    return

def esmfold(model, batch, cfg):
    data_type = cfg.get('data_type', 'general')
    assert (data_type == 'general')

    with torch.no_grad():
        ret = model(batch)
    save_batch_pdb(ret, batch, cfg.output_dir, data_type)

def carbonfold(model, batch, cfg):
    data_type = cfg.get('data_type', 'general')
    assert (data_type == 'general')

    bs = batch['seq'].shape[0]
    device = batch['seq'].device

    diffuser = SE3Diffuser.get(cfg.transforms.make_sample_ref.se3_conf)

    def _update_batch(batch, values, t):
        mask = batch['mask']
        
        batch_t = torch.full((bs,), t, device=device)
        
        # curret regidts_t
        rigids_t = batch['rigids_t']
        (_, trans_t) = rigids_t

        pred_rigids_0 = values['heads']['structure_module']['traj'][-1]
        (pred_rot_0, pred_trans_0) = pred_rigids_0
        
        # pred_rot_score for time t
        pred_delta_quat = values['heads']['structure_module']['delta_quat']
        quat_inv0_t = quat_affine.quat_invert(pred_delta_quat)
        pred_rot_score = diffuser.calc_rot_score(quat_inv0_t, batch_t)

        # pred_trans_score for time t
        pred_trans_score = diffuser.calc_trans_score(trans_t, pred_trans_0, batch_t)

        # pred score for time t
        score_t = (pred_rot_score, pred_trans_score)
        

        # rigidts_{t_1} which is denoised rigidts_t 
        rigids_t_1 = diffuser.reverse(rigids_t, score_t, mask, batch_t, dt, cfg.noise_scale)
        
        batch.update(t=batch_t, rigids_t=rigids_t_1)

        return batch
    
    def _update_batch2(batch, values, t):
        mask = batch['mask']
        
        batch_t = torch.full((bs,), t, device=device)
        
        # curret regidts_t
        rigids_t = batch['rigids_t']
        (_, trans_t) = rigids_t

        pred_rigids_0 = values['heads']['structure_module']['traj'][-1]
        (pred_rot_0, pred_trans_0) = pred_rigids_0

        pred_quat_0 = quat_affine.matrix_to_quaternion(pred_rot_0)
        
        batch.update(t=batch_t, rigids_t=(pred_quat_0, pred_trans_0))

        return batch

    def _compute_plddt(values, mask):
        logits = values['heads']['predicted_lddt']['logits']
        plddt = compute_plddt(logits)
        full_plddt = torch.sum(plddt * mask, dim=1) / torch.sum(mask, dim=1)

        str_plddt = ','.join([str(x.item()) for x in full_plddt.to('cpu')])
        logging.info(f'plddt= {str_plddt}')

        return plddt, full_plddt

    # step 0
    logging.info('step= 0, time=1.0')
    with torch.no_grad():
        ret = model(batch, compute_loss=True)

    plddt, full_plddt = _compute_plddt(ret, batch['mask'])
    ret.update(plddt=plddt)

    save_batch_pdb(ret, batch, cfg.output_dir, data_type, step=0)
    
    dt = 1. / cfg.T
    timesteps = np.linspace(1., 0., cfg.T + 1)[1:-1]

    for i, t in enumerate(timesteps):
        logging.info(f'step= {i+1}, time={t}')

        batch = _update_batch(batch, ret, t)
        with torch.no_grad():
            ret = model(batch, compute_loss=True)

        plddt, full_plddt = _compute_plddt(ret, batch['mask'])
        ret.update(plddt=plddt)

        save_batch_pdb(ret, batch, cfg.output_dir, data_type, step=i+1)
   
    # save last step
    save_batch_pdb(ret, batch, cfg.output_dir, data_type)

def abfold(model, batch, cfg):
    data_type = cfg.get('data_type', 'ig')
    assert (data_type == 'ig')

    with torch.no_grad():
        ret = model(batch)
    save_batch_pdb(ret, batch, cfg.output_dir, data_type)

def abdesign(model, batch, cfg):
    raise NotImplementedError('abdesign mode not implemented yet!')

def carbonnovo(model, batch, cfg):
    raise NotImplementedError('carbonnovo mode not implemented yet!')

def predict_batch(model, batch, cfg):
    mode = cfg.get('mode', 'esmfold')
    if mode == 'esmfold':
        esmfold(model, batch, cfg)
    elif mode == 'carbonfold':
        carbonfold(model, batch, cfg)
    elif mode == 'abfold':
        abfold(model, batch, cfg)
    elif mode == 'abdesign':
        abdesign(model, batch, cfg)
    elif mode == 'carbonnovo':
        carbonnovo(model, batch, cfg)
    else:
        raise NotImplementedError(f'mode {args.mode} not implemented yet!')

def predict(cfg):
    if cfg.data_io == 'dir':
        dataset = SeqDatasetDirIO(cfg.test_data, cfg.test_name_idx)
        collate_fn = collate_fn_seq
    elif cfg.data_io == 'fasta':
        dataset = SeqDatasetFastaIO(cfg.test_data)
        collate_fn = collate_fn_seq
    elif cfg.data_io == 'npz':
        dataset = StructureDatasetNpzIO(cfg.test_data, cfg.test_name_idx, 1024)
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

    ckpt = torch.load(cfg.restore_model_ckpt)

    if cfg.restore_esm2_model is not None:
        cfg.model.esm2_model_file = cfg.restore_esm2_model

    logging.info(f'esm2-model-{cfg.model.esm2_model_file}')

    model = CarbonFold(config = cfg.model)
    model.impl.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()


    for batch in test_loader:
        logging.info('names= {}'.format(','.join(batch['name'])))
        logging.info('str_len= {}'.format(','.join([str(len(x)) for x in batch['str_seq']])))

        predict_batch(model, batch, cfg)


@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg : DictConfig):
    setup(cfg)

    predict(cfg)

if __name__ == '__main__':
    main()
