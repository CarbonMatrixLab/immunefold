import os
import logging
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from einops import rearrange

from carbonmatrix.data.pdbio import save_pdb
from carbonmatrix.model.carbonfold import CarbonFold
from carbonmatrix.data.dataset import SeqDatasetDirIO, SeqDatasetFastaIO
from carbonmatrix.trainer.dataset import StructureDatasetNpzIO
from carbonmatrix.data.base_dataset import TransformedDataLoader as DataLoader
from carbonmatrix.data.base_dataset import collate_fn_seq
from carbonmatrix.data.seq import create_residx
from carbonmatrix.trainer.base_dataset import collate_fn_seq, collate_fn_struc
from carbonmatrix.sde.se3_diffuser import SE3Diffuser
from carbonmatrix.model import quat_affine
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
        chain_ids = names[i].split('_')[1:]
        chain_ids = list(filter(None, chain_ids))
        multimer_str_seq = multimer_str_seqs[i] if data_type == 'ig' else None
        str_seq = str_seqs[i]
        # chain_ids = ['H', 'L'] if data_type == 'ig' else None
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

def _compute_ptm(values, mask, chain_id=None, interface=True):
    logits = values['heads']['predicted_aligned_error']['logits']
    breaks = values['heads']['predicted_aligned_error']['breaks']
    ptm = compute_ptm(logits, breaks, mask)
    if interface and chain_id is not None:
        iptm = compute_ptm(logits, breaks, mask, chain_id, interface)
        ptm = 0.8 * iptm + 0.2 * ptm

    str_ptm = ','.join([str(x.item()) for x in ptm.to('cpu')])
    logging.info(f'ptm= {str_ptm}')

    return ptm

def esmfold(model, batch, cfg):
    data_type = cfg.get('data_type', 'general')
    assert (data_type == 'general')

    with torch.no_grad():
        ret = model(batch, compute_loss=True)

    plddt, full_plddt = _compute_plddt(ret, batch['mask'])
    ret.update(plddt=plddt)

    ptm = _compute_ptm(ret, batch['mask'])
    ret.update(ptm=ptm)

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

        batch.update(t=batch_t, rigids_t=rigids_t_1, prev_pos=pred_trans_0)

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

    def _update_batch3(batch, values, t):
        mask = batch['mask']

        batch_t = torch.full((bs,), t, device=device)

        pred_rigids_0 = values['heads']['structure_module']['traj'][-1]
        (pred_rot_0, pred_trans_0) = pred_rigids_0
        pred_quat_0 = quat_affine.matrix_to_quaternion(pred_rot_0)

        center = torch.sum(pred_trans_0 * mask[...,None], dim=1) / (torch.sum(mask, dim=1, keepdims=True) + 1e-12)

        pred_trans_0 = pred_trans_0  - rearrange(center, 'b c -> b () c')


        # update
        ret = diffuser.forward_marginal((pred_quat_0, pred_trans_0), batch_t)
        # rigidts_{t_1} which is denoised rigidts_t
        batch.update(t=batch_t, rigids_t=ret['rigids_t'], prev_pos=pred_trans_0)

        return batch

    # step 0
    logging.info('step= 0, time=1.0')
    with torch.no_grad():
        ret = model(batch, compute_loss=True)

    #plddt, full_plddt = _compute_plddt(ret, batch['mask'])
    #ret.update(plddt=plddt)

    ptm = _compute_ptm(ret, batch['mask'])
    ret.update(ptm=ptm)

    save_batch_pdb(ret, batch, cfg.output_dir, data_type, step=0)

    dt = 1. / cfg.T
    timesteps = np.linspace(1., 0., cfg.T + 1)[1:-1]

    for i, t in enumerate(timesteps):
        logging.info(f'step= {i+1}, time={t}')

        batch = _update_batch3(batch, ret, t)
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

    def _change_order(pair_seq_lengths:list, x:torch.Tensor, offset=0):
        new_x = []
        for i, (seq_len1, seq_len2) in enumerate(pair_seq_lengths):
            new_x.append(torch.cat([x[i, :offset],x[i,offset+seq_len1:offset+seq_len1+seq_len2], x[i, offset:offset+seq_len1], x[i, offset+seq_len1+seq_len2:]], dim=0))
        return torch.stack(new_x, dim=0)

    def _light_first(batch):
        new_batch = {}
        new_batch.update(name=batch['name'], batch_len=batch['batch_len'])

        pair_seq_lengths = []
        new_str_seq, new_multimer_str_seq = [], []

        for multimer_str_seq in batch['multimer_str_seq']:
            seq_len1, seq_len2 = len(multimer_str_seq[0]), len(multimer_str_seq[1])

            pair_seq_lengths.append((seq_len1, seq_len2))

            new_str_seq.append(':'.join(multimer_str_seq[::-1]))
            new_multimer_str_seq.append(multimer_str_seq[::-1])

        new_batch.update(str_seq=new_str_seq, multimer_str_seq=new_multimer_str_seq)

        for n in ('seq', 'mask', 'atom14_atom_exists', 'atom14_atom_is_ambiguous', 'residx_atom37_to_atom14', 'atom37_atom_exists'):
            new_batch[n] = _change_order(pair_seq_lengths, batch[n])

        # esm_seq
        new_batch['esm_seq'] = _change_order(pair_seq_lengths, batch['esm_seq'], offset=1)

        # residx
        new_batch['residx'] = torch.from_numpy(
            create_residx(new_batch['multimer_str_seq'], new_batch['esm_seq'].shape[1], new_batch['esm_seq'].shape[0])).to(new_batch['esm_seq'].device)

        return new_batch

    def _rank_by_confidence(heavy_first_ret, light_first_ret, heavy_first_batch, light_first_batch, condence_type='ptm'):
        ptm1 = _compute_ptm(heavy_first_ret, batch['mask'], chain_id=batch['chain_id'], interface=True)
        plddt1, full_plddt1 = _compute_plddt(heavy_first_ret, batch['mask'])
        heavy_first_ret.update(ptm=ptm1, plddt=plddt1)

        ptm2 = _compute_ptm(light_first_ret, batch['mask'], chain_id=batch['chain_id'], interface=True)
        plddt2, full_plddt2 = _compute_plddt(light_first_ret, batch['mask'])
        light_first_ret.update(ptm=ptm2, plddt=plddt2)


        final_str_seq, final_multimer_str_seq = [], []
        final_pred_atom14_coords, final_plddt = [], []

        batch_size = len(heavy_first_batch['name'])
        for i in range(batch_size):
            name = heavy_first_batch['name'][i]

            #heavy_first_is_btter = (ptm1[i].item() > ptm2[i].item())
            #print(name, 'ptm', ptm1[i], ptm2[i])

            heavy_first_is_btter = (full_plddt1[i].item() > full_plddt2[i].item())
            print(name, 'plddt', full_plddt1[i], full_plddt2[i])

            final_str_seq.append(heavy_first_batch['str_seq'][i])
            final_multimer_str_seq.append(heavy_first_batch['multimer_str_seq'][i])

            if heavy_first_is_btter:
                final_pred_atom14_coords.append(heavy_first_ret['heads']['structure_module']['final_atom14_positions'][i])
                final_plddt.append(plddt1[i])
            else:
                pair_seq_lengths = [(len(light_first_batch['multimer_str_seq'][i][0]), len(light_first_batch['multimer_str_seq'][i][1]))]
                final_pred_atom14_coords.append(
                    _change_order(pair_seq_lengths, light_first_ret['heads']['structure_module']['final_atom14_positions'][i:i+1])[0]
                    )
                final_plddt.append(
                    _change_order(pair_seq_lengths, plddt2[i:i+1])[0],
                    )

        final_pred_atom14_coords = torch.stack(final_pred_atom14_coords, dim=0)
        final_plddt = torch.stack(final_plddt, dim=0)

        final_batch = dict(
            name=heavy_first_batch['name'],
            batch_len=heavy_first_batch['batch_len'],
            str_seq=final_str_seq,
            multimer_str_seq=final_multimer_str_seq,
            )
        final_ret = dict(
            heads=dict(
                structure_module=dict(
                    final_atom14_positions=final_pred_atom14_coords,
                )
            ),
            plddt=final_plddt,
        )

        return final_ret, final_batch

    with torch.no_grad():
        heavy_first_ret = model(batch, compute_loss=True)

        new_batch = _light_first(batch)

        light_first_ret = model(new_batch, compute_loss=True)

    final_ret, final_batch = _rank_by_confidence(heavy_first_ret, light_first_ret, batch, new_batch)
    save_batch_pdb(final_ret, final_batch, cfg.output_dir, data_type)
    #save_batch_pdb(heavy_first_ret, batch, cfg.output_dir, data_type)

def tcrfold(model, batch, cfg):
    data_type = cfg.get('data_type', 'ig')
    assert (data_type == 'ig')

    def _rank_by_confidence(ret, batch, condence_type='ptm'):
        ptm1 = _compute_ptm(ret, batch['mask'], chain_id=batch['chain_id'], interface=True)
        plddt1, full_plddt1 = _compute_plddt(ret, batch['mask'])
        ret.update(ptm=ptm1, plddt=plddt1)

        final_str_seq, final_multimer_str_seq = [], []
        final_pred_atom14_coords, final_plddt = [], []

        batch_size = len(batch['name'])
        for i in range(batch_size):
            name = batch['name'][i]

            #heavy_first_is_btter = (ptm1[i].item() > ptm2[i].item())
            #print(name, 'ptm', ptm1[i], ptm2[i])

            print(name, 'plddt', full_plddt1[i])

            final_str_seq.append(batch['str_seq'][i])
            final_multimer_str_seq.append(batch['multimer_str_seq'][i])

            final_pred_atom14_coords.append(ret['heads']['structure_module']['final_atom14_positions'][i])
            final_plddt.append(plddt1[i])

        final_pred_atom14_coords = torch.stack(final_pred_atom14_coords, dim=0)
        final_plddt = torch.stack(final_plddt, dim=0)

        final_batch = dict(
            name=batch['name'],
            batch_len=batch['batch_len'],
            str_seq=final_str_seq,
            multimer_str_seq=final_multimer_str_seq,
            )
        final_ret = dict(
            heads=dict(
                structure_module=dict(
                    final_atom14_positions=final_pred_atom14_coords,
                )
            ),
            plddt=final_plddt,
        )

        return final_ret, final_batch

    with torch.no_grad():
        ret = model(batch, compute_loss=True)

        # new_batch = _light_first(batch)

        # light_first_ret = model(new_batch, compute_loss=True)

    final_ret, final_batch = _rank_by_confidence(ret, batch)
    save_batch_pdb(final_ret, final_batch, cfg.output_dir, data_type)
    #save_batch_pdb(heavy_first_ret, batch, cfg.output_dir, data_type)



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
    elif mode == 'tcrfold':
        tcrfold(model, batch, cfg)
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

    ckpt = torch.load(cfg.restore_model_ckpt, map_location='cpu')

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


@hydra.main(version_base=None, config_path="config", config_name="inference_tcr")
def main(cfg : DictConfig):
    setup(cfg)

    predict(cfg)

if __name__ == '__main__':
    main()
