import torch
from torch.nn import functional as F
from einops import rearrange
import random
from carbonmatrix.common import residue_constants
from carbonmatrix.data.transform_factory import registry_transform
from carbonmatrix.model.r3 import rigids_from_3_points
from carbonmatrix.model.common_modules import pseudo_beta_fn
from carbonmatrix.model.utils import batched_select
from carbonmatrix.trainer import geometry

@registry_transform
def make_center_positions(batch, ):
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    ca = batch['atom14_gt_positions'][...,1,:]
    ca_mask = batch['atom14_gt_exists'][...,1]
    center = torch.sum(ca * ca_mask[...,None], dim=1) / (torch.sum(ca_mask, dim=1, keepdims=True) + 1e-12)
    
    batch['atom14_gt_positions'] = batch['atom14_gt_positions'] - rearrange(center, 'b c -> b () () c')

    return batch

@registry_transform
def make_atom14_alt_gt_positions(batch, is_training=True):
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    device = batch['seq'].device

    restype_atom_swap_index = batched_select(torch.tensor(residue_constants.restype_ambiguous_atoms_swap_index, device=device), batch['seq'])
    batch['atom14_alt_gt_positions'] = batched_select(batch['atom14_gt_positions'], restype_atom_swap_index, batch_dims=2)
    batch['atom14_alt_gt_exists'] = batched_select(batch['atom14_gt_exists'], restype_atom_swap_index, batch_dims=2)

    return batch

@registry_transform
def make_pseudo_beta(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch['pseudo_beta'], batch['pseudo_beta_mask'] = pseudo_beta_fn(
            batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists'])

    return batch

@registry_transform
def make_gt_frames(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)

    batch.update(
            geometry.atom37_to_frames(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))
    
    return batch

@registry_transform
def make_calpha3_frames(batch, is_training=True):
    calpha_pos = batch['atom37_gt_positions'][:,:,1]
    calpha_mask = batch['atom37_gt_exists'][:,:,1]
    
    batch.update(geometry.calpha3_to_frames(calpha_pos, calpha_mask))

    return batch

@registry_transform
def make_torsion_angles(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch.update(
            geometry.atom37_to_torsion_angles(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))

    return batch

@registry_transform
def make_gt_structure(batch, is_training=True):
    assert 'chain_id' in batch
    if is_training:
        chain_id_unique = torch.unique(batch['chain_id']).cpu().numpy()
        random.shuffle(chain_id_unique)
        chain_id_unique = chain_id_unique[chain_id_unique>2]
        num_to_gt = random.randint(0, len(chain_id_unique))
        gt_chain_id = torch.tensor(chain_id_unique[:num_to_gt]).to(batch['chain_id'].device)
        gt_mask = torch.isin(batch['chain_id'], gt_chain_id)
        
    else:
        gt_mask = torch.zeros_like(batch['chain_id'], dtype=torch.bool)
    batch.update(
        {"gt_mask": gt_mask,}
    )
    return batch


def make_atom37_positions(batch):
    device = batch['seq'].device
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    
    batch['atom37_gt_positions'] = batched_select(batch['atom14_gt_positions'], batch['residx_atom37_to_atom14'], batch_dims=2)
    batch['atom37_gt_exists'] = torch.logical_and(
            batched_select(batch['atom14_gt_exists'], batch['residx_atom37_to_atom14'], batch_dims=2),
            batch['atom37_atom_exists'])

    return batch
