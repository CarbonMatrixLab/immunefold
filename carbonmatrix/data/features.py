from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.common import residue_constants
from carbonmatrix.data.base_features import registry_feature
from carbonmatrix.model.utils import batched_select

@registry_feature
def make_restype_atom_constants(batch, is_training=False):
    device = batch['seq'].device

    batch['atom14_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), batch['seq'])
    batch['atom14_atom_is_ambiguous'] = batched_select(torch.tensor(residue_constants.restype_atom14_is_ambiguous, device=device), batch['seq'])

    if 'residx_atom37_to_atom14' not in batch:
        batch['residx_atom37_to_atom14'] = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), batch['seq'])

    if 'atom37_atom_exists' not in batch:
        batch['atom37_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=device), batch['seq'])

    return batch

@registry_feature
def make_to_device(protein, fields, device, is_training=True):
    if isfunction(device):
        device = device()

    for k in fields:
        if k in protein:
            protein[k] = protein[k].to(device)
    return protein
