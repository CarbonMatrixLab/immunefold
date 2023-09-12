from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.data.seq import create_esm_seq, esm_alphabet
from carbonmatrix.common import residue_constants
from carbonmatrix.data.feature_factory import registry_feature
from carbonmatrix.model.utils import batched_select
from carbonmatrix.common.operator import pad_for_batch

@registry_feature
def make_restype_atom_constants(batch):
    device = batch['seq'].device

    batch['atom14_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), batch['seq'])
    batch['atom14_atom_is_ambiguous'] = batched_select(torch.tensor(residue_constants.restype_atom14_is_ambiguous, device=device), batch['seq'])

    if 'residx_atom37_to_atom14' not in batch:
        batch['residx_atom37_to_atom14'] = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), batch['seq'])

    if 'atom37_atom_exists' not in batch:
        batch['atom37_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=device), batch['seq'])

    return batch

@registry_feature
def make_esm_seq(batch,):
    device = batch['seq'].device
    bs = batch['seq'].shape[0]
    
    esm_seq = [torch.from_numpy(create_esm_seq(s)) for s in batch['str_seq']]
    
    max_len = max([x.shape[0] for x in esm_seq])
    padded_esm_seq = pad_for_batch(esm_seq, max_len, esm_alphabet.padding_idx)

    batch.update(
            esm_seq = padded_esm_seq.to(device=device),
            residx = torch.tile(torch.arange(max_len, device=device), [bs, 1]),
            )

    return batch

@registry_feature
def make_sliced_sample(batch, max_seq_len):
    batch_len = batch['seq'].shape[1]
    if batch_len <= max_seq_len:
        return batch
    
    for k, v in batch.items():
        if isinstance(k, list):
            batch[k] = [x[:max_seq_len] for x in v]
        elif isinstance(k, torch.Tensor):
            batch[k] = v[:, :max_seq_len]

    return batch
