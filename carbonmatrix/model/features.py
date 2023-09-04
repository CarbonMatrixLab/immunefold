import os
import functools
from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.common import residue_constants
from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.model.utils import batched_select

_feats_fn = {}

def take1st(fn):
    """Supply all arguments but the first."""

    @functools.wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feats_fn
    _feats_fn[fn.__name__] = fc

    return fc

@take1st
def make_restype_atom_constants(batch, is_training=False):
    device = batch['seq'].device

    batch['atom14_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), batch['seq'])
    batch['atom14_atom_is_ambiguous'] = batched_select(torch.tensor(residue_constants.restype_atom14_is_ambiguous, device=device), batch['seq'])

    if 'residx_atom37_to_atom14' not in batch:
        batch['residx_atom37_to_atom14'] = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), batch['seq'])

    if 'atom37_atom_exists' not in batch:
        batch['atom37_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=device), batch['seq'])

    return batch

@take1st
def make_to_device(protein, fields, device, is_training=True):
    if isfunction(device):
        device = device()

    for k in fields:
        if k in protein:
            protein[k] = protein[k].to(device)
    return protein

@take1st
def make_selection(protein, fields, is_training=True):
    return {k: protein[k] for k in fields}

class FeatureBuilder:
    def __init__(self, config, is_training=True):
        self.config = config
        self.training = is_training

    def build(self, protein):
        for fn, kwargs in self.config:
            f = _feats_fn[fn](is_training=self.training, **kwargs)
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
