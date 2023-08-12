import os
import functools
from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange

from abfold.common import residue_constants

from abfold.data.esm import ESMEmbeddingExtractor
from abfold.utils import default,exists
from abfold.data.utils import pad_for_batch
from abfold.model.utils import batched_select

try:
    from abfold.data.AntiBERTyRunner import AntiBERTyRunner 
except:
    pass

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
def make_esm_embed(protein, model_path, sep_pad_num=0, repr_layer=None, max_seq_len=None, device=None, return_attnw=False, field='esm_embed', is_training=True):
    esm_extractor = ESMEmbeddingExtractor.get(model_path, device=device)

    def _one(key, linker_mask=None):
        data_in = list(zip(protein['name'], protein[key]))
        data_out = esm_extractor.extract(data_in, repr_layer=repr_layer, device=device, return_attnw=return_attnw, linker_mask=linker_mask)
        return data_out

    data_type = protein['data_type']
    if data_type == 'general':
        batch_embed = _one('str_seq')
        
        if len(batch_embed['single']) == 1:
            batch_embed = batch_embed['single'][0]
        else:
            batch_embed = torch.stack(batch_embed['single'], dim=-1)
        
        protein[field] = batch_embed

        if return_attnw:
            protein[f'{field}_pair'] = batch_embed['pair']

        return protein
    elif data_type == 'ig':
        lengths = (len(h) + len(l) for h, l in zip(protein['str_heavy_seq'], protein['str_light_seq']))
        batch_length =  max(lengths)
        
        if sep_pad_num == 0:
            heavy_embed = _one('str_heavy_seq')
            light_embed = _one('str_light_seq')

            embed = [torch.cat([heavy_embed['single'][k,:len(x)], light_embed['single'][k,:len(y)]], dim=0) for k, (x,y) in enumerate(zip(protein['str_heavy_seq'],protein['str_light_seq']))]
        else:
            protein['sep_pad_seq'] = [h + 'G' * sep_pad_num + l for h, l in zip(protein['str_heavy_seq'], protein['str_light_seq'])]
            
            #linker_mask = [torch.tensor([0] * (1 + len(h)) + [1] * sep_pad_num + [0] * (1 + len(l)), dtype=torch.bool, device=device) \
            #        for h, l in zip(protein['str_heavy_seq'], protein['str_light_seq'])]

            #linker_mask = pad_for_batch(linker_mask, batch_length + sep_pad_num + 2, 'msk')

            embed = _one('sep_pad_seq', linker_mask=None)

            if len(embed['single']) == 1:
                embed = embed['single'][0]
            else:
                embed = torch.stack(embed['single'], dim=-1)
            
            embed = [torch.cat([
                embed[k,:len(x)],
                embed[k,len(x)+sep_pad_num:len(x)+sep_pad_num+len(y)]],
                dim=0) for k, (x,y) in enumerate(zip(protein['str_heavy_seq'],protein['str_light_seq']))]


        protein[field] = pad_for_batch(embed, batch_length, dtype='ebd')
        
        # FIXED ME
        if return_attnw:
            embed = [torch.cat([
                F.pad(light_embed['pair'][k,:len(x), :len(x)], (0, 0, 0, len(y)), value=0.),
                F.pad(heavy_embed['pair'][k,:len(y), :len(y)], (0, 0, len(x), 0), value=0.)], dim=0) for k, (x,y) in enumerate(zip(protein['st_light_seq'],protein['str_heavy_seq']))]


            batch_embed = pad_for_batch(embed, batch_length, dtype='pair')

            protein[f'{field}_pair'] = batch_embed

        return protein
    else:
        raise NotImplementedError(f'{data_type} not implemented.')

@take1st
def make_ablang_embed(protein, model_path, device=None, field='ablang_embed', is_training=True):
    data_type = protein['data_type']
    if data_type == 'general':
        bs, l = protein['seq'].shape[:2]
        batch_embed = torch.zeros((bs, l, 768), device=device)
        protein[field] = batch_embed

        return protein

    heavy_extractor = AbLangExtractor.get('heavy', os.path.join(model_path, 'heavy'), device=device)
    light_extractor = AbLangExtractor.get('light', os.path.join(model_path, 'light'), device=device)

    heavy_embed = heavy_extractor.rescoding(protein['str_heavy_seq'])
    light_embed = light_extractor.rescoding(protein['str_light_seq'])

    embed = [torch.cat([heavy_embed[k,:len(x)], light_embed[k,:len(y)]], dim=0) for k, (x,y) in enumerate(zip(protein['str_heavy_seq'],protein['str_light_seq']))]
    lengths = (e.shape[0] for e in embed)
    batch_length =  max(lengths)

    protein[field] = pad_for_batch(embed, batch_length, dtype='ebd')

    return protein

@take1st
def make_abrep_embed(protein, model_path, repr_layer=None, device=None, return_attnw=False, field='abrep_embed', is_training=True):
    data_type = protein['data_type']
    if data_type == 'general':
        '''
        bs, l = protein['seq'].shape[:2]
        batch_embed = torch.zeros((bs, l, 512), device=device)
        protein[field] = batch_embed

        if return_attnw:
            protein[f'{field}_pair'] = torch.zeros((bs, l, l, 512), device=device)
        '''
        return protein

    extractor = AbRepExtractor.get(model_path, device=device)

    heavy_embed = extractor.rescoding(protein['str_heavy_seq'], protein['str_heavy_numbering'], chain_name = 'Heavy', repr_layer=repr_layer, device=device, return_attnw=return_attnw)
    light_embed = extractor.rescoding(protein['str_light_seq'], protein['str_light_numbering'], chain_name = 'Light', repr_layer=repr_layer, device=device, return_attnw=return_attnw)
    
    # FIX ME
    if len(repr_layer) == 1:
        light_embed = light_embed['single'][0]
        heavy_embed = heavy_embed['single'][0]
    else:
        light_embed = torch.stack(light_embed['single'], dim=-1)
        heavy_embed = torch.stack(heavy_embed['single'], dim=-1)

    embed = [torch.cat([light_embed[k,:len(x)], heavy_embed[k,:len(y)]], dim=0) for k, (x,y) in enumerate(zip(protein['str_light_seq'],protein['str_heavy_seq']))]

    lengths = (e.shape[0] for e in embed)
    batch_length =  max(lengths)

    protein[field] = pad_for_batch(embed, batch_length, dtype='ebd')

    # FIX ME
    if return_attnw:
        embed = [torch.cat([
            F.pad(light_embed['pair'][k,:len(x), :len(x)], (0, 0, 0, len(y)), value=0.),
            F.pad(heavy_embed['pair'][k,:len(y), :len(y)], (0, 0, len(x), 0), value=0.)], dim=0) for k, (x,y) in enumerate(zip(protein['str_light_seq'],protein['str_heavy_seq']))]


        batch_embed = pad_for_batch(embed, batch_length, dtype='pair')

        protein[f'{field}_pair'] = batch_embed

    return protein

@take1st
def make_antiberty_embed(protein, model_path, vocab_path, device=None, field='antiberty_embed', is_training=True):
    data_type = protein['data_type']
    if data_type == 'general':
        bs, l = protein['seq'].shape[:2]
        batch_embed = torch.zeros((bs, l, 512), device=device)
        protein[field] = batch_embed

        return protein

    extractor = AntiBERTyRunner.get(model_path=model_path, vocab_path=vocab_path, device=device)

    heavy_embed = extractor.embed(protein['str_heavy_seq'])
    light_embed = extractor.embed(protein['str_light_seq'])

    embed = [torch.cat([heavy_embed[k][1:-1], light_embed[k][1:-1]], dim=0) for k, (x,y) in enumerate(zip(protein['str_heavy_seq'],protein['str_light_seq']))]
    lengths = (e.shape[0] for e in embed)
    batch_length =  max(lengths)

    protein[field] = pad_for_batch(embed, batch_length, dtype='ebd')

    return protein

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
        for fn, kwargs in default(self.config, []):
            f = _feats_fn[fn](is_training=self.training, **kwargs)
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
