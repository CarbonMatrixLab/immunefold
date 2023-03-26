import os
import functools
import logging
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from abfold.common import residue_constants
from abfold.common.utils import str_seq_to_index
from abfold.model.features import FeatureBuilder
import abfold.trainer.features

from abfold.data.utils import pad_for_batch

logger = logging.getLogger(__file__)


class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, rank, word_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.word_size = word_size

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.word_size == self.rank:
                #logger.info(f'rank= {self.rank} idx= {idx} {sample["name"]}')
                yield sample
    
    def collate_fn(self, *args, **kwargs):
        return self.dataset.collate_fn(*args, **kwargs)


class SeqDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.name_idx = name_idx
        self.data_dir = data_dir

    def __iter__(self):
        for name in self.name_idx:
            ret = self.get_npz(name)
            heavy_len, light_len = len(ret['str_heavy_seq']), len(ret['str_light_seq'])
            if self.max_seq_len is not None:
                if heavy_len + light_len > self.max_seq_len:
                    logger.warn(f'{name} too long. heavy - {len(ret["str_heavy_seq"])}, light - {len(ret["str_light_seq"])}')
            yield ret

    def get_npz(self, name):
        struc = np.load(os.path.join(self.data_dir, name + '.npz'))

        cdr_def = torch.from_numpy(np.concatenate([struc['heavy_cdr_def'], struc['light_cdr_def']], axis=0))

        str_heavy_seq = str(struc['heavy_str_seq'])
        str_light_seq = str(struc['light_str_seq'])

        str_heavy_numbering = str(struc['heavy_numbering'])
        str_light_numbering = str(struc['light_numbering'])

        heavy_seq = torch.tensor(str_seq_to_index(str_heavy_seq), dtype=torch.int64)
        light_seq = torch.tensor(str_seq_to_index(str_light_seq), dtype=torch.int64)

        chain_id = torch.cat([
            torch.zeros((len(str_heavy_seq),), dtype=torch.int32),
            torch.ones((len(str_light_seq),), dtype=torch.int32),], axis=-1)

        ret = dict(name=name,
                str_heavy_seq = str_heavy_seq, str_light_seq=str_light_seq,
                seq = torch.cat([heavy_seq, light_seq], axis=0),
                mask = torch.ones((len(str_heavy_seq) + len(str_light_seq),)),
                heavy_seq = heavy_seq, light_seq = light_seq,
                str_heavy_numbering=str_heavy_numbering, str_light_numbering=str_light_numbering,
                chain_id = chain_id,
                cdr_def = cdr_def
                )
        
        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'mask', 
                'str_heavy_seq', 'str_light_seq',
                'seq', 'heavy_seq', 'light_seq',
                'str_heavy_numbering', 'str_light_numbering',
                'chain_id', 'cdr_def')
        name, mask, str_heavy_seq, str_light_seq, seq, heavy_seq, light_seq, str_heavy_numbering, str_light_numbering, chain_id, cdr_def =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_heavy_len = max(tuple(len(s) for s in str_heavy_seq))
        max_light_len = max(tuple(len(s) for s in str_light_seq))
        max_full_len = max(tuple(len(a) + len(b) for a, b in zip(str_heavy_seq, str_light_seq)))
        padded_seqs = pad_for_batch(seq, max_full_len, 'seq')
        padded_masks = pad_for_batch(mask, max_full_len, 'msk')

        padded_heavy_seqs = pad_for_batch(heavy_seq, max_heavy_len, 'seq')
        padded_light_seqs = pad_for_batch(light_seq, max_light_len, 'seq')

        padded_chain_id = pad_for_batch(chain_id, max_full_len, 'msk')

        padded_region_embed  = pad_for_batch(cdr_def, max_full_len, 'seq')

        ret = dict(
		name=name,
                seq=padded_seqs,
                mask=padded_masks,
                heavy_seq=padded_heavy_seqs,
                light_seq=padded_light_seqs,
                str_heavy_seq=str_heavy_seq,
                str_light_seq=str_light_seq,
                str_heavy_numbering=str_heavy_numbering,
                str_light_numbering=str_light_numbering,
                chain_id=padded_chain_id,
                data_type = 'ig',
                region_embed = padded_region_embed
                )

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

class GeneralSeqDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.name_idx = name_idx
        self.data_dir = data_dir

    def __iter__(self):
        for name in self.name_idx:
            ret = self.get_npz(name)
            yield ret

    def get_npz(self, name):
        struc = np.load(os.path.join(self.data_dir, name + '.npz'))

        str_seq = str(struc['seq'])

        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)

        chain_id = torch.zeros((len(str_seq),), dtype=torch.int32)


        ret = dict(name=name,
                str_seq = str_seq,
                mask = torch.ones((len(str_seq),)),
                seq = seq, 
                chain_id = chain_id,
                )
        
        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'mask', 'str_seq', 'seq','chain_id')

        name, mask, str_seq, seq, chain_id = list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))
        padded_seqs = pad_for_batch(seq, max_len, 'seq')
        padded_masks = pad_for_batch(mask, max_len, 'msk')

        padded_seqs = pad_for_batch(seq, max_len, 'seq')

        padded_chain_id = pad_for_batch(chain_id, max_len, 'msk')

        ret = dict(
		name=name,
                mask=padded_masks,
                seq=padded_seqs,
                str_seq=str_seq,
                chain_id=padded_chain_id,
                region_embed = torch.zeros_like(padded_seqs),
                data_type = 'general')

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def load_test(data_dir, name_idx, feats=None, data_type='ig', rank=None, world_size=1, **kwargs):
    if data_type == 'ig':
        dataset = SeqDataset(data_dir, name_idx, max_seq_len=None, reduce_num=None)
    elif data_type == 'general':
        dataset = GeneralSeqDataset(data_dir, name_idx)
    else:
        raise NotImplementedError('data type {data_type} not implemented.')

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, is_training=False))

    return torch.utils.data.DataLoader(dataset, **kwargs)

