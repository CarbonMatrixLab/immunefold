import os
import functools
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F


from carbonmatrix.common import residue_constants
from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.data.seq import str_seq_to_index, esm_alphabet
from carbonmatrix.data.feature_factory import FeatureFactory

class Cluster(object):
    def __init__(self, names):
        self.names = names
        self.idx = 0
        assert len(names) > 0

    def get_next(self):
        item = self.names[self.idx]
        self.idx += 1
        if self.idx == len(self.names):
            self.idx = 0
        return item

    def __expr__(self):
        return self.names[self.idx]

    def __str__(self):
        return self.names[self.idx]

def parse_cluster(file_name):
    ret = []
    with open(file_name) as f:
        for line in f:
            items = line.strip().split()
            ret.append(Cluster(names=items))
    return ret

class TransformedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, feats, device, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        self.feature_factory = FeatureFactory(feats)
        self.device = device
        
    def set_epoch(self, epoch):
        if self.sampler is not None:
            print('set_epoch')
            self.sampler.set_epoch(epoch)
    
    def __iter__(self,):
        for batch in super().__iter__():
            batch = {k : v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            yield self.feature_factory(batch)

#class SeqDataset(torch.utils.data.IterableDataset):
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, max_seq_len=None):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.alphabet = esm_alphabet 

    def _create_esm_seq(self, str_seq):
        L = len(str_seq)
        seq = np.zeros((L + 2,), dtype=np.int64)
        seq[0] = esm_alphabet.cls_idx
        seq[-1] = esm_alphabet.eos_idx

        for i, a in enumerate(str_seq):
            seq[i+1] = esm_alphabet.get_idx(a)

        return seq

    def _next_item(self,):
        raise NotImplementedError('_get_next_seq not implemented')

    def __get_item(self, idx):
        raise NotImplementedError('_get_next_seq not implemented')

    def _create_seq_data(self, name, str_seq):
        N = len(str_seq)
        return dict(
                name = name,
                str_seq = str_seq,
                seq = torch.from_numpy(str_seq_to_index(str_seq)),
                esm_seq = torch.from_numpy(self._create_esm_seq(str_seq)),
                residx = torch.arange(0, N+2),
                mask = torch.ones((N,), dtype=torch.bool)
                )
    
    def __iter__(self,):
        for name, str_seq in self._next_item():
            yield self._create_seq_data(name, str_seq)
    
    def __getitem__(self, idx):
        name, str_seq = self._get_item(idx)
        return self._create_seq_data(name, str_seq)

def collate_fn_seq(batch):
    # fields = ('name', 'str_seq', 'seq', 'esm_seq', 'residx', 'mask')
    
    def _gather(n):
        return [b[n] for b in batch]
    
    name = _gather('name')
    str_seq = _gather('str_seq')
    max_len = max(tuple(len(s) for s in str_seq))
        
    return dict(
            name=name,
            str_seq = str_seq,
            seq = pad_for_batch(_gather('seq'), max_len, residue_constants.unk_restype_index),
            esm_seq = pad_for_batch(_gather('esm_seq'), max_len + 2, esm_alphabet.padding_idx),
            residx = pad_for_batch(_gather('residx'), max_len + 2, 0),
            mask = pad_for_batch(_gather('mask'), max_len, 0),
            batch_len = max_len, 
            )
