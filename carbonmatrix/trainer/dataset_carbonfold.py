import os
import functools
import logging
import pathlib
import random

import numpy as np
import torch

from torch.nn import functional as F

from esm.data import Alphabet

from carbonmatrix.common import residue_constants
from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.data.feature_factory import FeatureFactory
from carbonmatrix.data.parser import make_stage1_feature_from_pdb
from carbonmatrix.data.seq import str_seq_to_index

logger = logging.getLogger(__file__)

class DistributedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rank, world_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        return len(self.dataset) // self.world_size

    def __getitem__(self, idx):
        return self.dataset[idx * self.world_size + self.rank]

    def collate_fn(self, *args, **kwargs):
        return self.dataset.collate_fn(*args, **kwargs)

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.name_idx = name_idx
        self.max_seq_len = max_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx
        self.alphabet = Alphabet.from_architecture(name='ESM-1b')

        logger.info(f'dataset size= {len(name_idx)} max_seq_len= {max_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')

        self.epoch_count = 0

    def __len__(self):
        return len(self.name_idx)

    def _get_name_idx(self):
        if self.reduce_num is None:
            return self.name_idx

        random.seed(2022 + self.epoch_count)
        random.shuffle(self.name_idx)
        self.epoch_count += 1
        logging.info(f'ig data: epoch_count={self.epoch_count} reduce_num={self.reduce_num} all={len(self.name_idx)} ex={",".join([str(x) for x in self.name_idx[:4]])}')

        return self.name_idx[:self.reduce_num]

    def __getitem__(self, idx):
        name = self.name_idx[idx]

        return self.get_seq_from_fasta(name)

    def _create_esm_seq_sample(self, str_seq):
        L = len(str_seq)
        seq = np.zeros((L + 2,), dtype=np.int64)
        seq[0] = self.alphabet.cls_idx
        seq[-1] = self.alphabet.eos_idx

        for i, a in enumerate(str_seq):
            seq[i+1] = self.alphabet.get_idx(a)

        return seq

    def get_seq_from_fasta(self, name):
        seq_file = os.path.join(self.data_dir, name + '.fasta')

        str_seq = ''
        with open(seq_file) as f:
            for line in f:
                if line.startswith('>'):
                    continue
                str_seq += line.strip()


        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)
        esm_seq = self._create_esm_seq_sample(str_seq)

        N = len(seq)

        residx = np.arange(N)
        residx = np.concatenate([[0], residx + 1, [residx[-1]+2]], axis=-1)

        mask = torch.ones((len(str_seq),))

        ret = dict(name=name,
                str_seq = str_seq,
                mask = mask,
                seq = seq,
                esm_seq = torch.from_numpy(esm_seq),
                residx = torch.from_numpy(residx),
                )

        return ret

    def collate_fn(self, batch, feat_factory=None):
        fields = ('name', 'str_seq', 'seq', 'mask', 'residx', 'esm_seq')
        name, str_seq, seq, mask, residx, esm_seq =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))

        def _pad(x, pad_len, pad_value):
            return torch.stack([F.pad(xx, [0, pad_len - xx.shape[-1]], value=pad_value) for xx in x], dim=0)

        padded_seq = _pad(seq, max_len, self.alphabet.padding_idx)

        padded_esm_seq = _pad(esm_seq, max_len + 2, self.alphabet.padding_idx)
        padded_residx = pad_for_batch(residx, max_len + 2, 'msk')

        padded_masks = pad_for_batch(mask, max_len, 'msk')

        ret = dict(
		name=name,
                str_seq = str_seq,
                seq = padded_seq,
                esm_seq = padded_esm_seq,
                mask = padded_masks,
                residx = padded_residx,
                )

        if feat_factory:
            ret = feat_factory.build(ret)

        return ret

class StructureDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_dir, name_idx,
            device=None, feats=None,
            max_seq_len=None, reduce_num=None, is_cluster_idx=False,):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.name_idx = name_idx
        self.max_seq_len = max_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx
        self.alphabet = Alphabet.from_architecture(name='ESM-1b')

        logger.info(f'dataset size= {len(name_idx)} max_seq_len= {max_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')

        self.epoch_count = 0

    def __len__(self):
        return len(self.name_idx)

    def _get_name_idx(self):
        if self.reduce_num is None:
            return self.name_idx

        random.seed(2022 + self.epoch_count)
        random.shuffle(self.name_idx)
        self.epoch_count += 1
        logging.info(f'ig data: epoch_count={self.epoch_count} reduce_num={self.reduce_num} all={len(self.name_idx)} ex={",".join([str(x) for x in self.name_idx[:4]])}')

        return self.name_idx[:self.reduce_num]

    def __getitem__(self, idx):
        name = self.name_idx[idx]

        return self.get_structure_from_pdb(name)

    def _create_esm_seq_sample(self, str_seq):
        L = len(str_seq)
        seq = np.zeros((L + 2,), dtype=np.int64)
        seq[0] = self.alphabet.cls_idx
        seq[-1] = self.alphabet.eos_idx

        for i, a in enumerate(str_seq):
            seq[i+1] = self.alphabet.get_idx(a)

        return seq

    def get_structure_from_pdb(self, name):
        pdb_file = os.path.join(self.data_dir, name + '.pdb')

        struc = make_stage1_feature_from_pdb(pdb_file)

        str_seq = struc['str_seq']

        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)

        esm_seq = self._create_esm_seq_sample(str_seq)

        residx = struc['residx']
        residx = np.concatenate([[0], residx + 1, [residx[-1]+2]], axis=-1)

        mask = torch.ones((len(str_seq),))

        ret = dict(name=name,
                str_seq = str_seq,
                mask = mask,
                seq = seq,
                esm_seq = torch.from_numpy(esm_seq),
                residx = torch.from_numpy(residx),
                atom14_gt_positions = torch.from_numpy(struc['coords']),
                atom14_gt_exists =torch.from_numpy(struc['coord_mask']),)

        return ret

    def collate_fn(self, batch, feat_factory=None):
        fields = ('name', 'str_seq', 'seq', 'mask',
                'atom14_gt_positions', 'atom14_gt_exists', 'residx', 'esm_seq')
        name, str_seq, seq, mask, atom14_gt_positions, atom14_gt_exists, residx, esm_seq =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))

        def _pad(x, pad_len, pad_value):
            return torch.stack([F.pad(xx, [0, pad_len - xx.shape[-1]], value=pad_value) for xx in x], dim=0)

        padded_seq = _pad(seq, max_len, self.alphabet.padding_idx)

        padded_esm_seq = _pad(esm_seq, max_len + 2, self.alphabet.padding_idx)
        padded_residx = pad_for_batch(residx, max_len + 2, 'msk')

        padded_masks = pad_for_batch(mask, max_len, 'msk')

        padded_atom14_gt_positions = pad_for_batch(atom14_gt_positions, max_len, 'crd')
        padded_atom14_gt_existss = pad_for_batch(atom14_gt_exists, max_len, 'crd_msk')

        ret = dict(
		name=name,
                str_seq = str_seq,
                seq = padded_seq,
                esm_seq = padded_esm_seq,
                mask = padded_masks,
                residx = padded_residx,
                atom14_gt_positions = padded_atom14_gt_positions,
                atom14_gt_exists = padded_atom14_gt_existss,
                )

        if feat_factory:
            ret = feat_factory.build(ret)

        return ret

def load(data_dir, name_idx,
        feats=None,
        is_training=True,
        max_seq_len=None, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        **kwargs):

    if is_training:
        dataset = StructureDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)
    else:
        dataset = SeqDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_factory=FeatureFactory(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, **kwargs)
