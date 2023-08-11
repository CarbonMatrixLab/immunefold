import os
import functools
import logging
import pathlib
import random

import numpy as np
import torch

from torch.nn import functional as F

from esm.data import Alphabet

from abfold.common import residue_constants
from abfold.model.features import FeatureBuilder
import abfold.trainer.features

from abfold.data.utils import pad_for_batch
from abfold.preprocess.parser import make_stage1_feature_from_pdb
from abfold.common.utils import str_seq_to_index

logger = logging.getLogger(__file__)

class Stage1StructureDataset(torch.utils.data.Dataset):
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

    def __iter__(self):
        name_idx = self._get_name_idx()
        for item in name_idx:
            if self.is_cluster_idx:
                name = item.get_next()
            else:
                name = item
            ret = self.get_structure_label_npz(name)
            heavy_len, light_len = len(ret.get('str_heavy_seq', '')), len(ret.get('str_light_seq', ''))
            if self.max_seq_len is not None:
                if heavy_len + light_len > self.max_seq_len:
                    logger.warn(f'{name} too long. heavy - {heavy_len}, light - {light_len}')
            yield ret

    def __getitem__(self, idx):
        name = self.name_idx[idx]

        return self.get_structure_from_pdb(name)
    
    def _create_train_seq_sample(self, str_seq):
        L = len(str_seq)
        seq = np.zeros((L + 2,), dtype=np.int64)
        label_seq = np.zeros((L + 2,), dtype=np.int64)
        label_mask = np.zeros((L + 2,), dtype=np.float32)
        
        seq[0] = self.alphabet.cls_idx
        label_seq[0] = self.alphabet.cls_idx
        seq[-1] = self.alphabet.eos_idx
        label_seq[-1] = self.alphabet.eos_idx
        
        # 15
        # 80, 10, 10
        ru = np.random.uniform(0., 1., (L,))
        for i, (a, r) in enumerate(zip(str_seq, ru)):
            label_seq[i+1] = self.alphabet.get_idx(a)
            if r < 0.15:
                label_mask[i+1] = 1.0
                if r < 0.12:
                    seq[i+1] = self.alphabet.mask_idx
                elif r < 0.135:
                    r_a = np.random.choice(residue_constants.restypes)
                    seq[i+1] = self.alphabet.get_idx(r_a)
                else:
                    seq[i+1] = label_seq[i+1]
            else:
                seq[i+1] = label_seq[i+1]

        return seq, label_seq, label_mask

    def get_structure_from_pdb(self, name):
        pdb_file = os.path.join(self.data_dir, name + '.pdb')

        struc = make_stage1_feature_from_pdb(pdb_file) 

        str_seq = struc['str_seq']
        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)
        
        esm_seq, label_esm_seq, label_esm_mask = self._create_train_seq_sample(str_seq)

        mask = torch.ones((len(str_seq),))

        ret = dict(name=name,
                str_seq = str_seq,
                mask = mask,
                seq = seq,
                esm_seq = torch.from_numpy(esm_seq),
                label_esm_seq = torch.from_numpy(label_esm_seq),
                label_esm_mask = torch.from_numpy(label_esm_mask),
                residx = torch.from_numpy(struc['residx']),
                atom14_gt_positions = torch.from_numpy(struc['coords']), 
                atom14_gt_exists =torch.from_numpy(struc['coord_mask']),)

        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'str_seq', 'seq', 'mask',
                'atom14_gt_positions', 'atom14_gt_exists', 'residx', 'esm_seq', 'label_esm_seq', 'label_esm_mask')
        name, str_seq, seq, mask, atom14_gt_positions, atom14_gt_exists, residx, esm_seq, label_esm_seq, label_esm_mask =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))

        def _pad(x, pad_len, pad_value):
            return torch.stack([F.pad(xx, [0, pad_len - xx.shape[-1]], value=pad_value) for xx in x], dim=0)
        
        padded_seq = _pad(seq, max_len, self.alphabet.padding_idx)
        padded_esm_seq = _pad(esm_seq, max_len, self.alphabet.padding_idx)
        padded_label_esm_seq = _pad(label_esm_seq, max_len, self.alphabet.padding_idx)
        padded_label_esm_mask = _pad(label_esm_mask, max_len, 0.)

        padded_residx = pad_for_batch(residx, max_len, 'msk')
        padded_masks = pad_for_batch(mask, max_len, 'msk')

        padded_atom14_gt_positions = pad_for_batch(atom14_gt_positions, max_len, 'crd')
        padded_atom14_gt_existss = pad_for_batch(atom14_gt_exists, max_len, 'crd_msk')

        ret = dict(
		name=name,
                str_seq = str_seq,
                seq = padded_seq,
                esm_seq = padded_esm_seq,
                label_esm_seq = padded_label_esm_seq,
                label_esm_mask = padded_label_esm_mask,
                mask = padded_masks,
                residx = padded_residx,
                atom14_gt_positions = padded_atom14_gt_positions,
                atom14_gt_exists = padded_atom14_gt_existss,
                data_type = 'general'
                )

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def load(data_dir, name_idx,
        feats=None, 
        is_training=True,
        max_seq_len=None, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        **kwargs):

    dataset = Stage1StructureDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, **kwargs)
