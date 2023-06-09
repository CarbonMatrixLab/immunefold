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
from abfold.common.utils import str_seq_to_index as AF_str_seq_to_index
from abfold.model.features import FeatureBuilder
import abfold.trainer.features

from abfold.data.utils import pad_for_batch

logger = logging.getLogger(__file__)

def str_seq_to_index(x):
    return AF_str_seq_to_index(x)

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


class StructureDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.name_idx = name_idx
        self.max_seq_len = max_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx

        #logger.info(f'data dir= {self.data_dir} examples= {self.name_idx[:10]}')
        logger.info(f'dataset size= {len(name_idx)} max_seq_len= {max_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')

        self.epoch_count = 0

    def __len__(self):
        return len(self.name_idx)

class IgStructureDataset(StructureDataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)
    
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

    def get_structure_label_npz(self, name):
        num_atoms = 14
        
        struc = np.load(os.path.join(self.data_dir, name + '.npz'))
        coords = torch.from_numpy(np.concatenate([
            struc.get('heavy_coords', np.zeros((0, num_atoms, 3), dtype=np.float32)),
            struc.get('light_coords', np.zeros((0, num_atoms, 3), dtype=np.float32))], axis=0))
        coord_mask = torch.from_numpy(np.concatenate([
            struc.get('heavy_coord_mask', np.zeros((0, num_atoms), dtype=np.bool_)),
            struc.get('light_coord_mask', np.zeros((0, num_atoms), dtype=np.bool_))], axis=0))

        cdr_def = torch.from_numpy(np.concatenate([
            struc.get('heavy_cdr_def', np.zeros((0,), dtype=np.int32)),
            struc.get('light_cdr_def', np.zeros((0,), dtype=np.int32))], axis=0))
        region_embed = cdr_def + 1

        str_heavy_seq = str(struc.get('heavy_str_seq', ''))
        str_light_seq = str(struc.get('light_str_seq', ''))

        str_heavy_numbering = str(struc.get('heavy_numbering', ''))
        str_light_numbering = str(struc.get('light_numbering', ''))
        
        aa_mapping = residue_constants.restype_order_with_x

        heavy_seq = torch.tensor(str_seq_to_index(str_heavy_seq), dtype=torch.int64)
        light_seq = torch.tensor(str_seq_to_index(str_light_seq), dtype=torch.int64)

        chain_id = torch.cat([
            torch.zeros(len(str_heavy_seq), dtype=torch.int32),
            torch.ones(len(str_light_seq), dtype=torch.int32)], axis=-1)
                
        mask = torch.cat([
            torch.ones(len(str_heavy_seq), dtype=torch.bool), 
            torch.ones(len(str_light_seq), dtype=torch.bool)], dim=-1)

        seq = torch.cat([heavy_seq, light_seq], dim=-1)

        ret = dict(name=name,
                seq=seq,
                mask = mask,
                str_heavy_seq = str_heavy_seq, str_light_seq=str_light_seq,
                heavy_seq = heavy_seq, light_seq = light_seq,
                atom14_gt_positions=coords, atom14_gt_exists=coord_mask,
                str_heavy_numbering=str_heavy_numbering, str_light_numbering=str_light_numbering,
                cdr_def = cdr_def,
                region_embed = region_embed,
                chain_id = chain_id,)
        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'mask', 'str_heavy_seq', 'str_light_seq',
                'seq', 'heavy_seq', 'light_seq',
                'atom14_gt_positions', 'atom14_gt_exists', 'str_heavy_numbering', 'str_light_numbering', 'cdr_def', 'region_embed', 'chain_id')
        name, mask, str_heavy_seq, str_light_seq, seq, heavy_seq, light_seq, atom14_gt_positions, atom14_gt_exists,\
                str_heavy_numbering, str_light_numbering, cdr_def, region_embed, chain_id =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_heavy_len = max(tuple(len(s) for s in str_heavy_seq))
        max_light_len = max(tuple(len(s) for s in str_light_seq))
        max_full_len = max(tuple(len(a) + len(b) for a, b in zip(str_heavy_seq, str_light_seq)))

        padded_seqs = pad_for_batch(seq, max_full_len, 'seq')
        padded_masks = pad_for_batch(mask, max_full_len, 'msk')

        padded_heavy_seqs = pad_for_batch(heavy_seq, max_heavy_len, 'seq')
        padded_light_seqs = pad_for_batch(light_seq, max_light_len, 'seq')

        padded_atom14_gt_positions = pad_for_batch(atom14_gt_positions, max_full_len, 'crd')
        padded_atom14_gt_existss = pad_for_batch(atom14_gt_exists, max_full_len, 'crd_msk')

        padded_cdr_def = pad_for_batch(cdr_def, max_full_len, 'msk')
        padded_chain_id = pad_for_batch(chain_id, max_full_len, 'msk')
        padded_region_embed  = pad_for_batch(region_embed, max_full_len, 'msk')

        ret = dict(
		name=name,
                seq=padded_seqs,
                mask=padded_masks,
                heavy_seq=padded_heavy_seqs,
                light_seq=padded_light_seqs,
                str_heavy_seq=str_heavy_seq,
                str_light_seq=str_light_seq,
                atom14_gt_positions=padded_atom14_gt_positions,
                atom14_gt_exists=padded_atom14_gt_existss,
                str_heavy_numbering=str_heavy_numbering,
                str_light_numbering=str_light_numbering,
                cdr_def=padded_cdr_def,
                chain_id=padded_chain_id,
                region_embed = padded_region_embed,
                data_type = 'ig'
                )

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def sample_with_struc(struc_mask, str_len, max_seq_len):
    num_struc = torch.sum(struc_mask)
    if num_struc > 0 and num_struc < str_len:
        struc_start, struc_end = 0, str_len
        while struc_start < str_len and struc_mask[struc_start] == False:
            struc_start += 1
        while struc_end > 0 and struc_mask[struc_end - 1] == False:
            struc_end -= 1
        if struc_end - struc_start > max_seq_len:
            start = random.randint(struc_start, struc_end - max_seq_len)
            end = start + max_seq_len
        else:
            extra = max_seq_len - (struc_end - struc_start)
            left_extra = struc_start - extra // 2 - 10
            right_extra = struc_end + extra // 2 + 10
            start = random.randint(left_extra, right_extra)
            end = start + max_seq_len
            if start < 0:
                start = 0
                end = start + max_seq_len
            elif end > str_len:
                end = str_len
                start = end - max_seq_len
    else:
        start = random.randint(0, str_len - max_seq_len)
        end = start + max_seq_len
    return start, end

class GeneralStructureDataset(StructureDataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)
        
    
    def _get_name_idx(self):
        if self.reduce_num is None:
            return self.name_idx
        
        if self.epoch_count == 0:
            random.seed(2022 + self.epoch_count)
            random.shuffle(self.name_idx)

        start = self.reduce_num * self.epoch_count
        end = start + self.reduce_num
        
        if end > len(self.name_idx):
            start = 0
            end = self.reduce_num
            random.seed(2022 + self.epoch_count)
            random.shuffle(self.name_idx)
            
        logging.info(f'general data: epoch_count={self.epoch_count} reduce_num={self.reduce_num} all={len(self.name_idx)} start={start} end={end} ex={",".join([str(x) for x in self.name_idx[:4]])}')
        
        self.epoch_count += 1
        
        return self.name_idx[start:end]

    def __iter__(self):
        name_idx = self._get_name_idx()
        
        max_seq_len = self.max_seq_len

        for item in name_idx:
            if self.is_cluster_idx:
                name = item.get_next()
            else:
                name = item
            ret = self.get_structure_label_npz(name)
            if max_seq_len is not None:
                str_len  = len(ret['str_seq'])
                if str_len > max_seq_len:
                    #start = random.randint(0, str_len - max_seq_len)
                    #end = start + max_seq_len
                    start, end = sample_with_struc(ret['atom14_gt_exists'][:,1], str_len, max_seq_len)
                    for k, v in ret.items():
                        if k in ['name']:
                            continue
                        ret[k] = v[start:end]
                    logger.warn(f'{name} with len= {str_len} to be sliced at postion= {start}')
            yield ret

    def get_structure_label_npz(self, name):
        struc = np.load(os.path.join(self.data_dir, name + '.npz'))
        
        coords = torch.from_numpy(struc['coords'])
        coord_mask = torch.from_numpy(struc['coord_mask'])

        str_seq = str(struc['seq'])

        assert len(str_seq) == coords.shape[0] and len(str_seq) == coord_mask.shape[0] and len(str_seq) > 0

        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)

        mask = torch.ones((len(str_seq),))

        chain_id = torch.zeros((len(str_seq),), dtype=torch.int32)

        ret = dict(name=name,
                str_seq = str_seq,
                seq = seq,
                mask = mask,
                atom14_gt_positions=coords, atom14_gt_exists=coord_mask,
                chain_id = chain_id)

        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'str_seq', 'seq', 'mask',
                'atom14_gt_positions', 'atom14_gt_exists',
                'chain_id')
        name, str_seq, seq, mask, atom14_gt_positions, atom14_gt_exists, chain_id =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))
        padded_seqs = pad_for_batch(seq, max_len, 'seq')
        padded_masks = pad_for_batch(mask, max_len, 'msk')

        padded_seqs = pad_for_batch(seq, max_len, 'seq')

        padded_atom14_gt_positions = pad_for_batch(atom14_gt_positions, max_len, 'crd')
        padded_atom14_gt_existss = pad_for_batch(atom14_gt_exists, max_len, 'crd_msk')

        padded_chain_id = pad_for_batch(chain_id, max_len, 'msk')

        ret = dict(
		name=name,
                str_seq=str_seq,
                seq=padded_seqs,
                mask=padded_masks,
                atom14_gt_positions=padded_atom14_gt_positions,
                atom14_gt_exists=padded_atom14_gt_existss,
                chain_id=padded_chain_id,
                cdr_def=torch.ones_like(padded_seqs) * 14,
                region_embed = torch.zeros_like(padded_seqs),
                data_type = 'general'
                )

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def load(data_dir, name_idx,
        feats=None, data_type='ig', 
        is_training=True,
        max_seq_len=None, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        **kwargs):

    if data_type == 'ig':
        dataset = IgStructureDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)
    elif data_type == 'general':
        dataset = GeneralStructureDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)
    else:
        raise NotImplementedError('data type {data_type} not implemented.')

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, **kwargs)
