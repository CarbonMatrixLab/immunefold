import os
import functools
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F
import logging


from immunefold.common import residue_constants
from immunefold.common.operator import pad_for_batch
from immunefold.data.seq import str_seq_to_index
from immunefold.data.transform_factory import TransformFactory

import pdb

logger = logging.getLogger()


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

        self.transform_factory = TransformFactory(feats)
        self.device = device
        
    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def __iter__(self,):
        for batch in super().__iter__():
            batch = {k : v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            yield self.transform_factory(batch)

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, max_seq_len=None):
        super().__init__()

        self.max_seq_len = max_seq_len

    def __get_item(self, idx):
        raise NotImplementedError('_get_next_seq not implemented')

    def _create_seq_data(self, name, str_seq):
        multimer_str_seq = str_seq.split(':')
        chain_num = len(multimer_str_seq)
        chain_ids = []
        for i in range(chain_num):
            chain_id = np.ones((len(multimer_str_seq[i]),), dtype=np.int32) * i
            chain_ids.append(chain_id)
        chain_ids = np.concatenate(chain_ids, axis=0)
        str_seq = ''.join(multimer_str_seq)

        N = len(str_seq)
        return dict(
                name = name,
                str_seq = str_seq,
                seq = str_seq_to_index(str_seq),
                mask = np.ones((N,), dtype=np.bool_),
                multimer_str_seq = multimer_str_seq,
                chain_id = chain_ids,
                )
    
    def __getitem__(self, idx):
        item = self._get_item(idx)

        ret = self._create_seq_data(item['name'], item['seq'])
        if 'meta' in item:
            ret.update(meta=item['meta'])
        
        for k, v in ret.items():
            ret[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        return ret

def collate_fn_seq(batch):
    def _gather(n):
        return [b[n] for b in batch]
    
    name = _gather('name')
    str_seq = _gather('str_seq')
    multimer_str_seq = _gather('multimer_str_seq')
    
    meta = {} if 'meta' not in batch[0].keys() else _gather('meta')

    max_len = max(tuple(len(s) for s in str_seq))
    feature = dict(
            name=name,
            str_seq = str_seq,
            multimer_str_seq = multimer_str_seq,
            seq = pad_for_batch(_gather('seq'), max_len, residue_constants.unk_restype_index),
            mask = pad_for_batch(_gather('mask'), max_len, 0),
            batch_len = max_len, 
            meta=meta,
            )
    if 'chain_id' in batch[0].keys():
        feature.update(
            {'chain_id': pad_for_batch(_gather('chain_id'), max_len, 0)}
        )
    return feature





class StructureDataset(SeqDataset):
    def __init__(self, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)

    def _create_struc_data(self, item):
        # pdb.set_trace()
        ret = self._create_seq_data(item['name'], item['str_seq'])
        
        ret.update(
                atom14_gt_positions = item['coords'],
                atom14_gt_exists = item['coord_mask'],
                )
        if 'chain_id' in item:
            ret.update(chain_id=item['chain_id'])
        if 'antigen_contact_idx' in item:
            ret.update(antigen_contact_idx=item['antigen_contact_idx'])
        return ret

    def _slice_sample(self, item):
        str_len = len(item['str_seq'])
        receptor_flag = 0
        if 'chain_id' in item and 4 in item['chain_id']:
            receptor_flag = 1
            chain_id = item['chain_id']
            receptor_chain = chain_id[chain_id == 4]
            # receptor_id = np.unique(receptor_chain)
            # receptor_id = (receptor_id == 4).nonzero()[0]  
            receptor_len = len(receptor_chain)
            str_len = receptor_len
        
            indices = (chain_id == 4).nonzero()[0]  
            receptor_start = indices[0].item()
            receptor_end = indices[-1].item()
            chains = np.unique(chain_id)
            chain_start = []
            chain_end = []
            for chain in chains:
                if chain != 4:
                    chain_start.append((chain_id == chain).nonzero()[0][0].item())
                    chain_end.append((chain_id == chain).nonzero()[0][-1].item())


            if self.max_seq_len is not None and str_len > self.max_seq_len:
                name = item['name']
                if 'antigen_contact_idx' not in item:
                    start = np.random.randint(0, str_len - self.max_seq_len)
                    end = start + self.max_seq_len
                else:
                    antigen_contact_idx = item['antigen_contact_idx']
                    select_idx = random.randint(0, len(antigen_contact_idx)-1)
                    contact_idx = antigen_contact_idx[select_idx]
                    start = max(contact_idx - self.max_seq_len // 2, 0)
                    end = min(start + self.max_seq_len//2, str_len-1)

                logger.warn(f'{name} with len= {str_len} to be sliced at postion= {start}')
                chain_start.append(receptor_start+start)
                chain_end.append(receptor_start+end)
                chain_start.sort()
                chain_end.sort()
                # import pdb
                # pdb.set_trace()
            
                for k, v in item.items():
                    if receptor_flag:
                        if k in ['name', 'multimer_str_seq']:
                            continue
                        if type(v) is str:
                            item[k] = v[:receptor_start] + v[receptor_start+start: receptor_start+end+1]+v[receptor_end+1:]
                            multimer_str_seq = ''
                            for i in range(len(chain_start)):
                                multimer_str_seq += v[chain_start[i]:chain_end[i]+1] + ':'
                            if multimer_str_seq[-1] == ':':
                                multimer_str_seq = multimer_str_seq[:-1]
                            item['multimer_str_seq'] = multimer_str_seq.split(':')
                            # import pdb
                            # pdb.set_trace()
                        else:
                            item[k] = np.concatenate([v[:receptor_start], v[receptor_start+start: receptor_start+end+1], v[receptor_end+1:]])
            # import pdb 
            # pdb.set_trace()
            return item
        elif 'chain_id' in item and 4 not in item['chain_id']:
            return item
        elif 'chain_id' not in item:
            if self.max_seq_len is not None and str_len > self.max_seq_len:
                start = np.random.randint(0, str_len - self.max_seq_len)
                end = start + self.max_seq_len
                for k, v in item.items():
                    if k in ['name']:
                        continue
                    item[k] = v[start:end]

            return item

    def __getitem__(self, idx):
        item = self._get_item(idx)

        item = self._create_struc_data(item)

        # item = self._slice_sample(item)

        for k, v in item.items():
            item[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        
        return item

def collate_fn_struc(batch):
    def _gather(n):
        return [b[n] for b in batch]

    ret = collate_fn_seq(batch)
    max_len = ret['batch_len']

    ret.update(
        atom14_gt_positions = pad_for_batch(_gather('atom14_gt_positions'), max_len, 0.),
        atom14_gt_exists = pad_for_batch(_gather('atom14_gt_exists'), max_len, 0),
        )

    return ret
