import logging

import torch
import numpy as np
import random
from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.data.base_dataset import SeqDataset, collate_fn_seq
logger = logging.getLogger()

class StructureDataset(SeqDataset):
    def __init__(self, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)

    def _create_struc_data(self, item):
        ret = self._create_seq_data(item['name'], item['str_seq'])
        
        ret.update(
                atom14_gt_positions = item['coords'],
                atom14_gt_exists = item['coord_mask'],
                )
        if 'chain_id' in item:
            ret.update(chain_id=item['chain_id'])
        return ret

    def _slice_sample(self, item):
        str_len = len(item['str_seq'])
        mhc_flag = 0
        if 'chain_id' in item and 4 in item['chain_id']:
            mhc_flag = 1
            chain_id = item['chain_id']
            mhc_chain = chain_id[chain_id == 4]
            # mhc_id = np.unique(mhc_chain)
            # mhc_id = (mhc_id == 4).nonzero()[0]  
            mhc_len = len(mhc_chain)
            str_len = mhc_len
        
            indices = (chain_id == 4).nonzero()[0]  
            mhc_start = indices[0].item()
            mhc_end = indices[-1].item()
            chains = np.unique(chain_id)
            chain_start = []
            chain_end = []
            for chain in chains:
                if chain != 4:
                    chain_start.append((chain_id == chain).nonzero()[0][0].item())
                    chain_end.append((chain_id == chain).nonzero()[0][-1].item())


            if self.max_seq_len is not None and str_len > self.max_seq_len:
                name = item['name']
                import pdb

                start = np.random.randint(0, str_len - self.max_seq_len)
                end = start + self.max_seq_len

                logger.warn(f'{name} with len= {str_len} to be sliced at postion= {start}')
                chain_start.append(mhc_start+start)
                chain_end.append(mhc_start+end)
                chain_start.sort()
                chain_end.sort()
                for k, v in item.items():
                    if mhc_flag:
                        if k in ['name', 'multimer_str_seq']:
                            continue
                        # if k in ['multimer_str_seq']:
                        #     v[mhc_id] = v[mhc_id][start:end]
                        if type(v) is str:
                            item[k] = v[:mhc_start] + v[mhc_start+start: mhc_start+end]+v[mhc_end+1:]
                            multimer_str_seq = ''
                            for i in range(len(chain_start)):
                                multimer_str_seq += v[chain_start[i]:chain_end[i]+1] + ':'
                            if multimer_str_seq[-1] == ':':
                                multimer_str_seq = multimer_str_seq[:-1]
                            item['multimer_str_seq'] = multimer_str_seq.split(':')
                        else:
                            item[k] = np.concatenate([v[:mhc_start], v[mhc_start+start: mhc_start+end], v[mhc_end+1:]])
                        # pdb.set_trace()

                    else:
                        if k in ['name']:
                            continue
                        item[k] = v[start:end]
        return item

    def __getitem__(self, idx):
        item = self._get_item(idx)

        item = self._create_struc_data(item)

        item = self._slice_sample(item)

        for k, v in item.items():
            # if k in ['multimer_str_seq']:
            #     continue

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

def slice_structure(struc_mask, max_seq_len):
    num_struc = torch.sum(struc_mask)
    if num_struc > 0 and num_struc < str_len:
        struc_start, struc_end = 0, str_len
        while struc_start < str_len and struc_mask[struc_start] == False:
            struc_start += 1
        while struc_end > 0 and struc_mask[struc_end - 1] == False:
            struc_end -= 1
        if struc_end - struc_start > max_seq_len:
            start = np.random.randint(struc_start, struc_end - max_seq_len)
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
