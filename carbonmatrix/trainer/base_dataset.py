import torch

from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.data.base_dataset import SeqDataset, collate_fn_seq

class StructureDataset(SeqDataset):
    def __init__(self, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)
        
    def _create_struc_data(self, item):
        ret = self._create_seq_data(item['name'], item['str_seq'])
        ret.update(
                atom14_gt_positions = torch.from_numpy(item['coords']), 
                atom14_gt_exists =torch.from_numpy(item['coord_mask']),
                )

        return ret
    
    def __getitem__(self, idx):
        item = self._get_item(idx)
        
        return self._create_struc_data(item)

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
