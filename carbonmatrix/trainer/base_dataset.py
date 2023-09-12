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
    
    def __iter__(self,):
        for item in self._next_item():
            yield self._create_struc_data(item)

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
