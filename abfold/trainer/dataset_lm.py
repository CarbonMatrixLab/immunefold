import logging
import functools

import numpy as np
import torch
from torch.nn import functional as F

from esm.data import Alphabet

from abfold.trainer.dataset import DistributedDataset
from abfold.common import residue_constants
from abfold.model.features import FeatureBuilder

logger = logging.getLogger(__file__)


class LMDataset(torch.utils.data.IterableDataset):

    def __init__(self, fasta_file, chain_pad_num = 28, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__()

        self.fasta_file = fasta_file
        self.max_seq_len = max_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx
       
        self.chain_pad_num = chain_pad_num
        self.alphabet = Alphabet.from_architecture(name='ESM-1b')

        #logger.info(f'data dir= {self.data_dir} examples= {self.name_idx[:10]}')
        logger.info(f'max_seq_len= {max_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')

        self.epoch_count = 0

    def __iter__(self):
        max_seq_len = self.max_seq_len

        with open(self.fasta_file) as f:
            for line in f:
                ret, cdrh3_len = self.parse_seq_line(line)
                if cdrh3_len <= self.max_seq_len and cdrh3_len > 0:
                    yield ret
    
    def _create_train_sample(self, str_seq):
        L = len(str_seq)
        seq = np.zeros((L,), dtype=np.int64)
        label_seq = np.zeros((L,), dtype=np.int64)
        label_mask = np.zeros((L,), dtype=np.float32)
        # 15
        # 80, 10, 10
        ru = np.random.uniform(0., 1., (L,))
        for i, (a, r) in enumerate(zip(str_seq, ru)):
            label_seq[i] = self.alphabet.get_idx(a)
            if r < 0.15:
                label_mask[i] = 1.0
                if r < 0.12:
                    seq[i] = self.alphabet.mask_idx
                elif r < 0.135:
                    r_a = np.random.choice(residue_constants.restypes)
                    seq[i] = self.alphabet.get_idx(r_a)
                else:
                    seq[i] = label_seq[i]
            else:
                seq[i] = label_seq[i]

        return seq, label_seq, label_mask

    def parse_seq_line(self, line):

        if self.is_cluster_idx:
            content = np.random.choice(line.strip().split())
        else:
            content = line

        items = content.strip().split(':')

        chunk_id, heavy_contig, light_config, heavy_str_seq, light_str_seq, heavy_seq_numbering, light_seq_numbering = items

        seq, label_seq, label_mask = self._create_train_sample(heavy_str_seq)
        str_seq = heavy_str_seq

        def _cdrh3_len(x):
            b, e = 105, 117
            cnt = 0
            for it in x.split(','):
                n = int(it) if it[-1].isnumeric() else int(it[:-1])
                cnt += (n >= b and n <= e)

            return cnt

        is_multi = np.random.uniform() > 0.2
        if is_multi:
            l_seq, l_label_seq, l_label_mask = self._create_train_sample(light_str_seq)

            seq = np.concatenate([seq, np.full((self.chain_pad_num,), self.alphabet.padding_idx, dtype=np.int64), l_seq], axis=-1)
            label_seq = np.concatenate([label_seq, np.full((self.chain_pad_num,), self.alphabet.padding_idx, dtype=np.int64), l_label_seq], axis=-1)
            label_mask = np.concatenate([label_mask, np.zeros((self.chain_pad_num,)), l_label_mask], axis=-1)
            str_seq += 'G' * self.chain_pad_num + light_str_seq

        seq = np.concatenate([[self.alphabet.cls_idx], seq, [self.alphabet.eos_idx]], axis=-1)
        label_seq = np.concatenate([[self.alphabet.cls_idx], label_seq, [self.alphabet.eos_idx]], axis=-1)
        label_mask = np.concatenate([[0.], label_mask, [0.]], axis=-1)

        ret = dict(
                str_seq = str_seq,
                seq = torch.tensor(seq),
                label_seq = torch.tensor(label_seq),
                label_mask = torch.tensor(label_mask),
                )

        return ret, _cdrh3_len(heavy_seq_numbering)

    def collate_fn(self, batch, feat_builder=None):
        fields = ('str_seq', 'seq', 'label_seq', 'label_mask')
        str_seq, seq, label_seq, label_mask = list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))

        def _pad(x, pad_len, pad_value):
            return torch.stack([F.pad(xx, [0, pad_len - xx.shape[-1]], value=pad_value) for xx in x], dim=0)
        
        padded_seq = _pad(seq, max_len, self.alphabet.padding_idx)
        padded_label_seq = _pad(label_seq, max_len, self.alphabet.padding_idx)
        padded_label_mask = _pad(label_mask, max_len, 0.)

        ret = dict(
                str_seq = str_seq,
                seq = padded_seq,
                label_seq = padded_label_seq,
                label_mask = padded_label_mask,
                )

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def load(fasta_file, feats=None, is_training=True, max_seq_len=None, reduce_num=None, rank=None, world_size=1, is_cluster_idx=False, **kwargs):

    dataset = LMDataset(fasta_file, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] = functools.partial(dataset.collate_fn,
            feat_builder = FeatureBuilder(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, **kwargs)
