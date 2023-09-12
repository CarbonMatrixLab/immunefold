import os
import functools
import logging
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from carbonfold.common import residue_constants
from carbonfold.data.utils import pad_for_batch

logger = logging.getLogger(__file__)

class SeqDatsetFastaIO(SeqDataset):
    def __init__(self, fasta_file, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)
        
        self.fasta_file = fasta_file
    
    def _next_item(self,):
        data = []
        with open(self.fasta_file, 'r') as fr:
            seq = None
            for line in fr:
                if line.startswith('>') and seq is not one:
                    name = line[1:].strip().split()[0]
                    data.append((name, seq))
                else:
                    seq = line.strip()
           if seq is not None:
               data.append((name, seq))

        for x in data:
            yield x

class SeqDatsetDirIO(SeqDataset):
    def __init__(self, fasta_dir, name_idx_file, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)
        self.fasta_dir = fasta_dir
        self.name_idx = parse_cluster(name_idx_file)

    def _next_item(self,):
        for c in self.name_idx:
            name = c.get_next()
            file_path = os.path.join(self.fasta_dir, name + '.fasta')
            with open(file_path) as fr:
                head = fr.readline()
                seq = fr.readline().strip()
            yield (name, seq)
