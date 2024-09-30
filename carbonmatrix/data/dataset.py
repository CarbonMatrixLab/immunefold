import os

from carbonmatrix.common import residue_constants
from carbonmatrix.common.operator import pad_for_batch
from carbonmatrix.data.base_dataset import SeqDataset, StructureDataset
from carbonmatrix.data.base_dataset import parse_cluster
from carbonmatrix.data.parser import make_feature_from_pdb

import pdb

class SeqDatasetFastaIO(SeqDataset):
    def __init__(self, fasta_file, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)

        data = []
        with open(fasta_file, 'r') as fr:
            name = None
            for line in fr:
                if line.startswith('>'):
                    if name is not None:
                        data.append((name, seq))
                    name = line[1:].strip().split()[0]
                else:
                    seq = line.strip()
            if name is not None:
                data.append((name, seq))
        
        self.data = data

    def __len__(self,):
        return len(self.data)

    def _get_item(self, idx):
        (name, seq) = self.data[idx]

        return dict(name=name, seq=seq)

class SeqDatasetDirIO(SeqDataset):
    def __init__(self, fasta_dir, name_idx_file, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)
        self.fasta_dir = fasta_dir
        self.name_idx = parse_cluster(name_idx_file)

    def __len__(self,):
        return len(self.name_idx)

    def _get_item(self, idx):
        c = self.name_idx[idx]
        name = c.get_next()

        file_path = os.path.join(self.fasta_dir, name + '.fasta')

        with open(file_path) as fr:
            head = fr.readline()
            seq = fr.readline().strip()

        return dict(name=name, seq=seq)

class WeightedSeqDatasetFastaIO(SeqDataset):
    def __init__(self, fasta_file, max_seq_len=None):
        super().__init__(max_seq_len=max_seq_len)

        data = []
        with open(fasta_file, 'r') as fr:
            name = None
            for line in fr:
                if line.startswith('>'):
                    if name is not None:
                        data.append((name, seq, weight))
                    name, weight = line[1:].strip().split()[:2]
                else:
                    seq = line.strip()
            if name is not None:
                data.append((name, seq, weight))

        self.data = data

    def __len__(self,):
        return len(self.data)

    def _get_item(self, idx):
        (name, seq, weight) = self.data[idx]

        return dict(name=name, seq=seq, meta={'weight': float(weight)})

class AbStructureDataNpzIO(StructureDataset):
    def __init__(self, fasta_file, ag_pdb, contact_idx, ig_type='ab', shuffle_multimer_seq=False):
        super().__init__(max_seq_len=128)

        self.fasta = fasta_file
        data = []
        with open(fasta_file, 'r') as fr:
            name = None
            for line in fr:
                if line.startswith('>'):
                    if name is not None:
                        data.append((name, seq))
                    name = line[1:].strip().split()[0]
                else:
                    seq = line.strip()
        self.ag_pdb = ag_pdb
        # pdb.set_trace()
        self.feat = make_feature_from_pdb(seq, ag_pdb, contact_idx)
        self.feat.update(name=name)
        data.append(self.feat)
        self.data = data

    def __len__(self,):
        return 1

    def _get_item(self, idx):
        
        
        return self.data[idx]

