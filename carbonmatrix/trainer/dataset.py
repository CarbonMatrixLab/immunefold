import os

from carbonmatrix.data.base_dataset import parse_cluster
from carbonmatrix.data.parser import make_feature_from_pdb, make_feature_from_npz
from carbonmatrix.trainer.base_dataset import StructureDataset

class StructureDatasetPDBIO(StructureDataset):
    def __init__(self, data_dir, name_idx_file, max_seq_len):
        super().__init__(max_seq_len=max_seq_len)

        self.data_dir = data_dir
        self.name_idx = parse_cluster(name_idx_file)

class StructureDatasetNpzIO(StructureDataset):
    def __init__(self, data_dir, name_idx_file, max_seq_len):
        super().__init__(max_seq_len=max_seq_len)

        self.data_dir = data_dir
        self.name_idx = parse_cluster(name_idx_file)

    def __len__(self,):
        return len(self.name_idx)

    def _get_item(self, idx):
        c = self.name_idx[idx]
        name = c.get_next()
        file_path = os.path.join(self.data_dir, name + '.npz')
        struc = make_feature_from_npz(file_path)
        struc.update(name = name)

        return struc

class AbStructureDatasetNpzIO(StructureDataset):
    def __init__(self, data_dir, name_idx_file, max_seq_len, shuffle_multimer_seq=False):
        super().__init__(max_seq_len=max_seq_len)

        self.data_dir = data_dir
        self.name_idx = parse_cluster(name_idx_file)

        self.shuffle_multimer_seq = shuffle_multimer_seq

    def __len__(self,):
        return len(self.name_idx)

    def _get_item(self, idx):
        c = self.name_idx[idx]
        name = c.get_next()
        file_path = os.path.join(self.data_dir, name + '.npz')
        struc = make_feature_from_npz(file_path, is_ig_feature=True, shuffle_multimer_seq=self.shuffle_multimer_seq)
        struc.update(name = name)

        return struc
