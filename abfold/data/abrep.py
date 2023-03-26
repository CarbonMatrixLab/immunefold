from argparse import Namespace

import numpy as np

import torch
from einops import rearrange

from bert.model.model import ProteinBertModel as AbRep
from bert.train.schema import Schema

from bert.model.data import Alphabet
import bert.train.config as config

_extractor_dict = {}

class AbRepExtractor(object):
    def __init__(self, model_path):
        super().__init__()

        self.alphabet = Alphabet.from_architecture("roberta_large")
        self.pos_schema = Schema()

        self.model = AbRep(
            Namespace(**config.model_args),
            self.alphabet,
        )

        checkpoints = torch.load(model_path)
        self.model.load_state_dict(checkpoints["net"])


        self.model.eval()

    @staticmethod
    def get(model_path, device=None):
        global _extractor_dict

        name = 'abrep'

        if name not in _extractor_dict:

            obj = AbRepExtractor(model_path)
            if device is not None:
                obj.model.to(device)

            _extractor_dict[name] = obj

        return _extractor_dict[name]

    def rescoding(self, str_seqs, str_numberings, chain_name, device=None, return_attnw=False):
        assert chain_name in ['Heavy', 'Light']

        max_len = max([len(s) for s in str_seqs]) + 2
        bs = len(str_seqs)

        seqs = np.full((bs, max_len), self.alphabet.padding_idx)
        positions = np.full((bs, max_len), 0)

        for i, (s, n_str) in enumerate(zip(str_seqs, str_numberings)):
            n = n_str.split(',') if n_str != '' else []

            seqs[i, 0] = self.alphabet.cls_idx
            positions[i, 0] = 0

            for j, (ss, nn) in enumerate(zip(s, n)):
                seqs[i, j+1] = self.alphabet.get_idx(ss)
                _, positions[i, j+1] = self.pos_schema[(chain_name, nn)]

            seqs[i, 1 + len(s)] = self.alphabet.eos_idx

        if device is not None:
            seqs = torch.tensor(seqs, device=device)
            positions = torch.tensor(positions, device=device)

        with torch.no_grad():
            results = self.model(seqs, positions, repr_layers=[32], need_head_weights=return_attnw)
            single = results["representations"][32][:, 1:-1]

            ret = dict(single=single)

            if return_attnw:
                ret['pair'] = rearrange(results['attentions'][:,:,:,1:-1,1:-1], 'b l d i j -> b i j (l d)')

        return ret
