from argparse import Namespace

import numpy as np

import torch
from einops import rearrange

from progpt.model import ProGPT
from progpt.data.dataset import create_batch

_extractor_dict = {}

class ProGPTExtractor(object):
    def __init__(self, model_file):
        super().__init__()

        checkpoint = torch.load(model_file)

        model_config, model_state_dict = checkpoint['model_config'], checkpoint['model']

        model = ProGPT(model_config)
        model.load_state_dict(model_state_dict)
    
        model.eval()

        self.model = model

    @staticmethod
    def get(model_path, device=None):
        global _extractor_dict

        name = 'progpt'

        if name not in _extractor_dict:

            obj = ProGPTExtractor(model_path)
            if device is not None:
                obj.model.to(device)

            _extractor_dict[name] = obj

        return _extractor_dict[name]

    def extract(self, str_seqs, device=None, return_attnw=False):
        with torch.no_grad():
            seq, attn_mask = create_batch(str_seqs)

            seq = seq.to(device=device)
            attn_mask = attn_mask.to(device=device)

            results = self.model(seq, attn_mask, return_attn_weights=return_attnw)
            single = results['seq_repr'][:,1:]

            ret = dict(single=single)

            if return_attnw:
                pair = results['attns'][:,1:,1:]
                ret['pair'] = (pair + rearrange(pair, 'b i j c -> b j i c')) / 2.0

        return ret
