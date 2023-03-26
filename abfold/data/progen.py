from argparse import Namespace

import numpy as np

import torch
from einops import rearrange

from progen.utils import create_model, create_tokenizer_custom

_extractor_dict = {}

class ProGenExtractor(object):
    def __init__(self, model_file, token_file):
        super().__init__()

        self.model = create_model(model_file, fp16=True)
        self.tokenizer = create_tokenizer_custom(token_file)
        
        self.model.eval()

    @staticmethod
    def get(model_path, token_path, device=None):
        global _extractor_dict

        name = 'progen'

        if name not in _extractor_dict:

            obj = ProGenExtractor(model_path, token_path)
            if device is not None:
                obj.model.to(device)

            _extractor_dict[name] = obj

        return _extractor_dict[name]

    def extract(self, str_seqs, device=None, return_attnw=False):
        def embed(tokens):
            with torch.no_grad():
                target = torch.tensor(self.tokenizer.encode(tokens).ids).to(device)
                ret = self.model(target, labels=target)
                
                hidden_states = ret.hidden_states[:, 1:-1]
                if return_attnw:
                    attentions = rearrange(ret.attentions, 'b h i j -> b i j h')[:,1:-1,1:-1]
                    attentions = (attentions + rearrange(attentions, 'b i j h -> b j i h')) * 0.5
                    return hidden_states, attentions
                else:
                    return hidden_states, None

        
        tokens = '1' + str_seqs[0] + '2'

        hidden_states, attentions = embed(tokens)

        r_hidden_states, r_attentions = embed(tokens[::-1])
        r_hidden_states = torch.flip(r_hidden_states, dims=[1])
        if return_attnw:
            r_attentions = torch.flip(r_attentions, dims=[1,2])
            attentions = torch.cat([attentions, r_attentions], dim=-1)
            ret = dict(single=(hidden_states, r_hidden_states), pair=attentions)
        else:
            ret = dict(single=(hidden_states, r_hidden_states))
        
        return ret
