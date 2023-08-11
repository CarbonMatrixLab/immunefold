import torch
from einops import rearrange

#from esm.pretrained import load_model_and_alphabet_local
from abfold.model.lm.pretrained import load_model_and_alphabet_local

# adapted from https://github.com/facebookresearch/esm
_extractor_dict = {}

class ESMEmbeddingExtractor2:
    def __init__(self, model_path):
        #self.model, self.alphabet = load_model_and_alphabet_local(model_path)
        self.model, _, esm_cfg = load_model_and_alphabet_local(model_path)
        #self.model.requires_grad_(False)
        #self.model.half()

        #self.batch_converter = self.alphabet.get_batch_converter()

    def extract(self, tokens, residx, repr_layers=None):

        results = self.model(tokens, index=residx, repr_layers=repr_layers, need_head_weights=False)

        return results

    @staticmethod
    def get(model_path, device=None):
        global _extractor_dict

        if model_path not in _extractor_dict:
            obj = ESMEmbeddingExtractor2(model_path)
            if device is not None:
                obj.model.to(device=device)
            _extractor_dict[model_path] = obj
        return _extractor_dict[model_path]
