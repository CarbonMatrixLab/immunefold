import torch

from esm.pretrained import load_model_and_alphabet_local
from abfold.utils import default,exists
from einops import rearrange

ESM_EMBED_LAYER = 33
ESM_EMBED_DIM = 1280

# adapted from https://github.com/facebookresearch/esm
_extractor_dict = {}

class ESMEmbeddingExtractor:
    def __init__(self, model_path):
        self.model, alphabet = load_model_and_alphabet_local(model_path)
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()


    def extract(self, label_seqs, repr_layer=None, return_attnw=False, device=None):
        device = default(device, getattr(label_seqs, 'device', None))

        max_len = max([len(s) for l, s in label_seqs])

        if repr_layer is None:
            repr_layer = ESM_EMBED_LAYER

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter(label_seqs)
            batch_tokens = batch_tokens.to(device=device)

            results = self.model(batch_tokens, repr_layers=[repr_layer], need_head_weights=return_attnw)
            
            single = results['representations'][repr_layer][:,1 : 1 + max_len]
            ret = dict(single=single)

            if return_attnw:
                atten = rearrange(results['attentions'][:, :, :, 1:1+max_len, 1:1+max_len], 'b l d i j -> b i j (l d)')
                ret['pair'] = (atten + rearrange(atten, 'b i j h -> b j i h')) * 0.5
        
        return ret

    @staticmethod
    def get(model_path, device=None):
        global _extractor_dict

        if model_path not in _extractor_dict:
            obj = ESMEmbeddingExtractor(model_path)
            print('build esm')
            if exists(device):
                obj.model.to(device=device)
            _extractor_dict[model_path] = obj
        return _extractor_dict[model_path]
