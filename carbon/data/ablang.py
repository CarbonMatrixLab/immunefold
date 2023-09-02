import torch

from ablang.pretrained import pretrained as Pretrained

# embedding related constants

AbLang_EMBED_DIM = 768

_extractor_dict = {}

class AbLangExtractor(Pretrained):
    def __init__(self, chain, model_folder, device):
        super(AbLangExtractor, self).__init__(chain=chain, model_folder=model_folder, device=device)


    @staticmethod
    def get(chain, model_path, device=None):
        global _extractor_dict

        if chain not in _extractor_dict:
            obj = AbLangExtractor(chain=chain, model_folder=model_path, device=device)
            _extractor_dict[chain] = obj
        return _extractor_dict[chain]
    
    def rescoding(self, seqs, align=False):
        seqs = [''.join(['*' if s in 'BOJUZX' else s for s in seq]) for seq in seqs]

        with torch.no_grad():
            tokens = self.tokenizer(seqs, pad=True, device=self.used_device)
            residue_states = self.AbRep(tokens).last_hidden_states

        return residue_states[:, 1:-1]
