from argparse import Namespace

import numpy as np

import torch

from bert.train.schema import Schema

_template_obj = {}

class Template(object):
    def __init__(self, template_feats_path):
        super().__init__()

        self.pos_schema = Schema()
        
        data = np.load(template_feats_path)
        self.distogram, self.distogram_mask = data['distogram'], data['distogram_mask']
        self.distogram_dim = self.distogram.shape[-1]
        
        self.torsion, self.torsion_mask = data['torsion'], data['torsion_mask']
        self.torsion_dim = self.torsion.shape[-1]

    @staticmethod
    def get(template_feats_path):
        global _template_obj

        name = 'obj'
        
        if name not in _template_obj:
            obj = Template(template_feats_path)
            _template_obj[name] = obj

        return _template_obj[name]

    def make_template_feats(self, str_heavy_imgt_number, str_light_imgt_number, device=None):
        h_imgt_number = [h.split(',') if h != '' else [] for h in str_heavy_imgt_number]
        l_imgt_number = [l.split(',') if l != '' else [] for l in str_light_imgt_number]
        
        max_len = max([len(h) + len(l) for h, l in zip(h_imgt_number, l_imgt_number)])
        bs = len(h_imgt_number)

        distogram = np.zeros((bs, max_len, max_len, self.distogram_dim))
        #distogram_mask = np.zeros((bs, max_len, max_len), dtype=np.bool_)
        
        torsion = np.zeros((bs, max_len, 3, self.torsion_dim))
        #torsion_mask = np.zeros((bs, max_len, 3, self.torsion_dim), dtype=np.bool_)

        for i, (h, l) in enumerate(zip(h_imgt_number, l_imgt_number)):
            chain_id = ['Heavy'] * len(h) + ['Light'] * len(l)
            imgt_number = h + l
            n = len(imgt_number)
            positions = np.array([self.pos_schema[(a,m)][1] for (a,m) in zip(chain_id, imgt_number)])
            
            distogram[i, :n, :n] = self.distogram[positions][:, positions]
            #distogram_mask[i, :n, :n] = self.distogram_mask[positions][:, positions]
            torsion[i, :n] = self.torsion[positions]

        if device is not None:
            distogram = torch.tensor(distogram, device=device, dtype=torch.float32)
            torsion = torch.tensor(torsion, device=device, dtype=torch.float32)
            #distogram_mask = torch.tensor(distogram_mask, device=device, dtype=torch.float32)

        return distogram, torsion 
