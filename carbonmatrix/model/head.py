from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.common import residue_constants
from carbonmatrix.model import atom as functional, folding
from carbonmatrix.model.common_modules import(
        Linear,
        LayerNorm)
from carbonmatrix.model.common_modules import get_lora_config
from carbonmatrix.model.utils import squared_difference
from carbonmatrix.common.metrics import contact_precision, kabsch_torch, TMscore
from carbonmatrix.model.head_factory import registry_head

@registry_head(name='distogram')
class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, config, num_in_pair_channel):
        super().__init__()

        c = config
        lora_config = get_lora_config(c)

        self.breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1)
        self.proj = Linear(num_in_pair_channel, c.num_bins, init='final', **lora_config)

        self.config = config

    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.proj(x)
        logits = (x + rearrange(x, 'b i j c -> b j i c')) * 0.5
        breaks = self.breaks.to(logits.device)
        return dict(logits=logits, breaks=breaks)

@registry_head(name='structure_module')
class FoldingHead(nn.Module):
    """Head to predict 3d struct.
    """
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        self.struct_module = folding.StructureModule(config, num_in_seq_channel, num_in_pair_channel)

        self.config = config

    def forward(self, headers, representations, batch):
        return self.struct_module(representations, batch)

@registry_head(name='predicted_lddt')
class PredictedLDDTHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        dim = c.num_channel

        self.net = nn.Sequential(
                LayerNorm(c.structure_module_num_channel),
                Linear(c.structure_module_num_channel, dim, init='relu', bias=True),
                nn.ReLU(),
                Linear(dim, dim, init='relu', bias=True),
                nn.ReLU(),
                Linear(dim, c.num_bins, init='final', bias=True))
        self.config = config

    def forward(self, headers, representations, batch):
        assert 'structure_module' in headers

        act = headers['structure_module']['representations']['structure_module']

        return dict(logits=self.net(act))
