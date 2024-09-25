from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from immunefold.common import residue_constants
from immunefold.model import atom as functional, folding
from immunefold.model.common_modules import(
        Linear,
        LayerNorm)
from immunefold.model.common_modules import get_lora_config
from immunefold.model.utils import squared_difference
from immunefold.common.metrics import contact_precision, kabsch_torch, TMscore
from immunefold.model.head_factory import registry_head

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
                # nn.ReLU(),
                Linear(dim, dim, init='relu', bias=True),
                # nn.ReLU(),
                Linear(dim, c.num_bins, init='final', bias=True))
        self.config = config

    def forward(self, headers, representations, batch):
        assert 'structure_module' in headers

        act = headers['structure_module']['representations']['structure_module']

        return dict(logits=self.net(act))

# TM-score prediction
@registry_head(name='predicted_aligned_error')
class PredictedAlignedErrorHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.proj = Linear(c.pair_channel, c.num_bins, init='final')


        # Shape (num_bins,)
        self.breaks = torch.linspace(0., c.max_error_bin, steps=c.num_bins - 1)

        self.config = config

    def forward(self, headers, representations, batch):
        c = self.config

        act = representations['pair']

        # Shape (b, num_res, num_res, num_bins)
        logits = self.proj(act)

        return dict(logits=logits, breaks=self.breaks.to(device=logits.device))
