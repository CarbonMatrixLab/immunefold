import functools
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

        self.breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1)
        self.proj = Linear(num_in_pair_channel, c.num_bins, init='final')

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

class MetricDict(dict):
    def __add__(self, o):
        n = MetricDict(**self)
        for k in o:
            if k in n:
                n[k] = n[k] + o[k]
            else:
                n[k] = o[k]
        return n

    def __mul__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] * o
        return n

    def __truediv__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] / o
        return n

@registry_head(name='metric')
class MetricDictHead(nn.Module):
    """Head to calculate metrics
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):
        metrics = MetricDict()
        if 'distogram' in headers:
            assert 'logits' in headers['distogram'] and 'breaks' in headers['distogram']
            logits, breaks = headers['distogram']['logits'], headers['distogram']['breaks']
            positions = batch['pseudo_beta']
            mask = batch['pseudo_beta_mask']

            cutoff = self.config.get('contact_cutoff', 8.0)
            t =  torch.sum(breaks <= cutoff)
            pred = F.softmax(logits, dim=-1)
            pred = torch.sum(pred[...,:t+1], dim=-1)
            #truth = torch.cdist(positions, positions, p=2)
            truth = torch.sqrt(torch.sum(squared_difference(positions[:,:,None], positions[:,None]), dim=-1))

            precision_list = contact_precision(
                    pred, truth, mask=mask,
                    ratios=self.config.get('contact_ratios'),
                    ranges=self.config.get('contact_ranges'),
                    cutoff=cutoff)
            metrics['contact'] = MetricDict()
            for (i, j), ratio, precision in precision_list:
                #i, j = default(i, 0), default(j, 'inf')

                metrics['contact'][f'[{i},{j})_{ratio}'] = precision
        return dict(loss=metrics) if metrics else None

@registry_head(name='tmscore')
class TMscoreHead(nn.Module):
    """Head to predict TM-score.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):
        c = self.config

        # only for CA atom
        if 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch:
            preds, labels = headers['structure_module']['final_atom_positions'][...,1,:].detach(), batch['atom14_gt_positions'][...,1,:].detach()
            gt_mask = batch['atom14_gt_exists'][...,1]

            tmscore = 0.
            for b in range(preds.shape[0]):
                mask = gt_mask[b]

                with torch.cuda.amp.autocast(enabled=False):
                    pred_aligned, label_aligned = kabsch_torch(
                            rearrange(preds[b][mask], 'c d -> d c'),
                            rearrange(labels[b][mask], 'c d -> d c'))

                tmscore += TMscore(pred_aligned[None,:,:], label_aligned[None,:,:], L=torch.sum(mask).item())

            return dict(loss = tmscore / preds.shape[0])
        return None

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
