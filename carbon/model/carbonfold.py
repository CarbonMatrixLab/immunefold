import functools
import logging
import random

import torch
from torch import nn

from carbon.common import residue_constants
from carbon.model.lm.pretrained import load_model_and_alphabet_local
from carbon.model.seqformer import EmbeddingAndSeqformer
from carbon.model.head import HeaderBuilder
from carbon.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)

class CarbonFoldIteration(nn.Module):
    def __init__(self, config):
        super().__init__()

        c = config

        self.seqformer_module = EmbeddingAndSeqformer(c.embeddings_and_seqformer)

        self.heads = HeaderBuilder.build(
                c.heads,
                seq_channel=c.embeddings_and_seqformer.seq_channel,
                pair_channel=c.embeddings_and_seqformer.pair_channel,
                parent=self)

        self.config = config

    def forward(self, batch, compute_loss = False):
        c = self.config

        seq_act, pair_act = self.seqformer_module(batch)

        representations = {'pair': pair_act, 'seq': seq_act}

        ret = {}

        ret['representations'] = representations

        ret['heads'] = {}

        for name, module, options in self.heads:
            if compute_loss or name == 'folding':
                value = module(ret['heads'], representations, batch)
                if value is not None:
                    ret['heads'][name] = value

        return ret

class CarbonFold(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.esm, _, _ = load_model_and_alphabet_local(config['esm2_model_file'])

        self.impl = CarbonFoldIteration(config)

        self.config = config

    def _compute_language_model(self, tokens, residx):
        repr_layers = list(range(self.config.embeddings_and_seqformer.esm.num_layers + 1))

        results = self.esm(tokens, repr_layers=repr_layers, residx=residx, need_head_weights=False, return_contacts=False)

        ret = {}
        esm_embed = torch.stack([results['representations'][k][:,1:-1] for k in repr_layers], dim=-1)
        esm_logits = results['logits'][:,1:-1]

        return esm_embed, esm_logits

    def forward(self, batch, compute_loss=False):
        c = self.config

        seq = batch['seq']

        batch_size, num_residues, device = *seq.shape[:2], seq.device

        with torch.enable_grad():
            esm_embed, esm_logits = self._compute_language_model(batch['esm_seq'], batch['residx'])
            batch.update(esm_embed = esm_embed)

        def get_prev(ret):
            prev_pseudo_beta = pseudo_beta_fn_v2(batch['seq'], ret['heads']['folding']['final_atom_positions'])
            prev_disto_bins = dgram_from_positions(prev_pseudo_beta, **self.config.embeddings_and_seqformer.prev_pos)

            new_prev = {
                    'prev_pos': prev_disto_bins.detach(),
                    'prev_seq': ret['representations']['seq'].detach(),
                    'prev_pair': ret['representations']['pair'].detach()
                    }
            return new_prev

        # Just to adapt to ESMFOLD
        emb_config = c.embeddings_and_seqformer
        prev = {
                'prev_pos': torch.zeros([batch_size, num_residues, num_residues], device=device, dtype=torch.int64),
                'prev_seq': torch.zeros([batch_size, num_residues, emb_config.seq_channel], device=device),
                'prev_pair': torch.zeros([batch_size, num_residues, num_residues, emb_config.pair_channel], device=device)
        }
        batch.update(prev)

        if self.training:
            num_recycle = random.randint(0, c.num_recycle)
        else:
            num_recycle = c.num_recycle

        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
            with torch.no_grad():
                batch.update(is_recycling=True)
                for i in range(num_recycle):
                    ret = self.impl(batch, compute_loss=False)

                    prev = get_prev(ret)
                    batch.update(prev)

        batch.update(is_recycling=False)
        ret = self.impl(batch, compute_loss=compute_loss)
        ret.update(esm_logits = esm_logits)

        return ret
