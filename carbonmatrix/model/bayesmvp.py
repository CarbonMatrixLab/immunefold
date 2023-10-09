
import functools
import logging
import random

import torch
from torch import nn

from carbonmatrix.common import residue_constants
from carbonmatrix.model.lm.pretrained import load_model_and_alphabet_local

from carbonmatrix.model.seqformer import EmbeddingAndSeqformer
from carbonmatrix.model.head_factory import HeadFactory
from carbonmatrix.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)

class BayesMVP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.esm, _ = load_model_and_alphabet_local(config['esm2_model_file'])
        self.esm.half()


        self.config = config

    def _compute_language_model(self, tokens, residx):
        repr_layers = list(range(self.config.embeddings_and_seqformer.esm.num_layers + 1))

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=repr_layers, residx=residx, need_head_weights=False, return_contacts=False)

        ret = {}
        esm_embed = torch.stack([results['representations'][k][:,1:-1] for k in repr_layers], dim=-1)
        esm_logits = results['logits'][:,1:-1]

        return esm_embed, esm_logits

    def forward(self, batch, compute_loss=False):
        c = self.config

        seq = batch['seq']

        batch_size, num_residues, device = *seq.shape[:2], seq.device

        _, esm_logits = self._compute_language_model(batch['esm_seq'], batch['residx'])

        return ret
