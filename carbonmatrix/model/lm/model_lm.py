from typing import Union
import functools

from torch import nn
from torch.utils.checkpoint import checkpoint
import torch
import esm
from esm.model import esm2 as E
from carbonmatrix.model.lm.transformer import TransformerLayer

class ModelLM(E.ESM2):
    def __init__(self,
            num_layers: int = 33,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
            token_dropout: bool = True,
            lora_config = {},
            ):
        
        self.lora_config = lora_config
        super().__init__(
                num_layers = num_layers,
                embed_dim = embed_dim,
                attention_heads = attention_heads,
                alphabet = alphabet,
                token_dropout = token_dropout,
                )

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    lora_config=self.lora_config,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = E.ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

        self.emb_layer_norm_after = E.ESM1bLayerNorm(self.embed_dim)

        self.lm_head = E.RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )


    def forward(self, tokens, repr_layers=[], residx=None, need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            block_fn = functools.partial(layer,
                    residx=residx,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,)

            x, attn = block_fn(x)

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result
