import os
import argparse
import json
from collections import OrderedDict
import torch
import ml_collections

from abfold.model.abfold import AbFold

def load(args):
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
        config = ml_collections.ConfigDict(config)


    #checkpoint = torch.load(args.model, map_location=args.map_location)
    #model = checkpoint['model']
    abfold = AbFold(config=config.model)
    esmfold = torch.load(args.esmfold_ckpt)['model_state_dict']
    
    c = config.model
    
    def _load(des, source):
        des.load_weights(source)
    
    def _has(n):
        return n in esmfold

    def _get(n):
        return esmfold[n]

    def _set(n, m):
        esmfold[n] = m
    
    def _assign(m, n):
        if not _has(n + '.bias'):
            m.load_state_dict(OrderedDict(weight = _get(n + '.weight'), ), strict=True)
        else:
            m.load_state_dict(OrderedDict(weight = _get(n + '.weight'), bias = _get(n + '.bias')), strict=True)
    def _assign_data(m, n):
        m.data = _get(n)

    def _load_seqformer_block(it):
        block = abfold.impl.seqformer.seqformer.blocks[it]
        prefix = f'trunk.blocks.{it}.'
        
        # SeqAttention
        seq_attn = block.seq_attn
        _assign(seq_attn.seq_norm, prefix + 'layernorm_1')
        _assign(seq_attn.pair_norm, prefix + 'pair_to_sequence.layernorm')
        _assign(seq_attn.proj_pair.proj, prefix + 'pair_to_sequence.linear')

        proj_q, proj_k, proj_v = torch.chunk(_get(prefix + 'seq_attention.proj.weight'), 3, dim=0)
        _set(prefix + 'seq_attention.proj_q.weight', proj_q)
        _set(prefix + 'seq_attention.proj_k.weight', proj_k)
        _set(prefix + 'seq_attention.proj_v.weight', proj_v)
        _assign(seq_attn.attn.proj_q.proj, prefix + 'seq_attention.proj_q')
        _assign(seq_attn.attn.proj_k.proj, prefix + 'seq_attention.proj_k')
        _assign(seq_attn.attn.proj_v.proj, prefix + 'seq_attention.proj_v')

        _assign(seq_attn.attn.gate.proj, prefix + 'seq_attention.g_proj')
        _assign(seq_attn.attn.proj_out.proj, prefix + 'seq_attention.o_proj')

        # Seq Transition
        seq_tran = block.seq_transition
        _assign(seq_tran.transition[0], prefix + f'mlp_seq.mlp.0')
        _assign(seq_tran.transition[1].proj, prefix + f'mlp_seq.mlp.1')
        _assign(seq_tran.transition[3].proj, prefix + f'mlp_seq.mlp.3')

        # Outer 
        outer = block.outer_product_mean
        _assign(outer.norm, prefix + 'sequence_to_pair.layernorm')
        left_proj_w, right_proj_w = torch.chunk(_get(prefix + 'sequence_to_pair.proj.weight'), 2, dim=0)
        left_proj_b, right_proj_b = torch.chunk(_get(prefix + 'sequence_to_pair.proj.bias'), 2, dim=0)
        _set(prefix + 'sequence_to_pair.left_proj.weight', left_proj_w)
        _set(prefix + 'sequence_to_pair.left_proj.bias', left_proj_b)
        _set(prefix + 'sequence_to_pair.right_proj.weight', right_proj_w)
        _set(prefix + 'sequence_to_pair.right_proj.bias', right_proj_b)
        _assign(outer.left_proj.proj, prefix + 'sequence_to_pair.left_proj')
        _assign(outer.right_proj.proj, prefix + 'sequence_to_pair.right_proj')
        
        # triangle_multiplication_outgoing
        mul = block.triangle_multiplication_outgoing
        prefix2 = prefix + 'tri_mul_out.'
        _assign(mul.norm, prefix2 + 'layer_norm_in')
        _assign(mul.final_norm, prefix2 + 'layer_norm_out')
        _assign(mul.left_proj.proj, prefix2 + 'linear_a_p')
        _assign(mul.right_proj.proj, prefix2 + 'linear_b_p')
        _assign(mul.left_gate.proj, prefix2 + 'linear_a_g')
        _assign(mul.right_gate.proj, prefix2 + 'linear_b_g')
        _assign(mul.final_gate.proj, prefix2 + 'linear_g')
        _assign(mul.proj_out.proj, prefix2 + 'linear_z')

        # triangle_multiplication_incoming
        mul = block.triangle_multiplication_incoming
        prefix2 = prefix + 'tri_mul_in.'
        _assign(mul.norm, prefix2 + 'layer_norm_in')
        _assign(mul.final_norm, prefix2 + 'layer_norm_out')
        _assign(mul.left_proj.proj, prefix2 + 'linear_a_p')
        _assign(mul.right_proj.proj, prefix2 + 'linear_b_p')
        _assign(mul.left_gate.proj, prefix2 + 'linear_a_g')
        _assign(mul.right_gate.proj, prefix2 + 'linear_b_g')
        _assign(mul.final_gate.proj, prefix2 + 'linear_g')
        _assign(mul.proj_out.proj, prefix2 + 'linear_z')

        # triangle_attention_starting_node 
        tri = block.triangle_attention_starting_node
        prefix2 = prefix + 'tri_att_start.'
        _assign(tri.norm, prefix2 + 'layer_norm')
        _assign(tri.proj_pair.proj, prefix2 + 'linear')
        _assign(tri.attn.proj_q.proj, prefix2 + 'mha.linear_q')
        _assign(tri.attn.proj_k.proj, prefix2 + 'mha.linear_k')
        _assign(tri.attn.proj_v.proj, prefix2 + 'mha.linear_v')
        _assign(tri.attn.gate.proj, prefix2 + 'mha.linear_g')
        _assign(tri.attn.proj_out.proj, prefix2 + 'mha.linear_o')

        # triangle_attention_ending_node
        tri = block.triangle_attention_ending_node
        prefix2 = prefix + 'tri_att_end.'
        _assign(tri.norm, prefix2 + 'layer_norm')
        _assign(tri.proj_pair.proj, prefix2 + 'linear')
        _assign(tri.attn.proj_q.proj, prefix2 + 'mha.linear_q')
        _assign(tri.attn.proj_k.proj, prefix2 + 'mha.linear_k')
        _assign(tri.attn.proj_v.proj, prefix2 + 'mha.linear_v')
        _assign(tri.attn.gate.proj, prefix2 + 'mha.linear_g')
        _assign(tri.attn.proj_out.proj, prefix2 + 'mha.linear_o')

        # pair_transition
        # Seq Transition
        tran = block.pair_transition
        _assign(tran.transition[0], prefix + f'mlp_pair.mlp.0')
        _assign(tran.transition[1].proj, prefix + f'mlp_pair.mlp.1')
        _assign(tran.transition[3].proj, prefix + f'mlp_pair.mlp.3')
     
    def _load_embedding():
        embed = abfold.impl.seqformer
        _assign_data(embed.esm_embed_weights, 'esm_s_combine')
        
        _assign(embed.proj_aa_type, 'embedding') 
        _assign(embed.proj_rel_pos, 'trunk.pairwise_positional_embedding.embedding')
        
        _assign(embed.proj_esm_embed[0], 'esm_s_mlp.0')
        _assign(embed.proj_esm_embed[1].proj, 'esm_s_mlp.1')
        _assign(embed.proj_esm_embed[3].proj, 'esm_s_mlp.3')

    # load embedding
    _load_embedding()

    # load seqformer blocks
    for it in range(c.embeddings_and_seqformer.seqformer_num_block):
        _load_seqformer_block(it)

    # load structure module
    
    # load heads

def main(args):
    load(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--esmfold_ckpt', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
