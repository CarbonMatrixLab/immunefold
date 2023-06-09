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

    assigned_param_set = set()
    
    def _load(des, source):
        des.load_weights(source)
    
    def _has(n):
        return n in esmfold

    def _get(n):
        return esmfold[n]

    def _set(n, m):
        esmfold[n] = m

    def _del(n):
        del esmfold[n]
    
    def _assign(m, n):
        if not _has(n + '.bias'):
            m.load_state_dict(OrderedDict(weight = _get(n + '.weight'), ), strict=True)
            assigned_param_set.add(n + '.weight')
        else:
            m.load_state_dict(OrderedDict(weight = _get(n + '.weight'), bias = _get(n + '.bias')), strict=True)
            assigned_param_set.add(n + '.weight')
            assigned_param_set.add(n + '.bias')
    def _assign_data(m, n):
        m.data = _get(n)
        assigned_param_set.add(n)

    def _load_seqformer_block(it):
        block = abfold.impl.seqformer.seqformer.blocks[it]
        prefix = f'trunk.blocks.{it}.'
        
        # SeqAttention
        seq_attn = block.seq_attn
        _assign(seq_attn.seq_norm, prefix + 'layernorm_1')
        _assign(seq_attn.pair_norm, prefix + 'pair_to_sequence.layernorm')
        _assign(seq_attn.proj_pair, prefix + 'pair_to_sequence.linear')

        proj_q, proj_k, proj_v = torch.chunk(_get(prefix + 'seq_attention.proj.weight'), 3, dim=0)
        _del(prefix + 'seq_attention.proj.weight')
        _set(prefix + 'seq_attention.proj_q.weight', proj_q)
        _set(prefix + 'seq_attention.proj_k.weight', proj_k)
        _set(prefix + 'seq_attention.proj_v.weight', proj_v)
        _assign(seq_attn.attn.proj_q, prefix + 'seq_attention.proj_q')
        _assign(seq_attn.attn.proj_k, prefix + 'seq_attention.proj_k')
        _assign(seq_attn.attn.proj_v, prefix + 'seq_attention.proj_v')

        _assign(seq_attn.attn.gate, prefix + 'seq_attention.g_proj')
        _assign(seq_attn.attn.proj_out, prefix + 'seq_attention.o_proj')

        # Seq Transition
        seq_tran = block.seq_transition
        _assign(seq_tran.transition[0], prefix + f'mlp_seq.mlp.0')
        _assign(seq_tran.transition[1], prefix + f'mlp_seq.mlp.1')
        _assign(seq_tran.transition[3], prefix + f'mlp_seq.mlp.3')

        # Outer 
        outer = block.outer_product_mean
        prefix2 = prefix + 'sequence_to_pair.'
        _assign(outer.norm, prefix2 + 'layernorm')
        left_proj_w, right_proj_w = torch.chunk(_get(prefix2 + 'proj.weight'), 2, dim=0)
        left_proj_b, right_proj_b = torch.chunk(_get(prefix2 + 'proj.bias'), 2, dim=0)
        _del(prefix2 + 'proj.weight')
        _del(prefix2 + 'proj.bias')
        _set(prefix2 + 'left_proj.weight', left_proj_w)
        _set(prefix2 + 'left_proj.bias', left_proj_b)
        _set(prefix2 + 'right_proj.weight', right_proj_w)
        _set(prefix2 + 'right_proj.bias', right_proj_b)
        _assign(outer.left_proj, prefix2 + 'left_proj')
        _assign(outer.right_proj, prefix2 + 'right_proj')
        _assign(outer.out_proj, prefix2 + 'o_proj')
        
        # triangle_multiplication_outgoing
        mul = block.triangle_multiplication_outgoing
        prefix2 = prefix + 'tri_mul_out.'
        _assign(mul.norm, prefix2 + 'layer_norm_in')
        _assign(mul.final_norm, prefix2 + 'layer_norm_out')
        _assign(mul.left_proj, prefix2 + 'linear_a_p')
        _assign(mul.right_proj, prefix2 + 'linear_b_p')
        _assign(mul.left_gate, prefix2 + 'linear_a_g')
        _assign(mul.right_gate, prefix2 + 'linear_b_g')
        _assign(mul.final_gate, prefix2 + 'linear_g')
        _assign(mul.proj_out, prefix2 + 'linear_z')

        # triangle_multiplication_incoming
        mul = block.triangle_multiplication_incoming
        prefix2 = prefix + 'tri_mul_in.'
        _assign(mul.norm, prefix2 + 'layer_norm_in')
        _assign(mul.final_norm, prefix2 + 'layer_norm_out')
        _assign(mul.left_proj, prefix2 + 'linear_a_p')
        _assign(mul.right_proj, prefix2 + 'linear_b_p')
        _assign(mul.left_gate, prefix2 + 'linear_a_g')
        _assign(mul.right_gate, prefix2 + 'linear_b_g')
        _assign(mul.final_gate, prefix2 + 'linear_g')
        _assign(mul.proj_out, prefix2 + 'linear_z')

        # triangle_attention_starting_node 
        tri = block.triangle_attention_starting_node
        prefix2 = prefix + 'tri_att_start.'
        _assign(tri.norm, prefix2 + 'layer_norm')
        _assign(tri.proj_pair, prefix2 + 'linear')
        _assign(tri.attn.proj_q, prefix2 + 'mha.linear_q')
        _assign(tri.attn.proj_k, prefix2 + 'mha.linear_k')
        _assign(tri.attn.proj_v, prefix2 + 'mha.linear_v')
        _assign(tri.attn.gate, prefix2 + 'mha.linear_g')
        _assign(tri.attn.proj_out, prefix2 + 'mha.linear_o')

        # triangle_attention_ending_node
        tri = block.triangle_attention_ending_node
        prefix2 = prefix + 'tri_att_end.'
        _assign(tri.norm, prefix2 + 'layer_norm')
        _assign(tri.proj_pair, prefix2 + 'linear')
        _assign(tri.attn.proj_q, prefix2 + 'mha.linear_q')
        _assign(tri.attn.proj_k, prefix2 + 'mha.linear_k')
        _assign(tri.attn.proj_v, prefix2 + 'mha.linear_v')
        _assign(tri.attn.gate, prefix2 + 'mha.linear_g')
        _assign(tri.attn.proj_out, prefix2 + 'mha.linear_o')

        # pair_transition
        # Seq Transition
        tran = block.pair_transition
        _assign(tran.transition[0], prefix + f'mlp_pair.mlp.0')
        _assign(tran.transition[1], prefix + f'mlp_pair.mlp.1')
        _assign(tran.transition[3], prefix + f'mlp_pair.mlp.3')
     
    def _load_embedding():
        embed = abfold.impl.seqformer
        _assign_data(embed.esm_embed_weights, 'esm_s_combine')
        
        _assign(embed.proj_aa_type, 'embedding') 
        _assign(embed.proj_rel_pos, 'trunk.pairwise_positional_embedding.embedding')
        
        _assign(embed.proj_esm_embed[0], 'esm_s_mlp.0')
        _assign(embed.proj_esm_embed[1], 'esm_s_mlp.1')
        _assign(embed.proj_esm_embed[3], 'esm_s_mlp.3')

    def _load_structure_module():
        struc = abfold.impl.structure_module.struct_module
        _assign(struc.proj_init_seq_act, 'trunk.trunk2sm_s') 
        _assign(struc.proj_init_pair_act, 'trunk.trunk2sm_z') 
        
        prefix = 'trunk.structure_module.'
        _assign(struc.init_seq_layer_norm, prefix + 'layer_norm_s')
        _assign(struc.init_pair_layer_norm, prefix + 'layer_norm_z')

        _assign(struc.proj_seq, prefix + 'linear_in')

        # ipa 
        ipa = struc.attention_module
        prefix = 'trunk.structure_module.ipa.'
        _assign_data(ipa.trainable_point_weights, prefix + 'head_weights') 
        _assign(ipa.proj_q_scalar, prefix + 'linear_q')
        _assign(ipa.proj_kv_scalar, prefix + 'linear_kv')
        _assign(ipa.proj_q_point_local, prefix + 'linear_q_points')
        _assign(ipa.proj_kv_point_local, prefix + 'linear_kv_points')
        _assign(ipa.proj_kv_scalar, prefix + 'linear_kv')

        _assign(ipa.proj_pair, prefix + 'linear_b')
        _assign(ipa.final_proj, prefix + 'linear_out')

        _assign(struc.attention_layer_norm, 'trunk.structure_module.layer_norm_ipa')

        tran = struc.transition_module
        prefix = 'trunk.structure_module.transition.layers.'
        _assign(tran[0], prefix + '0.linear_1')
        _assign(tran[2], prefix + '0.linear_2')
        _assign(tran[4], prefix + '0.linear_3')

        # transition layer norm
        _assign(struc.transition_layer_norm, 'trunk.structure_module.transition.layer_norm') 

        # bb update
        _assign(struc.affine_update, 'trunk.structure_module.bb_update.linear')


        # sidechain
        sc =  struc.sidechain_module.torsion_module
        prefix = 'trunk.structure_module.angle_resnet.'
        _assign(sc.proj_act[1], prefix + 'linear_in')
        _assign(sc.proj_init_act[1], prefix + 'linear_initial')
        _assign(sc.blocks[0].net[1], prefix + 'layers.0.linear_1')
        _assign(sc.blocks[0].net[3], prefix + 'layers.0.linear_2')
        _assign(sc.blocks[1].net[1], prefix + 'layers.1.linear_1')
        _assign(sc.blocks[1].net[3], prefix + 'layers.1.linear_2')
        _assign(sc.projection, prefix + 'linear_out')

    def _load_recycling():
        r = abfold.impl.seqformer
        _assign(r.prev_seq_norm, 'trunk.recycle_s_norm')
        _assign(r.prev_pair_norm, 'trunk.recycle_z_norm')
        _assign(r.proj_prev_pos, 'trunk.recycle_disto')

    # load embedding
    _load_embedding()

    # load seqformer blocks
    for it in range(c.embeddings_and_seqformer.seqformer_num_block):
        _load_seqformer_block(it)

    # load structure module
    _load_structure_module()

    # load features in recycling
    _load_recycling()

    # load heads

    # check
    print('assigned params')
    for n in assigned_param_set:
        print(n)

    print('unassigned params')
    for n in esmfold:
        if n not in assigned_param_set and not n.startswith('esm.'):
            print(n)

def main(args):
    load(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--esmfold_ckpt', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
