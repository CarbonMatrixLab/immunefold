import torch
from omegaconf import open_dict

def set_lora_config(cfg, lora_r_seq, lora_r_pair):
    lora_config_seq = dict(
            lora_r = lora_r_seq,
            lora_dropout = 0.1,
            lora_alpha = 2 * lora_r_seq)

    lora_config_pair = dict(
            lora_r = lora_r_pair,
            lora_dropout = 0.1,
            lora_alpha = 2 * lora_r_pair)

    def _set_lora_seq(x):
        with open_dict(x):
            x.lora_config = lora_config_seq

    def _set_lora_pair(x):
        with open_dict(x):
            x.lora_config = lora_config_pair

    # evoformer
    seqformer = cfg.embeddings_and_seqformer.seqformer

    _set_lora_seq(seqformer.seq_attention_with_pair_bias)
    _set_lora_seq(seqformer.seq_transition)

    _set_lora_pair(seqformer.outer_product_mean)
    _set_lora_pair(seqformer.triangle_multiplication_outgoing)
    _set_lora_pair(seqformer.triangle_multiplication_incoming)
    _set_lora_pair(seqformer.triangle_attention_starting_node)
    _set_lora_pair(seqformer.triangle_attention_ending_node)
    _set_lora_pair(seqformer.pair_transition)

    # heads
    # the head of predicted LDDT is ignored
    heads = cfg.heads
    _set_lora_pair(heads.distogram)
    _set_lora_pair(heads.structure_module)
    _set_lora_pair(heads.structure_module.torsion)

def setup_model(model, config):
    c = config

    model.impl.requires_grad_(False)
    model.esm.requires_grad_(False)
    model.impl.seqformer_module.proj_aa_type.requires_grad = False
    model.impl.seqformer_module.proj_rel_pos.requires_grad = False

    trainable_variables = []

    for n, p in model.named_parameters():
        if n.endswith('lora_A') or n.endswith('lora_B') or 'predicted_lddt' in n:
            p.requires_grad = True
            trainable_variables.append(p)
        '''
        if p.requires_grad:
            trainable_variables.append(p)
        '''

    return trainable_variables
