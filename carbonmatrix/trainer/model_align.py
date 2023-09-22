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

def setup_model(model, config):
    c = config

    # aa_embed
    '''
    if not c.aa_embed.enable:
        model.impl.seqformer.proj_aa_type.requires_grad = False
        print('freeze aa')

    if not c.pos_embed.enable:
        model.impl.seqformer.proj_rel_pos = False
    '''
    '''
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.requires_grad = False
    '''

    '''
    if c.esm.enabled:
        for n, p in model.esm.named_parameters():
            if 'embed_tokens' in n or 'lm_head' in n or 'contact_head' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                trainable_variables.append(p)
    if c.seqformer.enabled:
        if isinstance(c.seqformer.align_layers, str) and c.seqformer.align_layers == 'all':
            align_layers = list(range(len(model.impl.seqformer_module.seqformer.blocks)))
        else:
            align_layers = c.seqformer.align_layers

        for n in align_layers:
            x = model.impl.seqformer_module.seqformer.blocks[n]
            for p in x.parameters():
                p.requires_grad = True
                trainable_variables.append(p)
    '''
    # model.impl.requires_grad_(False)

    trainable_variables = []

    model.esm.requires_grad_(False)
    model.impl.seqformer_module.proj_aa_type.requires_grad = False
    model.impl.seqformer_module.proj_rel_pos.requires_grad = False

    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_variables.append(p)

    return trainable_variables
