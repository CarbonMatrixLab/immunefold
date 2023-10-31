import torch
from omegaconf import open_dict

def set_lora_config(cfg, lora_r_seq, lora_r_pair, lora_scaling):
    lora_config_seq = dict(
            lora_r = lora_r_seq,
            lora_dropout = 0.1,
            lora_alpha = lora_scaling * lora_r_seq)

    lora_config_pair = dict(
            lora_r = lora_r_pair,
            lora_dropout = 0.1,
            lora_alpha = lora_scaling * lora_r_pair)

    def _set_lora_seq(x):
        with open_dict(x):
            x.lora_config = lora_config_seq

    def _set_lora_pair(x):
        with open_dict(x):
            x.lora_config = lora_config_pair

    # evoformer iteration
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
    if config.get('lora', False):
        return setup_model_lora(model, config)
    else:
        return setup_model_fft(model, config)

def setup_model_lora(model, config):
    c = config

    model.impl.requires_grad_(False)
    model.esm.requires_grad_(False)

    # assert
    model.impl.seqformer_module.proj_aa_type.requires_grad = False
    model.impl.seqformer_module.proj_rel_pos.requires_grad = False

    bias = c.get('bias', False)

    trainable_variables_dict = dict(list(model.impl.named_parameters()))
    trainable_variables = []

    for n, p in trainable_variables_dict.items():
        if n.endswith('lora_A'):
            p.requires_grad = True
            trainable_variables.append(p)

            lora_B_name = n[:-6] + 'lora_B'
            lora_B_p = trainable_variables_dict[lora_B_name]
            lora_B_p.requires_grad = True
            trainable_variables.append(lora_B_p)

            if bias:
                bias_name = n[:-6] + 'bias'
                if bias_name in trainable_variables_dict:
                    bias_p = trainable_variables_dict[bias_name]
                    bias_p.requires_grad = True
                    trainable_variables.append(bias_p)

        if 'predicted_lddt' in n or 'proj_prev_pos' in n or 'timestep_embedder' in n:
            p.requires_grad = True
            trainable_variables.append(p)

    return trainable_variables

def setup_model_fft(model, config):
    mode = config.get('fft_mode', 'full')

    if mode == 'full':
        setup_model_fft_full(model, config)
    elif mode == 'only_evoformer':
        setup_model_fft_evoformer(model, config)
    else:
        raise NotImplementedError(f'align mode {mode} not implemented yet!')

def setup_model_fft_full(model, config):
    c = config

    model.impl.requires_grad_(False)
    model.esm.requires_grad_(False)

    # assert
    model.impl.seqformer_module.proj_aa_type.requires_grad = False
    model.impl.seqformer_module.proj_rel_pos.requires_grad = False

    trainable_variables = []

    for n, p in model.impl.named_parameters():
        if 'proj_aa_type' in n or 'proj_rel_pos' in n or 'esm_embed_weights' in n or 'proj_esm_embed' in n:
            p.requires_grad = False
        #elif 'predicted_aligned_error' in n or 'predicted_lddt' in n:
        #    p.requires_grad = False
        else:
            p.requires_grad = True
            trainable_variables.append(p)
    return trainable_variables

def setup_model_fft_evoformer(model, config):
    c = config

    model.impl.requires_grad_(False)
    model.esm.requires_grad_(False)

    trainable_variables = []
    for x in model.impl.seqformer.seqformer.blocks:
        for n, p in x.named_parameters():
            p.requires_grad = True
            trainable_variables.append(p)

    return trainable_variables

def set_esm_lora_config(cfg, lora_r, lora_scaling):
    lora_config = dict(
            lora_r = lora_r,
            lora_dropout = 0.1,
            lora_alpha = lora_scaling * lora_r)
    with open_dict(cfg):
        cfg.lora_config = lora_config

    return

def setup_esm_model(model, config):
    c = config

    model.esm.requires_grad_(False)

    bias = c.get('bias', False)

    trainable_variables_dict = dict(list(model.named_parameters()))
    trainable_variables = []

    for n, p in trainable_variables_dict.items():
        if n.endswith('lora_A'):
            p.requires_grad = True
            trainable_variables.append(p)

            lora_B_name = n[:-6] + 'lora_B'
            lora_B_p = trainable_variables_dict[lora_B_name]
            lora_B_p.requires_grad = True
            trainable_variables.append(lora_B_p)

            if bias:
                bias_name = n[:-6] + 'bias'
                if bias_name in trainable_variables_dict:
                    bias_p = trainable_variables_dict[bias_name]
                    bias_p.requires_grad = True
                    trainable_variables.append(bias_p)

    return trainable_variables
