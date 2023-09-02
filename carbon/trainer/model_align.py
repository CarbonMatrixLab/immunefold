import torch

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
    trainable_variables = []

    if c.esm.enabled:
        for n, p in model.esm.named_parameters():
            if 'embed_tokens' in n or 'lm_head' in n or 'contact_head' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                trainable_variables.append(p)
    '''
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
    model.impl.requires_grad_(False)

    return trainable_variables
