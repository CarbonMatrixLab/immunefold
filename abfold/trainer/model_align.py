import torch

def setup_model(model, config):
    c = config
    print(c)

    # aa_embed
    '''
    if not c.aa_embed.enable:
        model.impl.seqformer.proj_aa_type.requires_grad = False
        print('freeze aa')
    
    if not c.pos_embed.enable:
        model.impl.seqformer.proj_rel_pos = False
    '''
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
  
    if isinstance(c.seqformer.align_layers, str) and c.seqformer.align_layers == 'all':
        align_layers = list(range(len(model.impl.seqformer.seqformer.blocks)))
    else:
        align_lauers = c.seqformer.align_layers
    print(align_layers)
    trainable_variables = []
    if c.seqformer.enabled:
        for n in align_layers:
            x = model.impl.seqformer.seqformer.blocks[n]
            for p in x.parameters():
                p.requires_grad = True
                trainable_variables.append(p)

    return trainable_variables
