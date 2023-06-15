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
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.requires_grad = False
  
    trainable_variables = []
    if c.seqformer.enabled:
        for n in c.seqformer.align_layers:
            x = model.impl.seqformer.seqformer.blocks[n]
            for n, p in x.named_parameters():
                p.requires_grad = True
                trainable_variables.append(p)

    return trainable_variables
