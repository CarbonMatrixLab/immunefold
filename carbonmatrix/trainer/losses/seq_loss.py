import torch
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.trainer.losses.loss_factory import registry_loss

@registry_loss
def seq_masked_loss(batch, values, config):
    labels, label_mask = batch['label_seq'], batch['label_seq_mask']
    logits = values['logits']
    eps = 1e-8

    logits = rearrange(logits, 'b l c -> b c l')
    
    loss = F.cross_entropy(logits, labels, reduction='none')
    
    if 'weight' in batch:
        loss = torch.sum(loss * label_mask * batch['weight'][:,None]) / (torch.sum(label_mask) + eps)
    else:
        loss = torch.sum(loss * label_mask) / (torch.sum(label_mask) + eps)

    return dict(loss=loss)
