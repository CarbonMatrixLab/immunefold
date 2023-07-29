import torch
from torch.nn import functional as F
from einops import rearrange

def loss_func(batch, values):
    labels, label_mask = batch['label_seq'], batch['label_mask']
    logits = values['logits']
    eps = 1e-8

    logits = rearrange(logits, 'b l c -> b c l')
    
    loss = F.cross_entropy(logits, labels, reduction='none')

    loss = torch.sum(loss * label_mask) / (torch.sum(label_mask) + eps)

    return loss
