import torch
from torch.nn import functional as F

def pad_for_batch(items, pad_len, pad_value):
    aux_shape = 2 * (items[0].dim() - 1) * [0]
    return torch.stack([F.pad(xx, aux_shape + [0, pad_len - xx.shape[0]], value=pad_value) for xx in items], dim=0)
