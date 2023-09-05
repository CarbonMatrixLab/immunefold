import functools

from torch import nn

_loss_factory = {}

def registry_loss(fn):
    global _loss_factory
    _loss_factory[fn.__name__] = fn
    return fn

class LossFactory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.loss_fns = {k : functools.partial(_loss_factory[v.loss_fn], config=v.config) for k, v in config.items()}

    def forward(self, value, batch):
        outputs = {}

        for k, v in self.config.items():
            outputs[k] = self.loss_fns[k](batch, value)
        outputs['loss'] = sum(v.weight * outputs[k]['loss'] for k, v in self.config.items())
        return outputs
