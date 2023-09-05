import functools

import torch

_feature_factory = {}

def registry_feature(fn):
    @functools.wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feature_factory
    _feature_factory[fn.__name__] = fc

    return fc

class FeatureFactory:
    def __init__(self, config, is_training=True):
        self.config = config
        self.training = is_training

    def build(self, protein):
        for fn, kwargs in self.config:
            f = _feature_factory[fn](is_training=self.training, **kwargs)
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
