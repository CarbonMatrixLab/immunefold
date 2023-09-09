import functools

import torch

_feature_factory = {}

def registry_feature(fn):
    global _feature_factory
    _feature_factory[fn.__name__] = fn

    return fn

class FeatureFactory:
    def __init__(self, config):
        self.config = config

    def _transform(self, batch):
        for fn, kwargs in self.config.items():
            batch = _feature_factory[fn](batch, **kwargs)
        return batch

    def __call__(self, batch):
        return self._transform(batch)
