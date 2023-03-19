import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

losses = {}

def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(loss_spec, args=None):
    loss_args = copy.deepcopy(loss_spec.get('args', {}))
    args = args or {}
    loss_args.update(args)

    loss = losses[loss_spec['name']](**loss_args)
    return loss

class BaseLoss(nn.Module):
    @abstractmethod
    def loss_terms(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        l = self.loss_terms(*args, **kwargs)

        loss = 0
        for k, v in l.items():
            loss += v
        
        l['loss'] = loss

        return l

@register('list-loss')
class ListLoss(BaseLoss):
    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = [make(spec) for spec in loss_list]
    
    def loss_terms(self, *args, **kwargs):
        l = {}
        for loss_ins in self.loss_list:
            terms = loss_ins.loss_terms(*args, **kwargs)
            l.update(terms)
        
        return l

        