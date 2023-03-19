import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses
from losses import BaseLoss

@losses.register('kl-loss')
class KLLoss(BaseLoss):
    def __init__(self, lam):
        self.lam = lam
    
    def loss_terms(self, out, batch, **kwargs):
        mu = out['mu']
        log_var = out['log_var']
        return {
            'kl': self.lam * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        }

@losses.register('kl-loss-two')
class KLLossTwo(BaseLoss):
    def __init__(self, lam):
        self.lam = lam
    
    def loss_terms(self, out, batch, **kwargs):
        mu = out['mu']
        log_var = out['log_var']

        tgt_mu = out['mu_detach']
        tgt_log_var = out['log_var_detach']

        return {
            'kl': self.lam * torch.mean(0.5 * torch.sum(-1 + tgt_log_var - log_var + (log_var.exp() +  (mu - tgt_mu) ** 2) / torch.clamp(tgt_log_var.exp(), min=0.1), dim=1), dim=0)
        }




