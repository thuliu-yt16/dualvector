import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses
from losses import BaseLoss
import lpips

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

@losses.register('img-loss')
class ImgLoss(BaseLoss):
    def __init__(self, lams):
        super().__init__()
        self.lams = lams
        if 'pct' in self.lams:
            self.lpips = lpips.LPIPS(net='vgg').cuda()
    
    def loss_terms(self, out, batch, **kwargs):
        l = {}
        if 'l2' in self.lams:
            l['l2'] = self.lams['l2'] * torch.mean((out['rec'] - batch['full_img']) ** 2)
        
        if 'pct' in self.lams:
            norm_rec = out['rec'].repeat(1, 3, 1, 1) * 2 - 1
            norm_gt = batch['full_img'].repeat(1, 3, 1, 1) * 2 - 1
            l['pct'] = self.lams['pct'] * torch.mean(self.lpips(norm_rec, norm_gt))

        return l

@losses.register('latent-loss')
class LatentLoss(BaseLoss):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
    
    def loss_terms(self, out, batch, epoch):
        l = {}
        l['z'] = self.lam * nn.MSELoss()(out['z'], out['z_detach'])
        return l

@losses.register('sdf-loss')
class SdfLoss(BaseLoss):
    def __init__(self, lams, udf_warmup=20):
        super().__init__()
        self.lams = lams
        self.udf_warmup = udf_warmup
        self.lpips = None
    
    def loss_terms(self, out, batch, epoch):
        lams = self.lams
        l = {}

        # rendered with diffvg loss
        if 'img' in lams:
            if out['rendered'].shape == batch['full_img'].shape:
                l['img_l2'] = lams['img'] * torch.mean((out['rendered'] - batch['full_img']) ** 2)
            elif out['rendered'].shape == batch['img_origin_res'].shape:
                l['img_l2'] = lams['img'] * torch.mean((out['rendered'] - batch['img_origin_res']) ** 2)
        
        if 'pct' in lams:
            if self.lpips is None:
                self.lpips = lpips.LPIPS(net='vgg').cuda()
            norm_rec = out['rendered'].repeat(1, 3, 1, 1) * 2 - 1
            norm_gt = batch['full_img'].repeat(1, 3, 1, 1) * 2 - 1
            l['img_pct'] = lams['pct'] * torch.mean(self.lpips(norm_rec, norm_gt))
        
        # rendered with occupancy loss 
        if 'occ' in lams:
            l['occ'] = lams['occ'] * torch.mean((out['occ'] - batch['pixel']) ** 2)
        if 'occ_l1' in lams:
            l['occ_l1'] = lams['occ_l1'] * torch.mean(torch.abs(out['occ'] - batch['pixel']))
        
        # unsigned distance loss
        if self.udf_warmup is not None and epoch <= self.udf_warmup:
            if 'l1' in lams:
                if 'weight' in batch:
                    l['l1'] = lams['l1'] * torch.sum(torch.abs(out['dis'] - batch['dis']) * batch['weight']) / torch.sum(batch['weight'])
                else:
                    l['l1'] = lams['l1'] * nn.L1Loss()(out['dis'], batch['dis'])
        
        # signed distance loss
        if 'sdf' in lams:
            gt_sdf = batch['dis'].clone()
            gt_sdf[batch['pixel'] < 0.5] *= -1
            l['sdf'] = lams['sdf'] * torch.abs(out['sdf'] - gt_sdf).mean()
        
        # gradient of sdf loss
        if 'eik' in lams:
            grad = gradient(out['sdf'], batch['xy'])
            l['eik'] = lams['eik'] * torch.abs(grad.norm(dim=-1) - 1).mean()
        
        return l