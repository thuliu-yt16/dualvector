import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses
from losses import BaseLoss
from models.gutils import bezier_length
from einops import reduce, rearrange, repeat
from objprint import objprint as op

# import lpips

@losses.register('local-loss')
class LocalLoss(BaseLoss):
    def __init__(self, lam_render, lam_reg=0):
        super().__init__()
        self.lam_render = lam_render
        self.lam_reg = lam_reg
    
    def get_curve_length(self, shape):
        points = shape.points
        num_control_points = shape.num_control_points
        n_curve = len(num_control_points)
        control_points = []
        start_index = 0
        total_len = 0
        assert(num_control_points.sum() + n_curve == points.shape[0])
        for i in range(n_curve - 1):
            num_p = num_control_points[i].item()
            assert(num_p == 1 or num_p == 0)
            if num_p == 1: # bezier curve, start_index, start_index + 1, start_index + 2
                control_points.append(points[start_index : start_index + num_p + 2])
            else: # length
                total_len += (points[start_index + 1] - points[start_index]).norm()
            start_index += num_p + 1

        if num_control_points[-1] == 1:
            index = [start_index, start_index + 1, 0]
            control_points.append(points[index])
        elif num_control_points[-1] == 0:
            total_len += (points[-1] - points[0]).norm()
        else:
            op(shape)
            exit(0)
        
        if len(control_points) > 0:
            control_points = rearrange(control_points, 'b n d -> b n d')
            curve_len = bezier_length(control_points[:, 0, :], control_points[:, 1, :], control_points[:, 2, :]).sum()
            total_len += curve_len
        return total_len 

    
    def loss_terms(self, img_rendered, target, shapes):
        loss_render = (img_rendered.mean(dim=-1) - target).pow(2).mean()
        l = {
            'render': self.lam_render * loss_render,
        }
        if self.lam_reg > 0:
            loss_reg = sum([self.get_curve_length(shape) for shape in shapes])
            l.update({
                'len': self.lam_reg * loss_reg,
            })
        return l
        

