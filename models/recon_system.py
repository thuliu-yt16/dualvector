import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from .mlp import MLP
from einops import rearrange, reduce, repeat, parse_shape
from .gutils import render_bezier_path, path_d_from_control_points, render_sdf
from objprint import op
import svgwrite
from PIL import Image
from .base import DualChannel, SingleChannel

@models.register('z-to-curve')
class CurveDecoder(nn.Module):
    def __init__(self, z_dim, n_points, n_curves, n_prims, hidden_dim, hidden_layers):
        super().__init__()
        self.z_dim = z_dim
        self.n_points = n_points
        self.n_curves = n_curves
        self.n_prims = n_prims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        out_dim = (n_points - 1) * 2 * n_curves * n_prims
        self.layers = [z_dim] + [hidden_dim] * hidden_layers + [out_dim]
        self.decoder = MLP(self.layers)

    def forward(self, z):
        # z: bs x z_dim
        # x: bs x out_dim 
        x = self.decoder(z)
        x = rearrange(x, 'b (p c pts) -> b p c pts', p=self.n_prims, c=self.n_curves, pts=(self.n_points - 1) * 2)
        start = x[:, :, :, :2]
        end = torch.roll(start, -1, dims=-2)
        x = torch.cat([x, end], dim=-1)
        x = x * 0.1
        return x


@models.register('occ-dual-channel')
# each part is treated as pos subtract neg
class OccDualChannnel(nn.Module, DualChannel):
    def __init__(self, encoder, decoder, dis_cal, img_decoder=None, detach_img_branch=False, submodule_config=None, sidelength=256, use_occ=False, occ_warmup=10):
        nn.Module.__init__(self)
        DualChannel.__init__(self)
        self.encoder = models.make(encoder)
        self.decoder = models.make(decoder)
        self.img_decoder = models.make(img_decoder) if img_decoder is not None else None
        self.dis_cal = models.make(dis_cal)

        self.n_prims = self.decoder.n_prims
        self.use_occ = use_occ
        self.sidelength = sidelength
        self._colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.occ_warmup = occ_warmup
        self.detach_img_branch = detach_img_branch
        self.submodule_config = submodule_config

    def init(self):
        self.dis_cal.init()
        if self.submodule_config is not None:
            for d in self.submodule_config:
                ckpt_path = d['ckpt']
                assert(os.path.exists(ckpt_path))
                ckpt = torch.load(ckpt_path)
                for key, name, freeze in zip(d['ckpt_key'], d['module_name'], d['freeze']):
                    print('load', name, 'from', ckpt_path + '.' + key, 'freeze:', freeze)
                    submodule = getattr(self, name)
                    models.load_submodule(submodule, ckpt, key)
                    if freeze:
                        models.freeze(submodule)

    def decode_image_from_latent_vector(self, z, **kwargs):
        if self.img_decoder is None:
            return {}
        
        if self.detach_img_branch:
            z = rearrange(z.detach(), 'b z -> b z 1 1')
        else:
            z = rearrange(z, 'b z -> b z 1 1')
        img = self.img_decoder(z)
        return {
            'rec': img,
        }
    
    def forward(self, batch, **kwargs):
        z = self.encoder(batch)

        out = {
            'z': z,
        }

        xy = batch['xy']
        curves = self.decoder(z)

        epoch = kwargs['epoch']
        is_train = kwargs['train']
        if is_train:
            kernel = 1 - ((1-4/(self.sidelength))*epoch)/self.occ_warmup
        else:
            kernel = 0
        res = self.render_paths_sdf(curves, xy, use_occ=True, kernel=kernel)
        out.update(res)

        res = self.decode_image_from_latent_vector(z, **kwargs)
        out.update(res)

        return out