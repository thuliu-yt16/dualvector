import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import models

from .mlp import MLP
from einops import rearrange, reduce, repeat, parse_shape
from .base import DualChannel

@models.register('vae-style-multi-ref-cnn')
class VAEStyleMultiRefCNN(nn.Module, DualChannel):
    def __init__(self, img_encoder, tfm, n_char, decoder, z_dim, dis_cal, latent_encoder, img_decoder, sidelength, encode_type, submodule_config=None, train_latent=True, detach_img_branch=False, use_diffvg=False):
        nn.Module.__init__(self)
        DualChannel.__init__(self)

        self.img_encoder = models.make(img_encoder)
        self.encode_type = encode_type
        self.tfm = models.make(tfm)
        self.latent_encoder = models.make(latent_encoder) if latent_encoder is not None else None
        self.mlp_head = MLP([z_dim, z_dim], bias=True, activate='gelu', activate_last=True)
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_var = nn.Linear(z_dim, z_dim)
        self.cls_token = nn.Embedding(n_char, z_dim)

        self.decoder = models.make(decoder)
        self.dis_cal = models.make(dis_cal)
        self.img_decoder = models.make(img_decoder) if img_decoder is not None else None

        self.merge = nn.Linear(z_dim * 2, z_dim)

        self.submodule_config = submodule_config

        self.n_prims = self.decoder.n_prims
        self.sidelength = sidelength
        self.train_latent = train_latent
        self.detach_img_branch = detach_img_branch
        self.use_diffvg = use_diffvg

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

    def reparameterize(self, mu, log_var):
        # mu, log_var: b x z_dim
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def ref_img_encode(self, batch):
        if self.encode_type == 'cnn':
            b, n = batch['refs'].shape[:2]

            imgs = rearrange(batch['refs'], 'b n c h w -> (b n) c h w')
            z_imgs = self.img_encoder(imgs)
            z_imgs = rearrange(z_imgs, '(b n) z -> b n z', b=b)

            mask = torch.arange(n).to(imgs.device).unsqueeze(0).expand(b, -1) >= batch['n_ref'].unsqueeze(1)
            z_imgs = self.tfm(z_imgs, mask)
            x = z_imgs.mean(dim=1) 

            x = self.mlp_head(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            return mu, log_var
        else:
            raise NotImplementedError
    
    def forward(self, batch, **kwargs):
        mu, log_var = self.ref_img_encode(batch)
        z_style = self.reparameterize(mu, log_var)
        emb_char = self.cls_token(batch['tgt_char_idx'])

        z = self.merge(torch.cat([z_style, emb_char], dim=-1))

        # dis: b p nxy
        out = {
            'mu': mu,
            'log_var': log_var,
            'z': z,
        }

        is_train = kwargs['train']
        if self.train_latent and is_train:
            if self.latent_encoder is not None:
                z_tgt = self.latent_encoder(batch).detach()
                out.update({
                    'z_detach': z_tgt
                })
        
        if not self.train_latent or not is_train:
            xy = batch['xy']
            curves = self.decoder(z)

            if self.use_diffvg:
                res = self.render_paths_diffvg(curves, sidelength=batch['img_origin_res'].size(-1))
                out.update(res)
            else:
                res = self.render_paths_sdf(curves, xy, use_occ=True, kernel=4/self.sidelength)
                out.update(res)

            res = self.decode_image_from_latent_vector(z, **kwargs)
            out.update(res)

        return out


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
