import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from einops import rearrange, reduce, repeat, parse_shape
import pydiffvg
from .gutils import render_bezier_path, path_d_from_control_points, render_sdf
from .mlp import MLP
from objprint import op
import svgwrite
from PIL import Image
from abc import abstractmethod

class SVGRender:
    def __init__(self):
        self._colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    @abstractmethod
    def render_paths_sdf(self, curves, xy, use_occ, **kwargs):
        pass

    @abstractmethod
    def render_paths_diffvg(self, curves, sidelength, **kwargs):
        pass

    @abstractmethod
    def write_paths_to_svg(self, path_tensor, filename):
        pass


class DualChannel(SVGRender):
    def __init__(self):
        super().__init__()

    def render_paths_sdf(self, curves, xy, use_occ=True, **kwargs):
        # dis: b p nxy
        dis, inner = self.dis_cal(curves, xy, return_mask=True)
        b, p, nxy = dis.shape

        pos_prim = torch.zeros((b, p, nxy), dtype=torch.bool, device=dis.device)
        pos_prim[:, :p//2, :] = True

        point_in_pos_prim = torch.logical_and(pos_prim, inner)
        point_out_neg_prim = torch.logical_and(torch.logical_not(pos_prim), torch.logical_not(inner))

        clip_dis = dis.clone()
        clip_dis[point_in_pos_prim] = 0
        clip_dis[point_out_neg_prim] = 0

        clip_dis = rearrange(clip_dis, 'b (ch hp) nxy -> b ch hp nxy', ch=2)
        clip_dis = reduce(clip_dis, 'b ch hp nxy -> b hp nxy', reduction='max')
        clip_dis = reduce(clip_dis, 'b hp nxy -> b nxy', reduction='min')

        out = {
            'dis': clip_dis,
            'curves': curves,
        }

        if use_occ:
            occ_dis = dis.clone()

            occ_dis[point_in_pos_prim] *= -1
            occ_dis[point_out_neg_prim] *= -1

            occ_dis = rearrange(occ_dis, 'b (ch hp) nxy -> b ch hp nxy', ch=2) # channel: pos/neg

            alias_warmup = kwargs['kernel']

            occ = F.sigmoid(occ_dis) * 2 - 1 
            # kernel size: 4
            occ = render_sdf(occ, max(alias_warmup, 4/(self.sidelength)))
            occ = reduce(occ, 'b ch hp nxy -> b hp nxy', reduction='max')
            occ = reduce(occ, 'b hp nxy -> b nxy', reduction='min')

            out.update({
                'occ': occ,
            })

        return out
    
    def render_paths_diffvg(self, curves, sidelength=256):
        out = {
            'curves': curves,
        }

        render = pydiffvg.RenderFunction.apply

        b, p, c, pts = curves.shape
        h, w = sidelength, sidelength

        background = torch.ones([h, w, 4]).to(curves.device)
        # curves: b p c pts
        num_control_points = torch.tensor([pts / 2 - 2] * c)
        all_points = rearrange(curves, 'b p c (n d) -> b p c n d', d=2)[:, :, :, :-1, :]
        all_points = rearrange(all_points, 'b p c n d -> b p (c n) d')

        assert(h == w)
        all_points = (all_points + 1) / 2 * h

        imgs_from_diffvg = []
        for i in range(b):
            parts = []
            for j in range(p//2):
                shapes = []
                shape_groups = []

                path_pos = pydiffvg.Path(num_control_points=num_control_points, points=all_points[i, j], is_closed=True)
                shapes.append(path_pos)

                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
                shape_groups.append(path_group)

                path_neg = pydiffvg.Path(num_control_points=num_control_points, points=all_points[i, j + p//2], is_closed=True)
                shapes.append(path_neg)

                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0]))
                shape_groups.append(path_group)

                scene_args = pydiffvg.RenderFunction.serialize_scene(\
                    w, h, shapes, shape_groups)

                rendered_img = render(w, # width
                            h, # height
                            3,   # num_samples_x
                            3,   # num_samples_y
                            42,   # seed
                            background,
                            *scene_args)
                parts.append(rendered_img[:, :, 0].transpose(0, 1))
            
            parts = rearrange(parts, 'hp h w -> hp h w')
            img_from_diffvg = reduce(parts, 'hp h w -> h w', 'min')
            imgs_from_diffvg.append(img_from_diffvg)

        imgs_from_diffvg = rearrange(imgs_from_diffvg, 'b h w -> b () h w')
        out.update({
            'rendered': imgs_from_diffvg,
        })

        return out

    def write_paths_to_svg(self, path_tensor, filename, xy_flip=True):
        if isinstance(path_tensor, torch.Tensor):
            path_tensor = path_tensor.detach().cpu().numpy()

        n_paths, n_curves, n_cp = path_tensor.shape
        n_cp = n_cp // 2

        n_pos_path = n_paths // 2

        sl = 256

        canvas = svgwrite.Drawing(filename=filename, debug=True)
        canvas.viewbox(0, 0, sl, sl)

        path_tensor = (path_tensor + 1) / 2 * sl
        canvas_rect = canvas.rect(insert=(0, 0), size=(sl, sl), fill='white')

        for i in range(n_pos_path):
            path_d = path_d_from_control_points(path_tensor[i], xy_flip=xy_flip)
            mask_d = path_d_from_control_points(path_tensor[i + n_pos_path], xy_flip=xy_flip)

            mask = canvas.mask(id=f'mask{i}')
            mask.add(canvas_rect)
            mask.add(canvas.path(d=mask_d, fill='black'))
            canvas.defs.add(mask)
            path = canvas.path(   
                d=path_d, 
                fill=self._colors[i%len(self._colors)],
                fill_opacity='0.3',
                mask=f'url(#mask{i})',
            )
            canvas.add(path)
        canvas.save()
        
        return canvas

class SingleChannel(SVGRender):
    def __init__(self):
        super().__init__()
    
    def render_paths_sdf(self, curves, xy, use_occ=True, **kwargs):
        # dis: b p nxy
        dis, inner = self.dis_cal(curves, xy, return_mask=True)
        dis[inner] *= -1

        dis = reduce(dis, 'b p nxy -> b nxy', reduction='min')

        out = {
            'dis': dis,
            'curves': curves,
        }

        if use_occ:
            alias_warmup = kwargs['kernel']

            occ = dis.clone()
            occ = F.sigmoid(occ) * 2 - 1 
            # kernel size: 4
            occ = render_sdf(occ, max(alias_warmup, 4/(self.sidelength)))
            out.update({
                'occ': occ,
            })

            return out
    
    def render_paths_diffvg(self, curves, sidelength=256):
        render = pydiffvg.RenderFunction.apply

        b, p, c, pts = curves.shape
        h, w = sidelength, sidelength

        background = torch.ones([h, w, 4]).to(curves.device)
        # curves: b p c pts
        num_control_points = torch.tensor([pts / 2 - 2] * c)
        all_points = rearrange(curves, 'b p c (n d) -> b p c n d', d=2)[:, :, :, :-1, :]
        all_points = rearrange(all_points, 'b p c n d -> b p (c n) d')

        assert(h == w)
        all_points = (all_points + 1) / 2 * h

        imgs_from_diffvg = []
        for i in range(b):
            shapes = []
            shape_groups = []
            for j in range(p):
                points = all_points[i, j]
                path = pydiffvg.Path(num_control_points=num_control_points, points=points, is_closed=True)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
                shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                w, h, shapes, shape_groups)

            rendered_img = render(w, # width
                        h, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        42,   # seed
                        background,
                        *scene_args)
            imgs_from_diffvg.append(rendered_img[:, :, 0].transpose(0, 1))
        imgs_from_diffvg = rearrange(imgs_from_diffvg, 'b h w -> b () h w')
        return {
            'rendered': imgs_from_diffvg,
        }

    def write_paths_to_svg(self, path_tensor, filename):
        if isinstance(path_tensor, torch.Tensor):
            path_tensor = path_tensor.detach().cpu().numpy()

        n_paths, n_curves, n_cp = path_tensor.shape
        n_cp = n_cp // 2

        sl = 256

        canvas = svgwrite.Drawing(filename=filename, debug=True)
        canvas.viewbox(0, 0, sl, sl)

        path_tensor = (path_tensor + 1) / 2 * sl

        for i in range(n_paths):
            path_d = path_d_from_control_points(path_tensor[i])
            path = canvas.path(   
                d=path_d, 
                fill=self._colors[i%len(self._colors)],
                fill_opacity='0.3',
            )
            canvas.add(path)
        canvas.save()
        return canvas


