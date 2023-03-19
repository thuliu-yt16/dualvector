import numpy as np
import torch
import torch.nn as nn
import models
from .mlp import MLP
from .gutils import sd_bezier, render_bezier_path
from einops import rearrange, reduce, repeat, parse_shape
import svgwrite
from .gutils import sd_parabola, eval_parabola

@models.register('analytic-df')
class AnalyticBezierDF(nn.Module):
    def __init__(self, n_control_points):
        super().__init__()
        assert(n_control_points == 3)
        self.n_control_points = n_control_points
    
    def forward(self, xy, control_points):
        assert(control_points.size(1) == 2 * self.n_control_points)
        sd = sd_bezier(control_points[:, 0:2], control_points[:, 2:4], control_points[:, 4:6], xy)
        return sd

@models.register('batched-curve-to-dis')
class DistanceCalculator(nn.Module):
    def __init__(self, ds, ckpt=None, sidelength=256):
        super().__init__()
        if isinstance(ds, dict):
            self.diff_sdf = models.make(ds)
        else:
            self.diff_sdf = ds
        self.ckpt = ckpt
        self.sidelength = sidelength
    
    def init(self):
        if self.ckpt is not None:
            ckpt = torch.load(self.ckpt)
            self.diff_sdf.load_state_dict(ckpt['model']['sd'], strict=True)
            models.freeze(self.diff_sdf)
            print('***** freeze diff_sdf *****')
    
    def forward(self, curves, xy, return_mask=False):
        b, p, c, pts = curves.shape
        _, nxy, _ = xy.shape

        curves_r = repeat(curves, 'b p c pts -> (b nxy p c) pts', nxy=nxy)
        xy_r = repeat(xy, 'b nxy dxy -> (b nxy p c) dxy', p=p, c=c)

        dis = self.diff_sdf(xy_r, curves_r)
        dis = rearrange(dis, '(b nxy p c) () -> b nxy p c', b=b, nxy=nxy, p=p, c=c)
        dis = reduce(dis, 'b nxy p c -> b p nxy', reduction='min')

        if return_mask:

            path_mask = np.zeros((b, p, self.sidelength, self.sidelength), dtype=np.bool) # inside is true
            curves_np = curves.detach().cpu().numpy()
            for i in range(b):
                for j in range(p):
                    path_mask[i, j] = render_bezier_path(curves_np[i, j], self.sidelength) > 128
            
            path_mask = torch.tensor(path_mask).to(curves.device)
            
            xy_grid = (xy + 1) / 2 * self.sidelength
            xy_grid = xy_grid.long()

            # xy_inside: b p nxy
            xy_inside = torch.zeros_like(dis, dtype=torch.bool)
            for i in range(b):
                for j in range(p):
                    xy_inside[i, j] = path_mask[i, j][xy_grid[i, :, 1], xy_grid[i, :, 0]]
            return dis, xy_inside

        else:
            # dis: b p nxy
            return dis

