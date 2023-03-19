import os
import time
import shutil
import math
import numpy as np
from PIL import Image

import torch
from torch import nn
import numpy as np
from torch.optim import SGD, Adam
# from optims import NAdam
from tensorboardX import SummaryWriter

import fnmatch

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def without(d, exc_keys):
    dct1 = {k: v for k, v in d.items() if k not in exc_keys}
    return dct1

def same_dict(d1, d2, exc_keys):
    return without(d1, exc_keys) == without(d2, exc_keys)


def input_checkbox(opts, keys=None, default='0', msg=''):
    prompts = [msg, f'Default: {default}'] + [f'\t{i}. {opt}' for i, opt in enumerate(opts)] + [f'Please select: ']
    if keys is None:
        keys = list(map(str, range(len(opts))))

    ind = input('\n'.join(prompts))
    while(ind not in keys and ind != ''):
        ind = input('Illegal input, input again: ')

    ind = keys.index(default if ind == '' else ind)
    return opts[ind], ind


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path)

def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in fnmatch.filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore
    return _ignore_patterns

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)

    # copy tree
    if os.path.exists(os.path.join(save_path, 'src')):
        shutil.rmtree(os.path.join(save_path, 'src'))
    shutil.copytree(os.getcwd(), os.path.join(save_path, 'src'), ignore=shutil.ignore_patterns('__pycache__*', 'data*', 'save*', '.git*', 'datasetprocess*'))

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'SGD': SGD,
        'Adam': Adam,
        # 'NAdam': NAdam,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant_(m.bias, 0)

from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, parse_path, wsvg

colors = []

def curves_to_svg(curves, filename, box=256, control_polygon=True):
    if not hasattr(curves_to_svg, "colors"):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        curves_to_svg.colors = colors
    
    if isinstance(curves, torch.Tensor):
        curves = curves.detach().cpu().numpy()

    n_paths, n_curves, n_cp = curves.shape
    n_cp = n_cp // 2

    bez_paths = []
    seg_paths = []

    for i in range(n_paths):
        bez = []
        segs = []
        for j in range(n_curves):
            cps = (curves[i, j] + 1) * box / 2
            if n_cp == 4:
                bez.append(CubicBezier(cps[1] + cps[0]*1j,
                                       cps[3] + cps[2]*1j,
                                       cps[5] + cps[4]*1j,
                                       cps[7] + cps[6]*1j))
                segs.append(Line(cps[1] + cps[0] * 1j, cps[3] + cps[2] * 1j))
                segs.append(Line(cps[3] + cps[2] * 1j, cps[5] + cps[4] * 1j))
                segs.append(Line(cps[5] + cps[4] * 1j, cps[7] + cps[6] * 1j))
            elif n_cp == 3:
                bez.append(QuadraticBezier(cps[1] + cps[0]*1j,
                                       cps[3] + cps[2]*1j,
                                       cps[5] + cps[4]*1j))
                segs.append(Line(cps[1] + cps[0] * 1j, cps[3] + cps[2] * 1j))
                segs.append(Line(cps[3] + cps[2] * 1j, cps[5] + cps[4] * 1j))
            else:
                raise NotImplementedError('not implemented order of bezier path')
        bez_paths.append(Path(*bez))
        seg_paths.append(Path(*segs))
    
    dimensions = (200, 200)
    viewbox = (0, 0, box, box)
    # colors = ['black'] * len(paths)
    # colors = curves_to_svg.colors[:len(bez_paths)]
    colors = curves_to_svg.colors

    attributes = [
        {
            'fill': colors[i % len(colors)],
            'fill-opacity': 0.3,
        } for i in range(len(bez_paths)) 
    ]

    if control_polygon:
        attributes += [
            {
                'stroke-dasharray': '10,10',
                'stroke': colors[i % len(colors)],
                'fill': 'none',
                'stroke-width': '0.1',
            } for i in range(len(seg_paths))
        ] 

    if control_polygon:
        wsvg(bez_paths + seg_paths, 
            # colors=colors, 
            # stroke_widths=stroke_widths, 
            dimensions=dimensions, 
            viewbox=viewbox, 
            attributes=attributes, 
            filename=filename)
    else:
        wsvg(bez_paths,
            # colors=colors, 
            # stroke_widths=stroke_widths, 
            dimensions=dimensions, 
            viewbox=viewbox, 
            attributes=attributes, 
            filename=filename)

def importance_sampling_create_image(idx, img_values, size = 128):
    idx_np = idx.detach().cpu().numpy()
    img_values_np = img_values.detach().cpu().numpy()

    idx_int = (((idx_np+1)/2)*size).astype(int)
    img = np.zeros([size, size])
    img[idx_int[:,0], idx_int[:,1]] = img_values_np

    return img

def tensor_to_image(ten, fname=None):
    if isinstance(ten, torch.Tensor):
        ten = ten.detach().cpu().numpy()
    
    if ten.dtype == np.float64 or ten.dtype == np.float32:
        ten = np.clip(ten, 0, 1)
        ten = (ten * 255).astype(np.uint8)
    
    if fname is None:
        return ten
    
    if len(ten.shape) == 3:
        if ten.shape[0] == 3:
            ten = ten.transpose(1, 2, 0)
        elif ten.shape[0] == 1:
            ten = ten[0]
    Image.fromarray(ten).save(fname)
    
def batched_ssim(pred, gt):
    '''
    pred: n x 1 x dim x dim
    gt: n x 1 x dim x dim
    '''


    from skimage.metrics import structural_similarity as ssim

    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    s = 0
    for i in range(pred.shape[0]):
        sv = ssim(pred[i, 0], gt[i, 0], data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=False)
        s += sv
    
    return s / pred.shape[0]