import argparse, os, sys, subprocess, copy, random, time, shutil, math
from PIL import Image

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from einops import repeat, rearrange, reduce, parse_shape
import utils, models

from svgpathtools import svg2paths, parse_path, wsvg, Line, QuadraticBezier, CubicBezier, Path
import svgwrite

def to_quadratic_bezier_segments(seg):
    if isinstance(seg, CubicBezier):
        p1 = seg.start
        p2 = seg.control1
        p3 = seg.control2
        p4 = seg.end
        return QuadraticBezier(p1, 0.75 * (p2 + p3) - 0.25 * (p1 + p4), p4)

    elif isinstance(seg, Line):
        return QuadraticBezier(seg.start, (seg.start + seg.end) / 2, seg.end)
    elif isinstance(seg, QuadraticBezier):
        return seg
    else:
        raise NotImplementedError('not expected type of segment')

def convert_path_to_control_points(path, pruned=False):
    sub_paths = path.continuous_subpaths()
    if pruned:
        sub_paths = prune_paths(sub_paths)
    
    cps_list = []
    for sub_path in sub_paths:
        cps = np.array([[i.start, i.control, i.end] for i in sub_path])
        cps = np.stack((cps.real, cps.imag), axis=-1)
        n_curve, n, _ = cps.shape
        cps = cps.reshape(n_curve, n * 2)
        cps_list.append(cps)
    return cps_list

def merge_d_string_single_channel(d_string_list):
    args = ['node', 'js/merge_sin.js'] + [f'"{ds}"'for ds in d_string_list]
    res = subprocess.run(args, check=True, encoding='utf-8', capture_output=True)
    d_string = res.stdout
    path = parse_path(d_string)

    new_segs = []
    for i in range(len(path)):
        new_segs.append(to_quadratic_bezier_segments(path[i]))

    new_path = Path(*new_segs)
    return new_path, d_string

def merge_d_string(d_string_list):
    args = ['node', 'js/merge.js'] + [f'"{ds}"'for ds in d_string_list]
    res = subprocess.run(args, check=True, encoding='utf-8', capture_output=True)
    d_string = res.stdout
    path = parse_path(d_string)

    new_segs = []
    for i in range(len(path)):
        new_segs.append(to_quadratic_bezier_segments(path[i]))

    new_path = Path(*new_segs)
    return new_path, d_string

def write_path_to_svg(curve_tensor_list, filename):
    sl = 256
    canvas = svgwrite.Drawing(filename=filename, debug=True)
    canvas.viewbox(0, 0, sl, sl)

    path_d = []
    for curve_tensor in curve_tensor_list:
        path_d.append(models.gutils.path_d_from_control_points(curve_tensor, xy_flip=False))
    
    path_d = ' '.join(path_d)
    path = canvas.path(   
        d=path_d, 
        fill='#000000',
    )
    canvas.add(path)
    canvas.save()
    return canvas

def simplify_path(path):
    new_segs = []
    last_end = None
    accumulated_length = 0

    for seg in path:
        if last_end is None:
            last_end = seg.start
        sl = seg.length()
        if sl + accumulated_length < 1e-3 * 256:
            accumulated_length += sl
            continue
    
        accumulated_length = 0
        seg.start = last_end
        new_segs.append(seg)
        last_end = seg.end
    
    if len(new_segs) >= 2:
        return Path(*new_segs)
    else:
        return None

def prune_paths(paths, length=1.0, area=50):
    # list of path / something from path.continuous_subpaths()
    pruned_paths = []
    for path in paths:
        if path.length() < length or abs(path.area()) < area:
            continue

        new_path = simplify_path(path)
        if new_path is not None:
            pruned_paths.append(new_path)

    return pruned_paths

def connect_cp_for_tensor_list(cp_tensor_list, norm=False, sidelength=0):
    cp_connected_tensor_list = []
    for cp in cp_tensor_list:
        cp_connected = torch.cat([cp, cp[:, :2].roll(shifts=-1, dims=0)], dim=-1)
        if norm:
            cp_connected = 2 * cp_connected / sidelength - 1
        cp_connected_tensor_list.append(cp_connected)
    
    return cp_connected_tensor_list

def refine_svg(control_points_list, target, w, h, num_iter, loss_fn, verbose=False, prog_bar=False):
    assert(w == h)

    target = torch.as_tensor(target).cuda()

    cp_tensor_list = []
    for control_points in control_points_list:
        cp_tensor = torch.tensor(control_points[:, :-2]).cuda()
        cp_tensor.requires_grad = True
        cp_tensor_list.append(cp_tensor)

    optim = torch.optim.Adam(cp_tensor_list, lr=2, betas=(0.9, 0.999))

    renderer = models.OccRender(sidelength=w).cuda()
    imgs = []

    with tqdm(range(num_iter), leave=False, desc='Refine', disable=not prog_bar) as iters:
        for i in iters:
            optim.zero_grad()
            cp_norm_list = connect_cp_for_tensor_list(cp_tensor_list, norm=True, sidelength=w)

            img_render = renderer(cp_norm_list, kernel=4)
            if verbose:
                imgs.append(utils.tensor_to_image(img_render))

            list_of_loss = loss_fn(img_render, target, cp_norm_list)
            loss = list_of_loss['loss']
            loss.backward()
            optim.step()

            for k, v in list_of_loss.items():
                list_of_loss[k] = f'{v.item():.6f}'

            iters.set_postfix({
                **list_of_loss,
            })
            loss = None

    cp_connected = connect_cp_for_tensor_list(cp_tensor_list, norm=False)
    return {
        'control_points': cp_connected,
        'rendered_images' : imgs, 
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svg', type=str, required=True)
    parser.add_argument('--inter', type=str, default=None)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--num_iter', type=int, default=100)
    args = parser.parse_args()

    assert(os.path.exists(args.svg))

    _, attributes, svg_attributes = svg2paths(args.svg, return_svg_attributes=True)

    viewbox = svg_attributes['viewBox']
    viewbox = list(map(int, viewbox.split(' ')))
    svg_w, svg_h = viewbox[2], viewbox[3]

    d_string_list = [a['d'] + ' Z' for a in attributes]

    path, d_string = merge_d_string(d_string_list)

    if args.inter is not None:
        wsvg(paths=[path], filename=args.inter, attributes=[{'fill': '#000000'}], svg_attributes=svg_attributes)

    cps = convert_path_to_control_points(path, pruned=True)

    img = np.asarray(Image.open(args.target).convert('L').resize((256, 256), resample=Image.BILINEAR)) / 255.
    refined_path = refine_svg(cps, img, svg_w, svg_h, num_iter=args.num_iter, verbose=True)['control_points']
    write_path_to_svg(refined_path, f'refine/final_refined.svg')
