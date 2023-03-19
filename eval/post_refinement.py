import argparse
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from PIL import Image
import imageio

import yaml, shutil
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

import torchvision

import datasets
import models, losses
import utils
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce, parse_shape

import refine_svg
from svgpathtools import svg2paths, parse_path, wsvg, Line, QuadraticBezier, CubicBezier, Path
import pydiffvg
import itertools
from tqdm import tqdm
from svg_simplification import svg_file_simplification

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', required=True, type=str)
parser.add_argument('--input', required=True, type=str)
parser.add_argument('--fmin', default=0, type=int)
parser.add_argument('--fmax', default=200, type=int)
args = parser.parse_args()

exp_dir = args.outdir
exp_name = 'p4'

config_str = '''
loss:
  name: local-loss
  args:
    lam_render: 1
    lam_reg: 1.e-6
'''
config = yaml.load(config_str, Loader=yaml.FullLoader)
loss_fn = losses.make(config['loss'])

def get_reference_image_path(path, font, char):
    return os.path.join(path, font, f'{char:02d}_rec.png')

def get_init_svg_path(path, font, char):
    return os.path.join(path, font, f'{char:02d}_init.svg')

def get_dst_svg_path(path, font, char):
    return os.path.join(get_dst_font_dir(path, font), f'{char:02d}_{exp_name}.svg')

def get_dst_font_dir(path, font):
    return os.path.join(exp_dir, font)

def post_refinement(path, font, glyph):
    dst_dir = get_dst_font_dir(path, font)
    os.makedirs(dst_dir, exist_ok=True)
    init_svg_path = get_init_svg_path(path, font, glyph)
    dst_svg_path = get_dst_svg_path(path, font, glyph)
    image_path = get_reference_image_path(path, font, glyph)

    target = torchvision.transforms.ToTensor()(Image.open(image_path).convert('L')).to(pydiffvg.get_device())
    n_times = 4

    if not os.path.exists(init_svg_path):
        return None
    
    if os.path.exists(dst_svg_path):
        # skip
        return None
    

    for i in range(n_times):
        if i == 0:
            canvas_width, canvas_height, shapes, shape_groups = \
                pydiffvg.svg_to_scene(init_svg_path)
        else:
            if i == 1:
                group, quad_line, merge, split = True, True, False, True
            elif i == 2:
                group, quad_line, merge, split = True, True, False, False
            elif i == 3:
                group, quad_line, merge, split = True, True, True, False
            
            success = svg_file_simplification(dst_svg_path.replace('.svg', f'_{i-1}.svg'), dst_svg_path.replace('.svg', f'_{i-1}_sim.svg'),\
                    group=group, quad_line=quad_line, merge=merge, split=split)
            if not success:
                return None
            canvas_width, canvas_height, shapes, shape_groups = \
                pydiffvg.svg_to_scene(dst_svg_path.replace('.svg', f'_{i-1}_sim.svg'))
        iter_steps = 50 

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)

        render = pydiffvg.RenderFunction.apply
        
        # The output image is in linear RGB space. Do Gamma correction before saving the image.
        points_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)

        # Optimize
        points_optim = torch.optim.Adam(points_vars, lr=0.5, betas=(0.9, 0.999))

        # Adam iterations.
        with tqdm(range(iter_steps), leave=False, desc='Refine') as iters:
            for _ in iters:
                points_optim.zero_grad()

                # Forward pass: render the image.
                scene_args = pydiffvg.RenderFunction.serialize_scene(\
                    canvas_width, canvas_height, shapes, shape_groups)
                try:
                    img = render(canvas_width, # width
                                canvas_height, # height
                                2,   # num_samples_x
                                2,   # num_samples_y
                                42,   # seed
                                None, # bg
                                *scene_args)
                except:
                    return None
                # Compose img with white background
                img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
                img = img[:, :, :3]
                loss_dict = loss_fn(img, target, shapes)
            
                # Backpropagate the gradients.
                loss_dict['loss'].backward()
            
                # Take a gradient descent step.
                points_optim.step()

                for k, v in loss_dict.items():
                    loss_dict[k] = f'{v.item():.6f}'

                iters.set_postfix({
                    **loss_dict,
                })

        if i < n_times - 1:
            pydiffvg.save_svg_paths_only(dst_svg_path.replace('.svg', f'_{i}.svg'), canvas_width, canvas_height, shapes, shape_groups)
        else:
            pydiffvg.save_svg_paths_only(dst_svg_path, canvas_width, canvas_height, shapes, shape_groups)

    return float(loss_dict['render'])
    
def main():
    origin_path = args.input
    font_list = [f'{i:04d}' for i in range(args.fmin, args.fmax)]
    glyph_list = list(range(52))

    font_list = font_list if font_list else os.listdir(origin_path)
    os.makedirs(exp_dir, exist_ok=True)

    task = sorted([(origin_path, f, g) for f, g in itertools.product(font_list, glyph_list)])

    losses = []
    for path, font, glyph in tqdm(task):

        loss_render = post_refinement(path, font, glyph)
        if loss_render is not None:
            losses.append(loss_render)
    
    print(sum(losses) / len(losses))

if __name__ == '__main__':
    main()
