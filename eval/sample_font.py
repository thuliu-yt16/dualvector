import argparse
import os
import sys
sys.path.append(os.getcwd())
import time
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

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--n-sample', default=20, type=int)
parser.add_argument('--begin-index', default=0, type=int)

args = parser.parse_args()

def seed_all(seed):
    random.seed(seed) # Python
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu vars

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=spec['shuffle'], num_workers=12, pin_memory=True)
    return loader

def make_data_loaders(config):
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return val_loader

ref_char_list = [0,1]

config_str = f'''
loss:
  name: local-loss
  args:
    lam_render: 1
    lam_reg: 1.e-6
'''

seed = args.seed
seed_all(seed)
print('seed:', seed)

config = yaml.load(config_str, Loader=yaml.FullLoader)
output_dir = args.outdir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'all_rec'), exist_ok=True)

sv_file = torch.load(args.resume)

system = models.make(sv_file['model'], load_sd=True).cuda()
system.init()
system.eval()
models.freeze(system)

sidelength = 256
loss_fn = losses.make(config['loss'])
char_nums = 52

n_sample = args.n_sample
sample_begin = args.begin_index

with open(os.path.join(output_dir, 'seed.txt'), 'w') as f:
    f.write(str(seed))

for sample_i in tqdm(range(sample_begin, sample_begin + n_sample)):
    parent_dir = os.path.join(output_dir, f'{sample_i:04d}')
    os.makedirs(parent_dir, exist_ok=True)

    with torch.no_grad():
        tgt_char_idx = torch.arange(char_nums).cuda()
        z_style = torch.randn(1, 256).cuda() * 1.5
        torch.save(z_style.detach().cpu(), os.path.join(parent_dir, 'z_style.pt'))

        z_style = z_style.expand(char_nums, -1)
        emb_char = system.cls_token(tgt_char_idx)
        z = system.merge(torch.cat([z_style, emb_char], dim=-1))
        curves = system.decoder(z)

        img_rec = system.decode_image_from_latent_vector(z)['rec']

    n = curves.shape[0]
    curves_np = curves.detach().cpu().numpy()
    curves_np = (curves_np + 1) * sidelength / 2
    targets = img_rec

    torchvision.utils.save_image(targets, os.path.join(output_dir, 'all_rec', f'{sample_i:04d}_rec.png'), nrow=13)

    for i in range(n):
        path_prefix = os.path.join(output_dir, f'{sample_i:04d}', f'{i:02d}')
        if os.path.exists(path_prefix + '_init.svg'):
            continue

        target = targets[i, 0]
        utils.tensor_to_image(img_rec[i, 0], path_prefix + '_rec.png')

        curve_np = curves_np[i]
        d_string_list = [models.gutils.path_d_from_control_points(cp, xy_flip=True) for cp in curve_np]
        path, d_string = refine_svg.merge_d_string(d_string_list)

        cps_list = refine_svg.convert_path_to_control_points(path, pruned=True)

        if len(cps_list) == 0:
            continue
        refine_svg.write_path_to_svg(cps_list, path_prefix + '_init.svg')
        continue