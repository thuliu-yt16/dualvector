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

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--seed', default=42, type=int)
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

seed_all(args.seed)

ref_char_list = [0,1,26,27]

config_str = f'''
val_dataset:
  dataset:
    name: dvf-eval
    args:
      data_root: ./data/dvf_png/font_pngs/test
      img_res: 128
      valid_list: null
      ref_list: {ref_char_list}
  batch_size: 1
  shuffle: false
'''

config = yaml.load(config_str, Loader=yaml.FullLoader)
output_dir = args.outdir

sv_file = torch.load(args.resume)

system = models.make(sv_file['model'], load_sd=True).cuda()
system.init()
system.eval()
models.freeze(system)

dataloader = make_data_loaders(config)
sidelength = 256

char_nums = 52

for batch in tqdm(dataloader):
    for k, v in batch.items():
        if type(v) is torch.Tensor:
            batch[k] = v.cuda()
    
    with torch.no_grad():
        refs = batch['refs'] # 1 x 2 x 1 x 64 x 64
        ref_char_idx = batch['ref_char_idx'] # 1 x 2
        n_ref = batch['n_ref']
        tgt_char_idx = torch.arange(char_nums).cuda()

        mu, _ = system.ref_img_encode(batch)
        mu_r = mu.expand(char_nums, -1) 
        emb_char = system.cls_token(tgt_char_idx)

        z = system.merge(torch.cat([mu_r, emb_char], dim=-1))
        curves = system.decoder(z)
        img_rec = system.decode_image_from_latent_vector(z)['rec']

    n = curves.shape[0]
    curves_np_raw = curves.detach().cpu().numpy()
    curves_np = (curves_np_raw + 1) * sidelength / 2
    targets = img_rec
    assert(n == char_nums)

    font_name = batch['font_name'][0]
    save_dir = os.path.join(output_dir, 'rec_init', font_name)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n):
        char_name = i

        svg_path = os.path.join(save_dir, f'{char_name:02d}_init.svg')
        raw_path = os.path.join(save_dir, f'{char_name:02d}_raw.svg')
        img_path = os.path.join(save_dir, f'{char_name:02d}_rec.png')
        if os.path.exists(svg_path) and os.path.exists(raw_path) and os.path.exists(img_path):
            continue

        system.write_paths_to_svg(curves_np_raw[i], raw_path)

        utils.tensor_to_image(img_rec[i, 0], img_path)

        curve_np = curves_np[i]
        d_string_list = [models.gutils.path_d_from_control_points(cp, xy_flip=True) for cp in curve_np]

        path, d_string = refine_svg.merge_d_string(d_string_list)

        cps_list = refine_svg.convert_path_to_control_points(path, pruned=True)

        if len(cps_list) == 0:
            continue
        refine_svg.write_path_to_svg(cps_list, svg_path)