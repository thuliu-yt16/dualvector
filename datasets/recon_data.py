import os
import json
import glob
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from einops import rearrange, reduce, repeat

import datasets
import utils

@datasets.register('deepvecfont-sdf')
class DeepvecfontSDF(Dataset):
    def __init__(self, data_root, 
                       img_res=224, 
                       coor_res=64, 
                       n_samples=4000, 
                       char_list=list(range(26)),
                       include_lower_case=False,
                       val=False, 
                       sample_inside=False, 
                       signed_distance=False,
                       full=False, 
                       occ=False, 
                       sample_dis=True,
                       weight=False,
                       repeat=1,
                       ratio=5,
                       use_cache=True,
                       random_sample=False,
                       dataset_length=None,
                       valid_list=None):
        self.data_root = data_root
        self.repeat = repeat

        if isinstance(valid_list, list):
            font_list = sorted(valid_list)
        elif isinstance(valid_list, str):
            with open(valid_list) as f:
                font_list = sorted([line.strip() for line in f.readlines()])
        else:
            font_list = sorted(os.listdir(data_root))
    
        if val:
            print(len(font_list), 'in all for validating')
        else:
            print(len(font_list), 'in all for training')
        
        n_fonts = len(font_list)
        self.font_list = font_list[:(n_fonts // ratio)]
        self.char_list = char_list

        self.edge_and_sample = [None for i in range(len(self.font_list) * len(self.char_list))]
        self.img_res = img_res
        self.coor_res = coor_res
        self.n_samples = n_samples
        self.val = val
        self.sample_inside = sample_inside
        self.full = full
        self.occ = occ
        self.weight = weight
        self.signed_distance = signed_distance
        self.sample_dis = sample_dis
        self.include_lower_case = include_lower_case
        self.cache = use_cache

        self.random_sample = random_sample
        self.dataset_length = dataset_length

    def __len__(self):
        # return len(self.font_list) * len(self.char_list) * self.repeat if not self.random_sample else self.dataset_length
        return len(self.font_list) * len(self.char_list) * self.repeat if self.dataset_length is None else self.dataset_length
    
    def make_tensor(self, t, dtype=torch.float32):
        return torch.tensor(t, dtype=dtype)
    
    def __getitem__(self, idx):
        if not self.random_sample:
            idx = idx % (len(self.font_list) * len(self.char_list))
            char_id = idx % len(self.char_list)
            font_id = idx // len(self.char_list)
        else:
            char_id = torch.randint(len(self.char_list), (1,)).item()
            font_id = torch.randint(len(self.font_list), (1,)).item()

        img_path = os.path.join(self.data_root, self.font_list[font_id], f'{self.char_list[char_id]}_1024.png')

        if self.include_lower_case:
            cropped = Image.open(img_path).convert('L').crop((0, 0, 1024, 1280))
            origin = Image.new('L', (1280, 1280), 255)
            origin.paste(cropped, (128, 0, 1152, 1280))
        else:
            origin = Image.open(img_path).convert('L').crop((0, 0, 1024, 1024))
        img_np = np.asarray(origin.resize((self.img_res, self.img_res), resample=Image.BICUBIC))
        img = self.make_tensor(img_np / 255.).view(1, self.img_res, self.img_res)

        ret = {
            'img_path': img_path,
            'img': img,
            'index': idx,
            'font_idx': font_id,
            'char_idx': char_id,
            'font_name': self.font_list[font_id],
            'char': self.char_list[char_id],
        }

        if self.edge_and_sample[idx] is None:
            sdf_img = np.asarray(origin.resize((self.coor_res, self.coor_res), resample=Image.BICUBIC))
            h, w = self.coor_res, self.coor_res
            mask = sdf_img < 128 # in the glyph
            edge = []
            locs = np.where(mask)

            def in_range(i, j, h, w):
                return 0 <= i < h and 0 <= j < w

            for i, j in zip(*locs):
                at_edge = (in_range(i - 1, j, h, w) and not mask[i - 1, j]) or \
                        (in_range(i, j - 1, h, w) and not mask[i, j - 1]) or \
                        (in_range(i, j + 1, h, w) and not mask[i, j + 1]) or \
                        (in_range(i + 1, j, h, w) and not mask[i + 1, j]) 
                if at_edge:
                    edge.append([2 * i / h - 1, 2 * j / w - 1])

            edge = self.make_tensor(np.array(edge))
            out_points = 2 * self.make_tensor(np.array(np.where(sdf_img >= 128)).T) / self.coor_res - 1
            in_points = 2 * self.make_tensor(np.array(np.where(sdf_img < 128)).T) / self.coor_res - 1

            if self.cache:
                self.edge_and_sample[idx] = (edge, out_points, in_points, sdf_img)
        else:
            edge, out_points, in_points, sdf_img = self.edge_and_sample[idx]
        
        if self.full:
            ret['full_img'] = self.make_tensor(sdf_img / 255.).view(1, self.coor_res, self.coor_res)

        if self.val or self.n_samples == 0:
            return ret
        
        ne = edge.shape[0]
        if self.sample_inside:
            all_points = torch.cat([out_points, in_points], dim=0)
            unif = torch.ones(all_points.shape[0])
            indices = unif.multinomial(self.n_samples, replacement=self.n_samples > all_points.shape[0])
            samples = all_points[indices]
        else:
            unif = torch.ones(out_points.shape[0])
            indices = unif.multinomial(self.n_samples, replacement=self.n_samples > out_points.shape[0])
            samples = out_points[indices] # n_samples x 2
        
        ret.update({
            'xy': samples
        })
        
        if self.sample_dis:
            samples_r = repeat(samples, 'ns d -> ns ne d', ne=ne)
            edge_r = repeat(edge, 'ne d -> ns ne d', ns=self.n_samples)
            dis = torch.pow(samples_r - edge_r, 2)
            dis = reduce(dis, 'ns ne d -> ns ne', 'sum')
            dis = reduce(dis, 'ns ne -> ns', 'min')
            dis = torch.sqrt(dis)

            if self.weight:
                w = torch.exp(-dis)
                ret.update({
                    'weight': w
                })
            
            if self.sample_inside:
                if self.signed_distance:
                    dis[indices >= out_points.shape[0]] *= -1
                else:
                    dis[indices >= out_points.shape[0]] = 0

            ret.update({
                'dis': dis,
            })

    
        if self.occ:
            samples = ((ret['xy'] + 1) * self.coor_res / 2).long()
            occ_gt = torch.tensor(sdf_img / 255.).view(self.coor_res, self.coor_res)
            occ = occ_gt[samples[:, 0], samples[:, 1]]
            ret.update({
                'pixel': occ.float(),
            })

        return ret