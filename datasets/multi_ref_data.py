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

@datasets.register('dvf-eval')
class DvfEval(Dataset):
    def __init__(self, data_root,
                       img_res=224,
                       ref_list=list(range(26)),
                       ratio=1,
                       use_lowercase=True,
                       valid_list=None):
        self.data_root = data_root
        if valid_list is not None:
            if isinstance(valid_list, list):
                font_list = sorted(valid_list)
            else:
                with open(valid_list) as f:
                    font_list = sorted([line.strip() for line in f.readlines()])
        else:
            font_list = sorted(os.listdir(data_root))

        n_fonts = len(font_list)
        self.font_list = font_list[:(n_fonts // ratio)]
        self.ref_list = ref_list
        self.img_res = img_res
        self.use_lowercase = use_lowercase

    def __len__(self):
        return len(self.font_list)
    
    def make_tensor(self, t, dtype=torch.float32):
        return torch.tensor(t, dtype=dtype)
    
    def get_image(self, font, char):
        ref_path = os.path.join(self.data_root, font, f'{char}_1024.png')

        if self.use_lowercase:
            cropped = Image.open(ref_path).convert('L').crop((0, 0, 1024, 1280))
            ref_origin = Image.new('L', (1280, 1280), 255)
            ref_origin.paste(cropped, (128, 0, 1152, 1280))
        else:
            ref_origin = Image.open(ref_path).convert('L').crop((0, 0, 1024, 1024))
        ref_np = np.asarray(ref_origin.resize((self.img_res, self.img_res), resample=Image.BILINEAR))
        ref_img = self.make_tensor(ref_np / 255.).view(1, self.img_res, self.img_res)
        
        return ref_img
    
    def __getitem__(self, idx):
        n_ref = len(self.ref_list)
        ref_list = self.make_tensor(self.ref_list, dtype=torch.long)

        font_id = idx
        font_name = self.font_list[font_id]
        font_path = os.path.join(self.data_root, font_name)

        ref_imgs = []
        for char in ref_list:
            ref_imgs.append(self.get_image(font_name, char))
        
        ref_imgs = rearrange(ref_imgs, 'n 1 h w -> n 1 h w')

        ret = {
            'refs': ref_imgs,
            'ref_char_idx': ref_list,
            'n_ref': n_ref,
            'font_idx': font_id,
            'index': idx,
            'font_path': font_path,
            'font_name': font_name, 
        }

        return ret



@datasets.register('multi-ref-dvf-generation')
class MultiRefDvfGeneration(Dataset):
    def __init__(self, data_root, 
                       img_res=224, 
                       coor_res=64, 
                       n_samples=4000, 
                       char_list=list(range(26)),
                       val=False, 
                       sample_inside=False, 
                       signed=False,
                       distance=False,
                       full=False, 
                       occ=False, 
                       ratio=5,
                       origin_res=False, # origin resolution
                       include_lower_case=False,
                       n_refs=[2,5],
                       use_cache=True,
                       length=None,
                       valid_list=None):
        self.data_root = data_root
        if valid_list is not None:
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

        self.img_res = img_res
        self.coor_res = coor_res
        self.n_samples = n_samples
        self.val = val
        self.sample_inside = sample_inside
        self.full = full
        self.occ = occ
        self.signed = signed
        self.distance = distance
        self.origin_res = origin_res
        self.min_ref = n_refs[0]
        self.max_ref = n_refs[1]
        self.include_lower_case = include_lower_case
        self.use_cache = use_cache

        self.edge_and_sample = [None for i in range(len(self.font_list) * len(self.char_list))]
        self.images_64 = [None for i in range(len(self.font_list) * len(self.char_list))]
        self.length = length

    def __len__(self):
        return self.length or len(self.font_list) * len(self.char_list)
    
    def make_tensor(self, t, dtype=torch.float32):
        return torch.tensor(t, dtype=dtype)
    
    def get_image(self, font_id, char_id):
        index = font_id * len(self.char_list) + char_id
        if self.images_64[index] is None:
            ref_path = os.path.join(self.data_root, self.font_list[font_id], f'{self.char_list[char_id]}_1024.png')

            if self.include_lower_case:
                cropped = Image.open(ref_path).convert('L').crop((0, 0, 1024, 1280))
                ref_origin = Image.new('L', (1280, 1280), 255)
                ref_origin.paste(cropped, (128, 0, 1152, 1280))
            else:
                ref_origin = Image.open(ref_path).convert('L').crop((0, 0, 1024, 1024))

            ref_np = np.asarray(ref_origin.resize((self.img_res, self.img_res), resample=Image.BILINEAR))
            ref_img = self.make_tensor(ref_np / 255.).view(1, self.img_res, self.img_res)
            if self.use_cache:
                self.images_64[index] = ref_img
            else:
                return ref_img
        
        return self.images_64[index]
    
    def __getitem__(self, idx):
        if self.length is not None:
            char_id = torch.randint(len(self.char_list), size=()).item()
            font_id = torch.randint(len(self.font_list), size=()).item()
        else:
            char_id = idx % len(self.char_list)
            font_id = idx // len(self.char_list)
        img_path = os.path.join(self.data_root, self.font_list[font_id], f'{self.char_list[char_id]}_1024.png')
        if not self.include_lower_case:
            n_ref = torch.randint(low=self.min_ref, high=self.max_ref + 1, size=(1,)).item()
            ref_char_idx = torch.randperm(26)[:self.max_ref]

            ref_imgs = []
            for i, ref_char_id in enumerate(ref_char_idx):
                if i < n_ref:
                    ref_imgs.append(self.get_image(font_id, ref_char_id.item()))
                else:
                    ref_imgs.append(0.5 * torch.ones((1, self.img_res, self.img_res), dtype=torch.float32))
            
            ref_imgs = rearrange(ref_imgs, 'n 1 h w -> n 1 h w')
        else:
            n_ref = torch.randint(low=self.min_ref, high=self.max_ref + 1, size=(1,)).item()
            ref_char_idx = torch.randperm(26)[:self.max_ref]
            ref_char_idx = ref_char_idx.repeat(2,1).transpose(0,1)
            ref_char_idx[:, 1] += 26
            ref_char_idx = ref_char_idx.flatten()

            ref_imgs = []
            for i, ref_char_id in enumerate(ref_char_idx):
                if i < n_ref * 2:
                    ref_imgs.append(self.get_image(font_id, ref_char_id.item()))
                else:
                    ref_imgs.append(0.5 * torch.ones((1, self.img_res, self.img_res), dtype=torch.float32))

            ref_imgs = rearrange(ref_imgs, 'n 1 h w -> n 1 h w')
            n_ref = n_ref * 2

        ret = {
            'refs': ref_imgs,
            'ref_char_idx': ref_char_idx,
            'n_ref': n_ref,
            'tgt_char_idx': char_id,
            'font_idx': font_id,
            'img_path': img_path,
            'index': idx,
        }

        if self.origin_res:
            ret['img_origin_res'] = self.get_image(font_id, char_id)

        if self.full:
            if self.include_lower_case:
                cropped = Image.open(img_path).convert('L').crop((0, 0, 1024, 1280))
                img_origin = Image.new('L', (1280, 1280), 255)
                img_origin.paste(cropped, (128, 0, 1152, 1280))
            else:
                img_origin = Image.open(img_path).convert('L').crop((0, 0, 1024, 1024))

            img_np = np.asarray(img_origin.resize((self.coor_res, self.coor_res), resample=Image.BICUBIC))
            ret['full_img'] = self.make_tensor(img_np / 255.).view(1, self.coor_res, self.coor_res)

        if self.val or self.n_samples == 0:
            return ret

        if self.edge_and_sample[idx] is None:
            origin = Image.open(img_path).convert('L').crop((0, 0, 1024, 1024))
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
            self.edge_and_sample[idx] = (edge, out_points, in_points, sdf_img)

        else:
            edge, out_points, in_points, sdf_img = self.edge_and_sample[idx]
        
        ne = edge.shape[0]
        
        if self.sample_inside:
            all_points = torch.cat([out_points, in_points], dim=0)
            unif = torch.ones(all_points.shape[0])
            indices = unif.multinomial(self.n_samples, replacement=self.n_samples > all_points.shape[0])
            samples = all_points[indices]
            ret['xy'] = samples
            if self.distance:
                samples_r = repeat(samples, 'ns d -> ns ne d', ne=ne)
                edge_r = repeat(edge, 'ne d -> ns ne d', ns=self.n_samples)
                dis = torch.pow(samples_r - edge_r, 2)
                dis = reduce(dis, 'ns ne d -> ns ne', 'sum')
                dis = reduce(dis, 'ns ne -> ns', 'min')
                dis = torch.sqrt(dis)

                if self.signed:
                    dis[indices >= out_points.shape[0]] *= -1
                else:
                    dis[indices >= out_points.shape[0]] = 0

                ret['dis'] = dis,
        else:
            unif = torch.ones(out_points.shape[0])
            out_indices = unif.multinomial(self.n_samples, replacement=self.n_samples > out_points.shape[0])

            samples = out_points[out_indices] # n_samples x 2
            ret['xy'] = samples

            if self.distance:
                samples_r = repeat(samples, 'ns d -> ns ne d', ne=ne)
                edge_r = repeat(edge, 'ne d -> ns ne d', ns=self.n_samples)
                dis = torch.pow(samples_r - edge_r, 2)
                dis = reduce(dis, 'ns ne d -> ns ne', 'sum')
                dis = reduce(dis, 'ns ne -> ns', 'min')
                dis = torch.sqrt(dis)
                ret['dis'] = dis

        if self.occ:
            samples = ((ret['xy'] + 1) * self.coor_res / 2).long()
            occ_gt = torch.tensor(sdf_img / 255.).view(self.coor_res, self.coor_res)
            occ = occ_gt[samples[:, 0], samples[:, 1]]
            ret.update({
                'pixel': occ.float(),
            })

        return ret
