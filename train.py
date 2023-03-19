import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--name', default=None)
parser.add_argument('--tag', default=None)
parser.add_argument('--gpu', default='0')
parser.add_argument('--resume', default=None)
parser.add_argument('--seed', default=None, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import sys
import copy
import random
import time
import shutil
from PIL import Image

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import torchvision
from einops import repeat, rearrange, reduce, parse_shape
import datasets
import losses
import models
import utils
from models import make_lr_scheduler

import warnings
warnings.filterwarnings("ignore")

def seed_all(seed):
    log(f'Global seed set to {seed}')
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
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if type(v) is torch.Tensor:
            log('  {}: shape={}, dtype={}'.format(k, tuple(v.shape), v.dtype))
        else:
            log('  {}: type={}'.format(k, type(v)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=spec['shuffle'], num_workers=min(spec['batch_size'], os.cpu_count(), 32), pin_memory=True)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        log('resume from {}'.format(config['resume']))
        sv_file = torch.load(config['resume'])

        if not utils.same_dict(sv_file['model'], config['model'], {'sd'}) or \
           not utils.same_dict(sv_file['optimizer'], config['optimizer'], {'sd'}):
            print('from ckpt:')
            print(yaml.dump(utils.without(sv_file['model'], {'sd'})))

            print('from config:')
            print(yaml.dump(config['model']))

            which_one, _ = utils.input_checkbox(['ckpt', 'config'], msg='Model/Optimizer configs are different')
            print('you select', which_one)
            if which_one == 'config':
                sv_file['model'].update(config['model'])
                sv_file['optimizer'].update(config['optimizer'])

        model = models.make(sv_file['model'], load_sd=True).cuda()

        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=config['load_optimizer'])
        
        lr_scheduler = make_lr_scheduler(optimizer, config.get('scheduler'))
        
        if config.get('run_step') is not None and config['run_step']:
            epoch_start = sv_file['epoch'] + 1
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
        else:
            epoch_start = 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        lr_scheduler = make_lr_scheduler(optimizer, config.get('scheduler'))

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    loss_fn = losses.make(config['loss'])
    return model, optimizer, epoch_start, lr_scheduler, loss_fn

def debug(model):
    has_nan = False
    v_n = []
    v_v = []
    v_g = []
    for name, parameter in model.named_parameters():
        v_n.append(name)
        v_v.append(parameter.detach().cpu() if parameter is not None else torch.zeros(1))
        v_g.append(parameter.grad.detach().cpu() if parameter.grad is not None else torch.zeros(1))
        has_nan = has_nan or \
            torch.any(torch.isnan(v_v[-1])) or \
            torch.any(torch.isnan(v_g[-1]))
    if has_nan:
        for i in range(len(v_n)):
            print(f'value {v_n[i]}: {v_v[i].min().item():.3e} ~ {v_v[i].max().item():.3e}')
            print(f'grad  {v_n[i]}: {v_g[i].min().item():.3e} ~ {v_g[i].max().item():.3e}')
        exit(0)

def train(train_loader, model, optimizer, loss_fn, epoch):
    global global_step
    model.train()
    train_loss = utils.Averager()

    with tqdm(train_loader, leave=False, desc='train') as pbar:
        for batch in pbar:
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.cuda()
            
            # with torch.autograd.detect_anomaly():
            out = model(batch, epoch=epoch, train=True)

            list_of_loss = loss_fn(out, batch, epoch=epoch)
            loss = list_of_loss['loss']

            train_loss.add(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = None

            writer.add_scalars('step loss', list_of_loss, global_step)
            global_step += 1

            for k, v in list_of_loss.items():
                list_of_loss[k] = f'{v.item():.6f}'
            
            pbar.set_postfix({
                **list_of_loss,
                'avg': train_loss.item()
            })

    return train_loss.item()

def val(val_loader, model, img_path, epoch):
    os.makedirs(img_path, exist_ok=True)
    model.eval()

    dim = 128
    _xy = utils.make_coord((dim, dim)).cuda()

    with torch.no_grad():
        with tqdm(val_loader, leave=False, desc='val') as pbar:
            for batch in val_loader:
                b = 0
                for k, v in batch.items():
                    if type(v) is torch.Tensor:
                        batch[k] = v.cuda()
                        b = v.size(0)
            
                xy = repeat(_xy, 'n d -> b n d', b=b)

                batch['xy'] = xy
                out = model(batch, epoch=epoch, train=False)
                curves = out['curves']
                curves_np = curves.detach().cpu().numpy()

                if 'occ' in out:
                    occ_img = rearrange(out['occ'], 'b (dim1 dim2) -> b () dim1 dim2', dim1=dim).detach().cpu()
                if 'iter_occs' in out:
                    iters_occ_img = rearrange(out['iter_occs'][-1], 'b (dim1 dim2) -> b dim1 dim2', dim1=dim).detach().cpu()
                for i in range(b):
                    curve_np = curves_np[i]
                    filename = os.path.join(img_path, f"{batch['index'][i]}.svg")
                    shutil.copyfile(batch['img_path'][i], filename.replace('.svg', '.png'))
                    if 'img' in batch:
                        utils.tensor_to_image(batch['img'][i, 0], filename.replace('.svg', '_inp.png'))
                    if 'refs' in batch:
                        n_ref = batch['n_ref'][i]
                        grid = torchvision.utils.make_grid(batch['refs'][i], nrow=5)
                        utils.tensor_to_image(grid[0], filename.replace('.svg', '_refs.png'))

                    if 'full_img' in batch:
                        utils.tensor_to_image(batch['full_img'][i, 0], filename.replace('.svg', '_full.png'))

                    if 'img_origin_res' in batch:
                        utils.tensor_to_image(batch['img_origin_res'][i, 0], filename.replace('.svg', '_origin_res.png'))

                    if 'rec' in out:
                        utils.tensor_to_image(out['rec'][i, 0], filename.replace('.svg', '_rec.png'))

                    if 'dis' in out:
                        utils.tensor_to_image(out['dis'][i].view(dim, dim) + 0.5, filename.replace('.svg', '_dis.png'))

                    if 'rendered' in out:
                        Image.fromarray((out['rendered'][i, 0].cpu().numpy() * 255).astype(np.uint8)).save(filename.replace('.svg', '_render.png'))

                    if 'occ' in out:
                        Image.fromarray((occ_img[i, 0].numpy() * 255).astype(np.uint8)).save(filename.replace('.svg', '_occ.png'))

                    if 'iter_occs' in out:
                        utils.tensor_to_image(iters_occ_img[i], filename.replace('.svg', '_occ_iter.png'))

                    if 'sdf' in out:
                        sdf_img = rearrange(torch.sigmoid(out['sdf']), 'b (dim1 dim2) -> b dim1 dim2', dim1=dim).detach().cpu().numpy()
                        Image.fromarray((sdf_img[i] * 255).astype(np.uint8)).save(filename.replace('.svg', '_sdf.png'))

                    if hasattr(model, 'write_paths_to_svg'):
                        utils.curves_to_svg(curve_np, filename)
                        model.write_paths_to_svg(curve_np, os.path.join(img_path, f"{batch['index'][i]}_mask.svg"))
                
                break

    return None

def main(config_, save_path):
    global config, log, writer, global_step
    config = config_
    log, writer = utils.set_save_path(save_path)
    global_step = 0

    seed_all(config['seed'])
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    ckpt_path = os.path.join(save_path, 'ckpt')
    img_path = os.path.join(save_path, 'img')
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler, loss_fn = prepare_training()
    model.init()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    val(val_loader, model, os.path.join(img_path, 'test'), epoch=0)

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, loss_fn, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec = copy.deepcopy(config['model'])
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = copy.deepcopy(config['optimizer'])
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'config': config,
        }

        torch.save(sv_file, os.path.join(ckpt_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(ckpt_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            
            val(val_loader, model_, os.path.join(img_path, str(epoch)), epoch)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    if args.seed is None:
        if 'seed' not in config:
            config['seed'] = int(time.time() * 1000) % 1000
    else:
        config['seed'] = args.seed

    config['cmd_args'] = sys.argv
    config['resume'] = args.resume

    main(config, save_path)
