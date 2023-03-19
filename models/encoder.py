from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import models

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int, planes: int,
        stride: int = 1,
        act: str = 'relu',
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)

        if act == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise NotImplementedError('not implemented activation')

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@models.register('img64-to-z')
class ImageEncoder64(nn.Module):
    def __init__(self, img_ef_dim, z_dim, key='img'):
        super().__init__()
        self.key = key
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self._norm_layer = nn.BatchNorm2d
        self.act = 'leaky_relu'

        # 64 x 64
        self.conv1 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.img_ef_dim)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)

        self.inplanes = self.img_ef_dim
        # 32 x 32
        self.layer1 = self._make_layer(self.img_ef_dim, 2, stride=1)
        self.layer2 = self._make_layer(self.img_ef_dim * 2, 2, stride=2)
        self.layer3 = self._make_layer(self.img_ef_dim * 4, 2, stride=2)
        self.layer4 = self._make_layer(self.img_ef_dim * 8, 3, stride=2)

        self.conv2 = nn.Conv2d(self.img_ef_dim * 8, self.img_ef_dim * 16, 4, stride=1, padding=0, bias=True)
        self.fc = nn.Linear(self.img_ef_dim * 16, self.z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        block = BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.act, downsample, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act=self.act, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, inp):
        if self.key is None:
            x = self.conv1(inp)
        else:
            x = self.conv1(inp[self.key])
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(x) # maybe remove
        x = self.fc(x)
        
        return x
    

@models.register('embed')
class Embed(nn.Module):
    def __init__(self, n_embed, z_dim):
        super().__init__()
        self.embed = nn.Embedding(n_embed, z_dim)
        self.init_embed_weight()

    def init_embed_weight(self):
        weights = torch.ones_like(self.embed.weight, requires_grad=True).to(self.embed.weight.device)*0.5
        self.embed.weight = torch.nn.Parameter(weights)
    
    def forward(self, x):
        return self.embed(x)

@models.register('index-to-z')
class IndexEmbed(nn.Module):
    def __init__(self, n_embed, z_dim):
        super().__init__()
        self.embed = nn.Embedding(n_embed, z_dim)
    
    def forward(self, batch):
        return self.embed(batch['index'])

@models.register('family-char-to-z')
class FamilyCharEmbed(nn.Module):
    def __init__(self, n_family, dim_family, n_char, dim_char):
        super().__init__()
        self.family_embed = nn.Embedding(n_family, dim_family)
        self.char_embed = nn.Embedding(n_char, dim_char)
    
    def forward(self, batch):
        embed_fam = self.family_embed(batch['font_idx'])
        embed_chr = self.char_embed(batch['char_idx'])
        out = torch.cat([embed_fam, embed_chr], dim=-1)
        return out

