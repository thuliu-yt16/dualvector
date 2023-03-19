from re import I
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import models

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm="layer"):
        super().__init__()
        if norm == "layer":
            self.norm = nn.LayerNorm(dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise NotImplementedError("unsupported norm", norm)
        self.fn = fn
        self.norm_name = norm
    def forward(self, x, **kwargs):
        if self.norm_name == "batch":
            return self.fn(self.norm(x.transpose(1,2)).transpose(1,2), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)

class ResConn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn, norm="layer"):
        super().__init__()
        if norm == "layer":
            self.norm = nn.LayerNorm(dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise NotImplementedError("unsupported norm", norm)
        self.fn = fn
        self.norm_name = norm
    
    def forward(self, x, **kwargs):
        if self.norm_name == "batch":
            return self.norm(self.fn(x, **kwargs).transpose(1,2)).transpose(1,2)
        else:
            return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        # mask: b x n
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if mask is None:
            # dots: b x h x n x n
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            b, n = mask.shape
            h = q.shape[1]
            attn_mask = mask.view(b, 1, 1, n).expand(-1, h, n, -1)
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            dots = attn_mask + torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@models.register('tfm')
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., norm="layer"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), norm=norm),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout), norm=norm)
            ]))
            
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x
