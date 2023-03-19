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

from svgpathtools import svg2paths2, parse_path, wsvg, Line, QuadraticBezier, CubicBezier, Path
import re
import glob


def implicitize_bezier_curve(a, b, c, norm=True):
    '''
    Bt = a(1-t)^2 + bt(1-t) + ct^2
    '''
    x1, x2, x3 = a.real, b.real, c.real
    y1, y2, y3 = a.imag, b.imag, c.imag
    A = y1**2 - 4*y1*y2 + 2*y1*y3 + 4*y2**2 - 4*y2*y3 + y3**2
    B = x1**2 - 4*x1*x2 + 2*x1*x3 + 4*x2**2 - 4*x2*x3 + x3**2
    C = -2*x1*y1 + 4*x1*y2 - 2*x1*y3 + 4*x2*y1 - 8*x2*y2 + 4*x2*y3 - 2*x3*y1 + 4*x3*y2 - 2*x3*y3
    D = 2*x1*y1*y3 - 4*x1*y2**2 + 4*x1*y2*y3 - 2*x1*y3**2 + 4*x2*y1*y2 -8*x2*y1*y3 + \
        2*x3*y1*y3 - 4*x3*y2**2 + 4*x3*y1*y2 - 2*x3*y1**2  + 4*x2*y2*y3

    E = 2*y1*x1*x3 - 4*y1*x2**2 + 4*y1*x2*x3 - 2*y1*x3**2 + 4*y2*x1*x2 -8*y2*x1*x3 + \
        2*y3*x1*x3 - 4*y3*x2**2 + 4*y3*x1*x2 - 2*y3*x1**2  + 4*y2*x2*x3
    
    F = (x1*y3)**2 - 4*x1*x2*y2*y3 -2*x1*x3*y1*y3 + 4*x1*x3*y2**2 + 4*x2**2*y1*y3 - 4*x2*x3*y1*y2 + (x3*y1)**2

    eff = np.array([A,B,C,D,E,F])
    if norm:
        norm_eff = eff / np.linalg.norm(eff, ord=np.inf)
        return norm_eff
    else:
        return eff


def quad_to_line(path, angle_threshold=171, length_threshold=1):
    new_segs = []
    cos_t = np.cos(angle_threshold/180*np.pi)
    for seg in path:
        if isinstance(seg, QuadraticBezier):
            # A = seg.start
            # B = seg.control
            # C = seg.end
            ab = seg.control - seg.start
            cb = seg.control - seg.end
            if abs(ab) < length_threshold or abs(cb) <  length_threshold:
                new_segs.append(Line(seg.start, seg.end))
            else:
                cos_abc = (ab.real * cb.real + ab.imag * cb.imag) / abs(ab) / abs(cb)
                if cos_abc < cos_t:
                    new_segs.append(Line(seg.start, seg.end))
                else:
                    new_segs.append(seg)
        else:
            new_segs.append(seg)
    
    return Path(*new_segs)

def cos_complex(a, b):
    return (a.real * b.real + a.imag * b.imag) / abs(a) / abs(b)

def merge_two_line(a, b):
    return Line(a.start, b.end)

def vec_intersect(s1, s2, a, b):
    '''
    A = s1 + a*t
    B = s2 + b*t
    '''
    ax, ay = a.real, a.imag
    bx, by = b.real, b.imag
    x1, y1 = s1.real, s1.imag
    x2, y2 = s2.real, s2.imag
    l = np.array([[ay, -ax], [by, -bx]])
    r = np.array([ay*x1-ax*y1, by*x2-bx*y2])
    x = np.linalg.solve(l, r)
    return complex(x[0] + x[1]*1j)

def merge_two_quad(a, b):
    s = a.start
    e = b.end
    c = vec_intersect(s, e, a.control - s, b.control - e)
    return QuadraticBezier(s, c, e)

def group_near_points(path, size, threshold=3):
    new_segs = []
    last_end = None
    accumulated_length = 0

    for seg in path:
        if last_end is None:
            last_end = seg.start
        sl = seg.length()
        if sl + accumulated_length < threshold:
            accumulated_length += sl
            continue
    
        accumulated_length = 0
        seg.start = last_end
        new_segs.append(seg)
        last_end = seg.end
    
    if accumulated_length > 0:
        if len(new_segs) == 0:
            return None
        new_segs[0].start = last_end
    
    if len(new_segs) >= 2:
        return new_segs
    else:
        return None

def get_wh(svg_attributes):
    if 'viewBox' in svg_attributes:
        vb = svg_attributes['viewBox']
        view_box_array_comma = vb.split(',')
        view_box_array = vb.split()
        if len(view_box_array) < len(view_box_array_comma):
            view_box_array = view_box_array_comma
        w = int(view_box_array[2])
        h = int(view_box_array[3])
        return w, h
    
    return int(svg_attributes['width']), int(svg_attributes['height'])

def split_segments(path, size, length_threshold=0.1, angle_threshold=90):
    new_segs = []
    w, h = size
    cos_th = np.cos(angle_threshold*np.pi/180)
    for seg in path:
        if isinstance(seg, QuadraticBezier):
            s = seg.start
            c = seg.control
            e = seg.end
            if seg.length() > length_threshold * w:
                c1 = (s + c) / 2
                c2 = (c + e) / 2
                mid = (c1 + c2) / 2
                new_segs.append(QuadraticBezier(s, c1, mid))
                new_segs.append(QuadraticBezier(mid, c2, e))
            else:
                new_segs.append(seg)

        elif isinstance(seg, Line):
            if seg.length() > length_threshold * w:
                s = seg.start
                e = seg.end
                mid = (s + e) / 2
                new_segs.append(Line(s, mid))
                new_segs.append(Line(mid, e))
            else:
                new_segs.append(seg)
    
    return new_segs

def merge_segments(path, size, angle_threshold=175, eff_threshold=0.02):
    '''
    coord norm to [-0.5, 0.5]
    consider only adjust segments
    '''
    w, h = size

    last_vec = None
    last_imp = None
    lp = len(path)

    cos_t = np.cos(angle_threshold/180*np.pi)
    new_segs = []

    buf = []

    def merge_buf(buf):
        if len(buf) == 1:
            return buf[0]
        fn = merge_two_quad if isinstance(buf[0], QuadraticBezier) else merge_two_line
        return fn(buf[0], buf[-1])

    for seg in path:
        if len(buf) == 0:
            buf.append(seg)
        elif not (type(seg) is type(buf[0])):
            res = merge_buf(buf)
            new_segs.append(res)
            buf = [seg]
        else:
            std = buf[0]
            if isinstance(seg, Line):
                vec = seg.end - seg.start
                vec_std = std.end - std.start
                cos_vec = cos_complex(vec, vec_std)

                if abs(cos_vec) > abs(cos_t):
                    buf.append(seg)
                else:
                    res = merge_buf(buf)
                    new_segs.append(res)
                    buf = [seg]
            else:
                eff_std = implicitize_bezier_curve(std.start / w, std.control / w, std.end / w, norm=True)
                eff = implicitize_bezier_curve(seg.start / w, seg.control / w, seg.end / w, norm=True)
                err = min(np.linalg.norm(eff_std - eff), np.linalg.norm(eff_std + eff))
                if err < eff_threshold:
                    buf.append(seg)
                else:
                    res = merge_buf(buf)
                    new_segs.append(res)
                    buf = [seg]
    
    if len(buf) > 0:
        res = merge_buf(buf)
        new_segs.append(res)
    
    return new_segs

def svg_file_simplification(svg_path, dst_path, group=True, quad_line=True, merge=True, split=True):
    '''
    svg_path: [path]
    path may contain multiple isolated paths
    '''
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    assert(len(paths) == 1)
    l = len(paths[0])
    iso_paths = paths[0].continuous_subpaths()
    w, h = get_wh(svg_attributes=svg_attributes)
    assert(w == h and w == 256)

    new_path = Path()
    for path in iso_paths:
        if group:
            path = group_near_points(path, size=(w, h))
        if path is not None:
            if quad_line:
                path = quad_to_line(path)
            if merge:
                path = merge_segments(path, size=(w, h))
            if split:
                path = split_segments(path, size=(w,h))
            new_path += Path(*path)
    
    if len(new_path) > 0:
        wsvg(paths=[new_path], attributes=attributes, svg_attributes=svg_attributes, forceZ=True, filename=dst_path)
        return True
    else:
        return False