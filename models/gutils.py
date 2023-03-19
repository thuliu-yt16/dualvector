import numpy as np
import cairosvg
import torch
import math

def surface_to_npim(surface):
    """ Transforms a Cairo surface into a numpy array. """
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H,W = surface.get_height(), surface.get_width()
    im.shape = (H,W,4) # for RGBA
    return im

def svg_to_npim(svg_bytestring, w, h):
    """ Renders a svg bytestring as a RGB image in a numpy array """
    tree = cairosvg.parser.Tree(bytestring=svg_bytestring)
    surf = cairosvg.surface.PNGSurface(tree,None,50,output_width=w, output_height=h).cairo
    return surface_to_npim(surf)

def render_bezier_path(curves, sidelength):
    '''
    curves: c (pts*2)
    '''

    curves = (curves + 1) * sidelength / 2
    d = path_d_from_control_points(curves, xy_flip=False)

    svg_string = f'''<?xml version="1.0" ?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="200" version="1.1" viewBox="0 0 {sidelength} {sidelength}" width="200">
    <defs/>
    <path d="{d}" fill="black" stroke="none"/>
</svg>'''
    im = svg_to_npim(svg_string.encode('utf-8'), sidelength, sidelength)
    return im[:, :, 3]

def render_multi_bezier_paths(paths, sidelength):
    '''
    paths: a list: [c (pts*2)]
    '''

    d = []
    for curve in paths:
        curve = (curve + 1) * sidelength / 2
        dstr = path_d_from_control_points(curve, xy_flip=False)
        d.append(f'''<path d="{dstr}" fill="black" stroke="none"/>''')

    svg_string = f'''<?xml version="1.0" ?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="200" version="1.1" viewBox="0 0 {sidelength} {sidelength}" width="200">
    <defs/>''' +  '\n'.join(d) + '</svg>'
    im = svg_to_npim(svg_string.encode('utf-8'), sidelength, sidelength)
    return im[:, :, 3]


def render_bezier_paths(paths, sidelength):
    '''
    render in the same path tag
    paths: a list:  [c (pts*2)]
    '''

    d = []
    for curve in paths:
        curve = (curve + 1) * sidelength / 2
        d.append(path_d_from_control_points(curve, xy_flip=False))
    
    d = ' '.join(d)

    svg_string = f'''<?xml version="1.0" ?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="200" version="1.1" viewBox="0 0 {sidelength} {sidelength}" width="200">
    <defs/>
    <path d="{d}" fill="black" stroke="none"/>
</svg>'''

    im = svg_to_npim(svg_string.encode('utf-8'), sidelength, sidelength)
    return im[:, :, 3]


def path_d_from_control_points(cp, xy_flip=True):
    if isinstance(cp, torch.Tensor):
        cp = cp.detach().cpu().numpy()

    # cp: n_curves (cps 2)
    n, n_cp = cp.shape
    n_cp = n_cp // 2

    assert(n_cp == 3 or n_cp == 4)

    cc = 'C' if n_cp == 4 else 'Q'
    
    d = []
    for i in range(n):
        if i == 0:
            d += ['M']
            if xy_flip:
                d += list(map(str, [cp[i, 1], cp[i, 0]]))
            else:
                d += list(map(str, [cp[i, 0], cp[i, 1]]))
        d += [cc]
        if n_cp == 4:
            if xy_flip:
                d += list(map(str, [cp[i, 3], cp[i, 2], cp[i, 5], cp[i, 4], cp[i, 7], cp[i, 6]]))
            else:
                d += list(map(str, cp[i, 2:8]))
        
        else:
            if xy_flip:
                d += list(map(str, [cp[i, 3], cp[i, 2], cp[i, 5], cp[i, 4]]))
            else:
                d += list(map(str, cp[i, 2:6]))

    d += ['Z']
    d_str = ' '.join(d)
    return d_str

def antialias_kernel(r):
    r = -r
    output = (0.5 + 0.25 * (torch.pow(r, 3) - 3 * r))
    #   output = -0.5*r + 0.5
    return output

def render_sdf(sdf, resolution):
    normalization = resolution  # 0.70710678*2/resolution
    normalized_sdf = sdf / normalization
    clamped_sdf = torch.clamp(normalized_sdf, min=-1, max=1)
    opacity = antialias_kernel(clamped_sdf)  # multiply by color here
    return opacity


def gradient(y, x, create_graph=True, allow_unused=False):
    return torch.autograd.grad(y, [x], create_graph=create_graph, grad_outputs=torch.ones_like(y), allow_unused=allow_unused)[0]

class SolveCubicNumpy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.set_materialize_grads(False)
        assert(inp.shape[1] == 4)
        inp_np = inp.cpu().numpy()
        n = inp_np.shape[0]
        roots = np.zeros([n, 3], dtype=inp_np.dtype)

        def find_root(i):
            r = np.roots(inp_np[i])
            # r.real[np.abs(r.imag)<1e-5] 
            r = r.real[np.isreal(r)]
            roots[i, :] = r

        for i in range(n):
            find_root(i)
        
        roots = torch.from_numpy(roots).type_as(inp)
        ctx.save_for_backward(inp, roots)
        return roots

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        
        inp, roots = ctx.saved_tensors

        a = inp[:, 0:1]
        b = inp[:, 1:2]
        c = inp[:, 2:3]

        grad_inp = torch.zeros_like(inp)

        for i in range(3):
            t = roots[:, i:(i+1)]
            dp_dt = t * (3 * a * t + 2 * b) + c
            dp_dd = torch.ones_like(dp_dt)
            dp_dc = t
            dp_db = dp_dc * t
            dp_da = dp_db * t
            dp_dC = torch.cat([dp_da, dp_db, dp_dc, dp_dd], dim=-1)
            dt_dC = - torch.sign(dp_dt) / torch.clamp(torch.abs(dp_dt), min=1e-5) * dp_dC 

            grad_inp += dt_dC * grad_output[:, i:(i + 1)]

        return grad_inp

def rdot(a, b): # row-wise dot
    return torch.sum(a * b, dim=-1).unsqueeze(-1)

def vec_norm(a):
    # in pytorch 1.6.0
    return a.norm(dim=-1, keepdim=True)


def sd_bezier(A, B, C, p):
    # def rdot(a, b): # row-wise dot
    #     return torch.sum(a * b, dim=-1).unsqueeze(-1)
    
    # s = abs(torch.sign(B * 2.0 - A - C))
    # B = (B + 1e-4) * (1 - s) + s * B
    
    # a = B - A
    # b = A - B * 2.0 + C
    # c = a * 2.0
    # d = A - p
    # k = torch.cat([3.*rdot(a,b), 2.*rdot(a,a)+rdot(d,b),rdot(d,a)], dim=-1) / torch.clamp(rdot(b,b), min=1e-4)     
    # t = torch.clamp(solve_cubic(k), 0.0, 1.0)
    # t = t.unsqueeze(1)

    # # n x 2 x 3
    # vec = A.unsqueeze(-1) + (c.unsqueeze(-1) + b.unsqueeze(-1) * t) * t - p.unsqueeze(-1)
    # dis = torch.min(torch.linalg.norm(vec, dim=1), dim=-1, keepdim=True)[0]

    # return dis
    return _sd_bezier(A, B, C, p)

def _sd_bezier(A, B, C, p):
    def rdot(a, b): # row-wise dot
        return torch.sum(a * b, dim=-1).unsqueeze(-1)
    
    s = abs(torch.sign(B * 2.0 - A - C))
    B = (B + 1e-4) * (1 - s) + s * B
    
    a = B - A
    b = A - B * 2.0 + C
    c = a * 2.0
    d = A - p

    bdb = torch.clamp(rdot(b, b), min=1e-4)
    # k = torch.cat([3.*rdot(a,b), 2.*rdot(a,a)+rdot(d,b),rdot(d,a)], dim=-1) / torch.clamp(rdot(b,b), min=1e-4)     
    t = torch.clamp(solve_cubic(3.*rdot(a, b) / bdb, (2.*rdot(a,a)+rdot(d,b))/bdb, rdot(d,a)/bdb), 0.0, 1.0)
    t = t.unsqueeze(1)

    # n x 2 x 3
    vec = A.unsqueeze(-1) + (c.unsqueeze(-1) + b.unsqueeze(-1) * t) * t - p.unsqueeze(-1)
    if hasattr(torch, 'linalg'):
        dis = torch.min(torch.linalg.norm(vec, dim=1), dim=-1, keepdim=True)[0]
    else:
        dis = torch.min(torch.norm(vec, dim=1), dim=-1, keepdim=True)[0]

    return dis

def _sd_bezier_np(A, B, C, p):

    solve_cubic_np = SolveCubicNumpy.apply
    
    s = abs(torch.sign(B * 2.0 - A - C))
    B = (B + 1e-4) * (1 - s) + s * B
    
    a = B - A
    b = A - B * 2.0 + C
    c = a * 2.0
    d = A - p

    # k = torch.cat([3.*rdot(a,b), 2.*rdot(a,a)+rdot(d,b),rdot(d,a)], dim=-1) / torch.clamp(rdot(b,b), min=1e-4)     

    k = torch.cat([rdot(b,b), 3.*rdot(a,b), 2.*rdot(a,a)+rdot(d,b), rdot(d,a)], dim=-1)
    t = torch.clamp(solve_cubic_np(k), 0.0, 1.0)
    t = t.unsqueeze(1)

    # n x 2 x 3
    vec = A.unsqueeze(-1) + (c.unsqueeze(-1) + b.unsqueeze(-1) * t) * t - p.unsqueeze(-1)
    dis = torch.min(torch.linalg.norm(vec, dim=1), dim=-1, keepdim=True)[0]

    return dis

# copy from https://members.loria.fr/SHornus/quadratic-arc-length.html
def bezier_length(a, b, c):
    '''
    a, b, c: n x 2
    '''
    A = a + c - 2 * b
    B = b - a
    C = a
    F = A + B

    A_norm = vec_norm(A)
    B_norm = vec_norm(B)
    F_norm = vec_norm(F)
    AB_dot = rdot(A, B)
    AF_dot = rdot(A, F)

    A_norm_clamp = torch.clamp(A_norm, min=1e-8)

    l = (F_norm * AF_dot - B_norm * AB_dot) / A_norm_clamp.pow(2) + \
        (A_norm.pow(2) * B_norm.pow(2) - AB_dot.pow(2)) / A_norm_clamp.pow(3) * \
        (torch.log(torch.clamp(A_norm * F_norm + AF_dot, min=1e-8)) - torch.log(torch.clamp(A_norm * B_norm + AB_dot, min=1e-8)))
    
    return l

# copied from https://www.shadertoy.com/view/ltXSDB by Inigo Quilez
def solve_cubic(a, b, c):
    '''
    abc: n x 3
    '''
    # a = abc[:, 0:1]
    # b = abc[:, 1:2]
    # c = abc[:, 2:3]
    p = b - a*a / 3.
    p3 = torch.pow(p, 3)
    q = a * (2.0*a*a - 9.0*b) / 27.0 + c
    d = q*q + 4.0*p3 / 27.0
    offset = -a / 3.0

    d_mask = (d >= 0)
    z = torch.sqrt(torch.clamp(d[d_mask], min=1e-10)).unsqueeze(-1)

    x = (torch.cat([z, -z], dim=-1) - q[d_mask].unsqueeze(-1)) / 2.0

    uv = torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=1e-10), 1.0/3)

    root1 = (offset[d_mask].unsqueeze(1) + uv[:, 0:1] + uv[:, 1:2]).repeat(1, 3)

    to_acos = torch.clamp(-torch.sqrt(-27.0 / torch.clamp(p3[~d_mask], max=-1e-8)) * q[~d_mask] / 2.0, -1 + 1e-4, 1 - 1e-4)
    v = torch.acos(to_acos) / 3.0
    m = torch.cos(v).unsqueeze(-1)
    n = torch.sin(v).unsqueeze(-1) * math.sqrt(3)
    root2 = torch.cat([m + m, -n - m, n - m], dim=-1) \
            * torch.sqrt(torch.clamp(-p[~d_mask].unsqueeze(-1) / 3.0, min=1e-10)) \
             + offset[~d_mask].unsqueeze(-1)

    root = torch.zeros((a.shape[0], 3), device=a.device)
    root[d_mask.repeat(1,3)] = root1.flatten()
    root[~d_mask.repeat(1,3)] = root2.flatten()

    return root

def solve_cubic_order2_zero(a_, b_, c_):
    def cubic_root(p):
        return p.sign() * p.abs().pow(1.0/3)

    # ax^3 + bx + c = 0
    # a is almost zero
    a_zero_mask = a_.abs() < 1e-8
    # a ~= 0
    root = torch.zeros((a_.shape[0], 3), device=a_.device)
    root[a_zero_mask.repeat(1,3)] = (- c_ / b_)[a_zero_mask].repeat(1, 3)

    a = a_
    c = b_
    d = c_
    A = -3*a*c
    B = -9*a*d
    C = c**2

    delta = B**2 - 4*A*C
    # delta > 0
    d_mask = (delta > 0) & ~a_zero_mask
    d_sqrt = torch.sqrt(delta[d_mask])
    Y1 = 3*a[d_mask]*(-B[d_mask] - d_sqrt) / 2
    Y2 = 3*a[d_mask]*(-B[d_mask] + d_sqrt) / 2
    X1 = (-cubic_root(Y1) - cubic_root(Y2)) / (3*a[d_mask])
    root[d_mask.repeat(1,3)] = X1.repeat(1, 3)

    # delta <= 0
    d_mask = (delta <= 0) & ~a_zero_mask
    theta = torch.arccos(-3*a[d_mask]*B[d_mask]/2/torch.pow(A[d_mask], 1.5))
    m = torch.sqrt(A[d_mask]) * torch.cos(theta / 3) / 3 / a[d_mask]
    n = torch.sqrt(A[d_mask]) * torch.sin(theta / 3) / 3 / a[d_mask]
    X1 = -2 * m
    X2 = m + n
    X3 = m - n
    d_mask_s = d_mask.squeeze()
    root[d_mask_s, 0] = X1
    root[d_mask_s, 1] = X2
    root[d_mask_s, 2] = X3
    return root


def sd_parabola(params, p):
    '''
    params: n x 4, a, theta, x0, y0 
    '''
    a, theta, x0, y0 = params.split(1, dim=-1)

    ct = torch.cos(theta)
    st = torch.sin(theta)

    x, y = p.split(1, dim=-1)

    x_ = x * ct + y * st - x0 
    y_ = -x * st + y * ct - y0 
    sig = torch.sign(a * x_ ** 2 - y_)

    # distance between (x_, y_) and y = a * x ** 2
    # dis'(x) = 2a^2x^3 + (1-2an)x - m
    # dis(x) = sqrt((x - m)^2 + (x^2 - n)^2)

    # t = solve_cubic_order2_zero(2*a**2, 1-2*a*y_, -x_)

    a_zero_mask = (a.abs() < 1e-5).squeeze(dim=-1)

    t = torch.zeros((a.shape[0], 3), device=a.device)
    t1 = (x_[a_zero_mask] / (1 - 2*a[a_zero_mask]*y_[a_zero_mask]))
    t[a_zero_mask] = t1.repeat(1, 3)
    t2 = solve_cubic(torch.zeros_like(a[~a_zero_mask]), (1-2*a[~a_zero_mask]*y_[~a_zero_mask])/2/a[~a_zero_mask]**2, -x_[~a_zero_mask]/2/a[~a_zero_mask]**2)
    t[~a_zero_mask] = t2

    dis = torch.min((t - x_) ** 2 + (a * t**2 - y_) ** 2, dim=-1)[0]
    return sig, torch.sqrt(torch.clamp(dis, min=1e-10)).unsqueeze(-1)


def eval_parabola(params, p):
    '''
    params: n x 4, a, theta, x0, y0 
    '''
    a, theta, x0, y0 = params.split(1, dim=-1)

    ct = torch.cos(theta)
    st = torch.sin(theta)

    x, y = p.split(1, dim=-1)

    x_ = x * ct + y * st - x0 
    y_ = -x * st + y * ct - y0 
    # sig = torch.sign(a * x_ ** 2 - y_)

    return a * x_ ** 2 - y_
