"""
Differentiable Torch Implementation of AutoAugment Image Functions.
"""

import torch
import torchvision as tv

from .constants import GAUSSIAN_KERNEL, SMOOTH_KERNEL

FLIP = True


def rgb_to_gray(x):
    return 0.2989 * x[:, 0:1, ...] + 0.5870 * x[:, 1:2, ...] + 0.1140 * x[:, 2:3, ...]


def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def apply_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.size(), align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return out


def blend(img1, img2, factor):
    factor = factor.reshape(-1, 1, 1, 1)
    diff = img2 - img1
    scaled = factor * diff
    tmp = img1 + scaled
    return torch.clamp(tmp, 0.0, 1.0)


def ShearX(x, v):
    if FLIP:
        rand_flip = torch.rand_like(v)
        v[rand_flip < 0.5] = -v[rand_flip < 0.5]

    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    shear_x = v.reshape(-1, 1, 1) * torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    return apply_affine(x, identity + shear_x)


def ShearY(x, v):
    if FLIP:
        rand_flip = torch.rand_like(v)
        v[rand_flip < 0.5] = -v[rand_flip < 0.5]

    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    shear_y = v.reshape(-1, 1, 1) * torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    return apply_affine(x, identity + shear_y)


def TranslateX(x, v):
    if FLIP:
        rand_flip = torch.rand_like(v)
        v[rand_flip < 0.5] = -v[rand_flip < 0.5]

    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    translate_x = v.reshape(-1, 1, 1) * torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    return apply_affine(x, identity + translate_x)


def TranslateY(x, v):
    if FLIP:
        rand_flip = torch.rand_like(v)
        v[rand_flip < 0.5] = -v[rand_flip < 0.5]

    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    translate_y = v.reshape(-1, 1, 1) * torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    return apply_affine(x, identity + translate_y)


def Rotate(x, v):
    if FLIP:
        rand_flip = torch.rand_like(v)
        v[rand_flip < 0.5] = -v[rand_flip < 0.5]

    cos_affine = torch.cos(v).view(-1, 1, 1) * torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    sin_affine = torch.sin(v).view(-1, 1, 1) * torch.tensor([0, 1, 0, -1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    return apply_affine(x, cos_affine + sin_affine)


def AutoContrast(x, v):
    out = tv.transforms.functional.autocontrast(x.detach())
    v = v.reshape(-1, 1, 1, 1)
    return out + x - x.detach() + v - v.detach()


def Equalize(x, v):
    out = (255.0 * x).to(torch.uint8)
    out = tv.transforms.functional.equalize(out)
    out = out.to(torch.float32) / 255.0
    v = v.reshape(-1, 1, 1, 1)
    return out + x - x.detach() + v - v.detach()


def Invert(x, v):
    out = 1.0 - x
    v = v.reshape(-1, 1, 1, 1)
    return out + v - v.detach()


def Solarize(x, v):
    x_out = torch.where(x < v.reshape(-1, 1, 1, 1), x, 1.0 - x)
    v = v.reshape(-1, 1, 1, 1)
    return x_out + v - v.detach() + x - x.detach()


def Posterize(x, v):
    x_uint = (255.0 * x).to(torch.uint8)
    shift = v.to(torch.int64)

    x_out = x_uint.clone()
    for i, (sample_x, sample_shift) in enumerate(zip(x_uint, shift)):
        x_out[i] = tv.transforms.functional.posterize(sample_x, sample_shift)

    x_out = x_out.to(torch.float32) / 255.0
    v = v.reshape(-1, 1, 1, 1)
    return x_out + x - x.detach() + v - v.detach()


def Contrast(x, v):
    mean_img = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    degenerate = torch.zeros_like(x) + mean_img
    return blend(degenerate, x, v)


def GaussianBlur(x, v):
    v = v.reshape(-1, 1, 1, 1, 1)
    gauss_kernel = GAUSSIAN_KERNEL.to(x.device)[None, ...] ** (1 / (v ** 2))
    w = gauss_kernel / gauss_kernel.sum(dim=(1, 2, 3, 4), keepdim=True)

    x = torch.nn.functional.pad(x, (3, 3, 3, 3), mode="replicate")

    batch_size = x.size(0)
    n_ch = x.size(1)
    x = x.permute(1, 0, 2, 3)

    o = torch.nn.functional.conv2d(
        x.view(n_ch, batch_size * 1, x.size(2), x.size(3)),
        w.view(batch_size * 1, 1, w.size(3), w.size(4)),
        groups=batch_size,
    )

    return o.permute(1, 0, 2, 3)


def Brightness(x, v):
    degenerate = torch.zeros_like(x)
    return blend(degenerate, x, v)


def Sharpness(x, v):
    n_ch = x.size(1)
    kernel = SMOOTH_KERNEL.to(x.device).repeat(n_ch, 1, 1, 1)
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    degenerate = torch.nn.functional.conv2d(x_pad, kernel, groups=n_ch)
    return blend(degenerate, x, v)


def Color(x, v):
    degenerate = gray_to_rgb(rgb_to_gray(x))
    return blend(degenerate, x, v)


def Identity(x, v):
    return x
