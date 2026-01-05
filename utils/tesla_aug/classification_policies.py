import itertools
import math
import torch

from .img_ops.autoaugment import (
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
    Contrast,
    Color,
    Brightness,
    Sharpness,
)

all_augmentations = [
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
    Contrast,
    Color,
    Brightness,
    Sharpness,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
]

IMG_OPS = {
    "Identity": [0.0, 1.0],
    "ShearX": [-0.3, 0.3],
    "ShearY": [-0.3, 0.3],
    "TranslateX": [-0.45, 0.45],
    "TranslateY": [-0.45, 0.45],
    "Rotate": [-math.pi / 6, -math.pi / 6],
    "AutoContrast": [0, 1],
    "Invert": [0, 1],
    "Equalize": [0, 1],
    "Solarize": [0, 1],
    "Posterize": [4, 8],
    "Contrast": [0.1, 1.9],
    "Color": [0.1, 1.9],
    "Brightness": [0.1, 1.9],
    "Sharpness": [0.1, 1.9],
    "GaussianBlur": [0.5, 2.0],
}


def get_sub_policies(n_aug=-1):
    all_augmentations_index = list(range(len(all_augmentations)))
    if n_aug == -1:
        return list(
            itertools.chain.from_iterable(
                itertools.combinations(all_augmentations_index, n)
                for n in range(1, len(all_augmentations_index) + 1)
            )
        )
    return list(itertools.combinations(all_augmentations_index, n_aug))


def apply_augment(x, fn_idx, mag):
    x = torch.clamp(x, 0.0, 1.0)
    fn = all_augmentations[fn_idx]
    min_val, max_val = IMG_OPS[fn.__name__]
    v = min_val + mag * (max_val - min_val)
    out = fn(x, v)
    return torch.clamp(out, 0.0, 1.0)
