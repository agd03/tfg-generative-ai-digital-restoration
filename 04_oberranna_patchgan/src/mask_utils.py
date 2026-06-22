import torch
import numpy as np
from PIL import Image, ImageDraw
import random

def random_circle_mask(size=128, min_radius=10, max_radius=40, rng=None):
    """
    Genera una máscara binaria con un círculo negro (0) en una zona aleatoria.
    - size: int o (h, w)
    - rng: objeto con .randint(a,b) (p.ej. random.Random(seed)).
           Si es None, usa el RNG global (comportamiento actual).
    """
    if rng is None:
        rng = random  # módulo random (global)

    if isinstance(size, int):
        h = w = size
    else:
        h, w = size

    mask = Image.new("L", (w, h), 255)  # blanco = 255
    draw = ImageDraw.Draw(mask)

    # Elegir centro y radio (determinista si rng lo es)
    max_r_allowed = min(max_radius, (w - 1)//2, (h - 1)//2)
    min_r_allowed = min(min_radius, max_r_allowed)
    r = rng.randint(min_r_allowed, max_r_allowed)

    # r = min(r, (w - 1)//2, (h - 1)//2)

    cx = rng.randint(r, w - r)
    cy = rng.randint(r, h - r)

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)

    mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)  # (1,h,w)
    return mask


def apply_mask(img_tensor, mask_tensor):
    """
    Aplica una máscara (1 visible / 0 tapado) a una imagen [C,H,W].
    """
    return img_tensor * mask_tensor  # broadcast automático
