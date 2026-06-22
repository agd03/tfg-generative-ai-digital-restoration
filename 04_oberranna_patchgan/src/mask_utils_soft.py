import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

def random_circle_mask(
    size=128,
    min_radius=10,
    max_radius=40,
    rng=None,
    blur_radius: float = 0.0
):
    """Genera una máscara de inpainting con un círculo (hueco).

    Convención:
      - 1 = visible (contexto)
      - 0 = hueco (zona a inpaint)

    Args:
      size: int o (h, w)
      min_radius / max_radius: radio del círculo
      rng: objeto con .randint(a,b) (p.ej. random.Random(seed)); si es None usa random global
      blur_radius: si > 0, aplica feathering gaussiano (máscara suave en [0,1]).
                  Ojo: el interior del hueco seguirá cerca de 0, pero el borde será degradado.

    Returns:
      mask: torch.FloatTensor (1, H, W) en [0,1]
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

    # Feathering opcional
    if blur_radius and blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)  # (1,h,w)
    return mask


def apply_mask(img_tensor, mask_tensor):
    """
    Aplica una máscara (1 visible / 0 tapado) a una imagen [C,H,W].
    """
    return img_tensor * mask_tensor  # broadcast automático
