import torch
import numpy as np
from PIL import Image, ImageDraw
import random

def random_circle_mask(size=128, min_radius=10, max_radius=40):
    """
    Genera una máscara binaria con un círculo negro (0) en una zona aleatoria.
    size: (h, w)
    """
    if isinstance(size, int):
        h = w = size
    else:
        h, w = size
    mask = Image.new("L", (w, h), 255)  # blanco = 255
    draw = ImageDraw.Draw(mask)

    # Elegir centro y radio
    r = random.randint(min_radius, max_radius)
    cx = random.randint(r, w - r)
    cy = random.randint(r, h - r)

    # Dibujar círculo negro (zona a tapar)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)

    # Convertir a tensor [0,1]
    mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0) # (1, h, w)
    return mask

def apply_mask(img_tensor, mask_tensor):
    """
    Aplica una máscara (1 visible / 0 tapado) a una imagen [C,H,W].
    """
    return img_tensor * mask_tensor  # broadcast automático
