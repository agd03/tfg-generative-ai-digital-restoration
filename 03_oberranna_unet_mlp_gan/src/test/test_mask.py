import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mask_utils import random_circle_mask, apply_mask

# --- Crear carpeta de salida ---
os.makedirs("tests/masks", exist_ok=True)

# --- Cargar imagen de ejemplo ---
img_path = "data/afhq_cats/cat_00001.jpg"  # cambia si tu dataset está en otro sitio
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img_t = transform(img)  # [3,128,128]

# --- Generar máscara circular ---
mask = random_circle_mask(size=(128, 128))
masked_img = apply_mask(img_t, mask)

# --- Convertir a formato PIL para guardar ---
def tensor_to_pil(t):
    t = (t * 255).clamp(0,255).byte()
    return transforms.ToPILImage()(t)

img_pil = tensor_to_pil(img_t)
mask_pil = transforms.ToPILImage()(mask)
masked_pil = tensor_to_pil(masked_img)

# --- Guardar imágenes ---
img_pil.save("tests/masks/original.png")
mask_pil.save("tests/masks/mask.png")
masked_pil.save("tests/masks/masked.png")

print("✅ Imágenes guardadas en tests/masks/:")
print("   - original.png")
print("   - mask.png")
print("   - masked.png")
