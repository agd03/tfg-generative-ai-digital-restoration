# inpaint_dataset.py
import os, glob, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from mask_utils import random_circle_mask  # ya lo tienes

class InpaintingCatsDataset(Dataset):
    """
    Dataset para la tarea de inpainting:
      - Carga imágenes de gatos desde 'root'
      - Genera una máscara circular aleatoria por imagen
      - Devuelve:
          cond = [imagen_enmascarada, máscara]  -> (4, H, W)
          gt   = imagen original (ground truth) -> (3, H, W)
          mask = máscara binaria                -> (1, H, W)
    """
    def __init__(self, root="data/afhq_cats", size=128,
                 min_radius=12, max_radius=40):
        self.paths = sorted(
            [p for p in glob.glob(os.path.join(root, "*.jpg"))]
            + [p for p in glob.glob(os.path.join(root, "*.png"))]
        )
        assert len(self.paths) > 0, f"No hay imágenes en {root}"

        self.size = size
        self.min_r = min_radius
        self.max_r = max_radius

        self.to_tensor = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),                      # [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3),     # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        gt = self.to_tensor(img)                       # (3,H,W) en [-1,1]

        # máscara 1=visible, 0=agujero
        mask = random_circle_mask(size=(self.size, self.size),
                                  min_radius=self.min_r, max_radius=self.max_r)  # (1,H,W) en [0,1]

        # rellenamos agujero con 0 (que en [-1,1] equivale a gris medio tras denormalizar, pero está ok)
        masked = gt * mask

        # concatenamos máscara como 4º canal para el generador
        # entrada condicional = [imagen_enmascarada (3), máscara (1)] => (4,H,W)
        cond = torch.cat([masked, mask], dim=0)

        return cond, gt, mask
