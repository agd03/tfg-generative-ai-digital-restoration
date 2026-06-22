import os, glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from mask_utils_soft import random_circle_mask


class InpaintingImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        size: int = 128,
        min_radius: int = 12,
        max_radius: int = 40,
        mask_blur_radius=0.0,
        augment: bool = True,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p_hflip: float = 0.5,
        split: str = "train",      # "train" | "val" | "test"
        mask_seed: int = 1234,     # para val/test deterministas
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split debe ser 'train', 'val' o 'test' (recibido: {split})")

        self.split = split
        self.mask_seed = int(mask_seed)

        # Para reproducibilidad: en val/test NO queremos augmentation.
        if self.split in ("val", "test"):
            augment = False
        self.augment = augment

        # Extensiones coherentes con data_split.py
        from data_split import list_image_paths
        self.paths = list_image_paths(root)

        assert len(self.paths) > 0, f"No hay imágenes en {root}"

        self.size = size
        self.min_r = min_radius
        self.max_r = max_radius
        self.mask_blur_radius = float(mask_blur_radius)

        # Augmentation / resize en PIL
        if self.augment:
            self.pil_tf = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=(size, size),
                    scale=scale,
                    ratio=ratio,
                    interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=p_hflip),
            ])
        else:
            self.pil_tf = transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # PIL -> PIL (augmentation/resize)
        img = self.pil_tf(img)

        # PIL -> tensor en [-1,1]
        gt = self.to_tensor(img)

        # Máscara determinista en val/test
        rng = None
        if self.split in ("val", "test"):
            rng = random.Random(self.mask_seed + idx)

        mask = random_circle_mask(
            size=(self.size, self.size),
            min_radius=self.min_r,
            max_radius=self.max_r,
            rng=rng,
            blur_radius=self.mask_blur_radius
        )  # (1,H,W), 1 visible / 0 hueco

        masked = gt * mask
        cond = torch.cat([masked, mask], dim=0)
        return cond, gt, mask
