# test_inpaint_dataset.py
import torch
from torch.utils.data import DataLoader
from inpaint_dataset import InpaintingCatsDataset
from model_inpainting import UNET

def main():
    ds = InpaintingCatsDataset(root="data/afhq_cats", size=128)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    cond, gt, mask = next(iter(loader))
    print("cond:", cond.shape)   # esperado: (B, 4, 128, 128)
    print("gt:  ", gt.shape)     # esperado: (B, 3, 128, 128)
    print("mask:", mask.shape)   # esperado: (B, 1, 128, 128)

    model = UNET(in_channels=4, out_channels=3)
    preds = model(cond)
    print("preds:", preds.shape) # esperado: (B, 3, 128, 128)

    assert preds.shape == gt.shape, "Salida del generador no coincide con gt"

if __name__ == "__main__":
    main()
