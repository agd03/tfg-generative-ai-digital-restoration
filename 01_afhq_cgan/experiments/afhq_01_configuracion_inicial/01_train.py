import os
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from inpaint_dataset import InpaintingCatsDataset
from models_01 import InpaintUNet, InpaintDiscriminator
from mask_utils import random_circle_mask, apply_mask
from utils import weights_init_dcgan


# --------------------------------------------------
# Configuración reconstruida para exp01
# --------------------------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.set_device(0)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100

LR_G = 5e-4
LR_D = 1e-4

LAMBDA_ADV = 0.01
LAMBDA_HOLE = 6.0
LAMBDA_VALID = 1.0

BASE_CH = 64
DROPOUT = 0.3

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/{run_id}"

os.makedirs(f"{run_dir}/samples", exist_ok=True)
os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)


# --------------------------------------------------
# Guardar configuración
# --------------------------------------------------

with open(f"{run_dir}/config.txt", "w", encoding="utf-8") as f:
    f.write(f"""Fecha: {run_id}
DEVICE={DEVICE}
IMAGE_SIZE={IMG_SIZE}
BATCH_SIZE={BATCH_SIZE}
EPOCHS={EPOCHS}
LR_G={LR_G}
LR_D={LR_D}
LAMBDA_ADV={LAMBDA_ADV}
LAMBDA_HOLE={LAMBDA_HOLE}
LAMBDA_VALID={LAMBDA_VALID}
BASE_CH={BASE_CH}
DROPOUT={DROPOUT}
Discriminator=condicional (cond[4ch] + img[3ch] → 7ch)
Loss=LSGAN (MSE) + L1(hole, valid)
""")


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = InpaintingCatsDataset(
    root="data/afhq_cats",
    size=IMG_SIZE,
    min_radius=12,
    max_radius=40,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


# --------------------------------------------------
# Modelos
# --------------------------------------------------

G = InpaintUNet(
    in_ch=4,
    out_ch=3,
    base=BASE_CH,
    dropout=DROPOUT,
).to(DEVICE)

D = InpaintDiscriminator(
    in_ch_cond=4,
    in_ch_img=3,
    base=BASE_CH,
    dropout=DROPOUT,
).to(DEVICE)

G.apply(weights_init_dcgan)
D.apply(weights_init_dcgan)


# --------------------------------------------------
# Optimizadores
# --------------------------------------------------

opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))


# --------------------------------------------------
# Logging
# --------------------------------------------------

log = {
    "loss_G": [],
    "loss_D": [],
    "L1_hole": [],
    "L1_valid": [],
}


def plot_losses(save_path):
    plt.figure(figsize=(10, 5))
    plt.title("Evolución de pérdidas durante el entrenamiento de Inpainting-GAN")
    plt.plot(log["loss_D"], label="Discriminator Loss", color="red")
    plt.plot(log["loss_G"], label="Generator Loss", color="blue")
    plt.plot(log["L1_hole"], label="L1 Hole", color="green")
    plt.plot(log["L1_valid"], label="L1 Valid", color="orange")

    # En el experimento original este eje estaba etiquetado como época,
    # aunque realmente se estaba registrando por iteración/batch.
    plt.xlabel("Época")
    plt.ylabel("Pérdida media")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------------------------------------
# Batch fijo para visualización
# --------------------------------------------------

fixed_data = next(iter(loader))
_, fixed_gt, _ = fixed_data
fixed_batch = fixed_gt[:8].to(DEVICE)


# --------------------------------------------------
# Entrenamiento
# --------------------------------------------------

print("Entrenando Inpainting-GAN exp01...")

for epoch in range(1, EPOCHS + 1):
    G.train()
    D.train()

    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_l1_hole = 0.0
    epoch_l1_valid = 0.0
    steps = 0

    loop = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)

    for cond, gt, mask in loop:
        cond = cond.to(DEVICE).float()
        gt = gt.to(DEVICE).float()
        mask = mask.to(DEVICE).float()

        # -------------------------
        # Forward generador
        # -------------------------
        pred = G(cond)
        comp_fake = pred * (1 - mask) + gt * mask

        # -------------------------
        # Entrenar discriminador
        # -------------------------
        opt_D.zero_grad(set_to_none=True)

        D_real = D(cond, gt)
        D_fake = D(cond, comp_fake.detach())

        loss_D = 0.5 * (
            (D_real - 1).pow(2).mean()
            + D_fake.pow(2).mean()
        )

        loss_D.backward()
        opt_D.step()

        # -------------------------
        # Entrenar generador
        # -------------------------
        opt_G.zero_grad(set_to_none=True)

        D_fake_for_G = D(cond, comp_fake)

        mask_d = F.interpolate(
            mask,
            size=D_fake_for_G.shape[-2:],
            mode="nearest",
        )
        hole_d = 1.0 - mask_d

        adv_elem = (D_fake_for_G - 1).pow(2)
        adv_loss = (adv_elem * hole_d).sum() / hole_d.sum().clamp_min(1.0)

        l1_hole = F.l1_loss(pred * (1 - mask), gt * (1 - mask))
        l1_valid = F.l1_loss(pred * mask, gt * mask)

        loss_G = (
            LAMBDA_ADV * adv_loss
            + LAMBDA_HOLE * l1_hole
            + LAMBDA_VALID * l1_valid
        )

        loss_G.backward()
        opt_G.step()

        # -------------------------
        # Logging por batch
        # -------------------------
        steps += 1
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        epoch_l1_hole += l1_hole.item()
        epoch_l1_valid += l1_valid.item()

        log["loss_G"].append(epoch_loss_G / steps)
        log["loss_D"].append(epoch_loss_D / steps)
        log["L1_hole"].append(epoch_l1_hole / steps)
        log["L1_valid"].append(epoch_l1_valid / steps)

        loop.set_postfix({
            "loss_D": f"{loss_D.item():.3f}",
            "loss_G": f"{loss_G.item():.3f}",
        })

    # --------------------------------------------------
    # Samples visuales
    # --------------------------------------------------

    G.eval()
    with torch.no_grad():
        b = fixed_batch.size(0)

        mask_f = torch.stack([
            random_circle_mask(
                IMG_SIZE,
                min_radius=12,
                max_radius=40,
            )
            for _ in range(b)
        ]).to(DEVICE)

        cond_rgb = apply_mask(fixed_batch, mask_f)
        cond_f = torch.cat([cond_rgb, mask_f], dim=1)

        pred_f = G(cond_f)
        comp_f = pred_f * (1 - mask_f) + fixed_batch * mask_f

        grid = torch.cat([
            cond_f[:, :3],
            mask_f.repeat(1, 3, 1, 1) * 2 - 1,
            pred_f,
            comp_f,
            fixed_batch,
        ], dim=0)

        save_image(
            grid,
            f"{run_dir}/samples/epoch_{epoch:03d}.png",
            nrow=b,
            normalize=True,
            value_range=(-1, 1),
        )

    # --------------------------------------------------
    # Guardados
    # --------------------------------------------------

    plot_losses(f"{run_dir}/losses_plot.png")
    pd.DataFrame(log).to_csv(f"{run_dir}/training_log.csv", index=False)

    if epoch % 20 == 0 or epoch == EPOCHS:
        torch.save(G.state_dict(), f"{run_dir}/checkpoints/G_epoch{epoch:03d}.pth")
        torch.save(D.state_dict(), f"{run_dir}/checkpoints/D_epoch{epoch:03d}.pth")

    print(
        f"[{epoch:03d}/{EPOCHS}] "
        f"D: {log['loss_D'][-1]:.3f} | "
        f"G: {log['loss_G'][-1]:.3f} | "
        f"L1(h): {log['L1_hole'][-1]:.3f} | "
        f"L1(v): {log['L1_valid'][-1]:.3f}"
    )

print(f"Listo. Resultados en: {run_dir}")
