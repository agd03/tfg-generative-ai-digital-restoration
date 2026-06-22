import os, datetime, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Mis modelos y dataset
from inpaint_dataset import InpaintingCatsDataset
#from models import UNetGenerator, Discriminator
from models_06 import InpaintUNet as UNetGenerator
from models_06 import InpaintDiscriminator as Discriminator
from dataset import AFHQCatsDataset
from utils import weights_init_dcgan, sample_noise_map, save_generated_images
from mask_utils import random_circle_mask, apply_mask

# -----------------------------
# 0) Configuración y utilidades
# -----------------------------

SEED = 42
CUDA_VISIBLE_DEVICES = 0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.set_device(0)  # GPU 0
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

# --- Hiperparámetros ---
WARMUP_EPOCHS = 5     # 3–5 suele ir bien
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 60
LR_G = 5e-4
LR_D = 1e-4             # TTUR suave: G más rápido que D
LAMBDA_ADV   = 0.01      # 2.5× más peso adversarial
LAMBDA_HOLE  = 3.0       # un poco menos, para no dominar
LAMBDA_VALID = 1.0
LAMBDA_VGG   = 0.05      # + detalle visual
DROPOUT      = 0.0       # elimina suavizado innecesario
BASE_CH      = 64       # ancho base G y D

# --- Crear carpeta única para esta sesión ---
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/{run_id}"
os.makedirs(run_dir, exist_ok=True)

# Crear directorios para guardar muestras y modelos
os.makedirs(f"{run_dir}/samples", exist_ok=True)
os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

# Guardar configuración
with open(f"{run_dir}/config.txt", "w") as f:
    f.write(f"""Fecha: {run_id}
        WARMUP_EPOCHS={WARMUP_EPOCHS}
        DEVICE={DEVICE}
        IMAGE_SIZE={IMG_SIZE}
        BATCH_SIZE={BATCH_SIZE}
        EPOCHS={EPOCHS}
        LR_G={LR_G}
        LR_D={LR_D}
        LAMBDA_ADV={LAMBDA_ADV}
        LAMBDA_HOLE={LAMBDA_HOLE}
        LAMBDA_VALID={LAMBDA_VALID}
        LAMBDA_VGG={LAMBDA_VGG}
        BASE_CH={BASE_CH}
        DROPOUT={DROPOUT}
        Discriminator=condicional (cond[4ch] + img[3ch] → 7ch)
        Loss = LSGAN (MSE) + L1(hole, valid) + perceptual(VGG16)
        """)

# --------------------------------
# 1) Dataset y DataLoader
# --------------------------------
dataset = InpaintingCatsDataset(root="data/afhq_cats", size=IMG_SIZE, min_radius=12, max_radius=40)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# --------------------------------
# 2) Modelos
# --------------------------------
G = UNetGenerator(in_ch=4, out_ch=3, base=BASE_CH, dropout=DROPOUT).to(DEVICE)
D = Discriminator(in_ch_cond=4, in_ch_img=3, base=BASE_CH).to(DEVICE)
G.apply(weights_init_dcgan)
D.apply(weights_init_dcgan)

# --------------------------------
# 3) Optimizadores + Scheduler
# --------------------------------
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

def make_linear_decay(optimizer, start_epoch, total_epochs):
    decay_span = max(1, total_epochs - start_epoch)  # evitar división por cero
    def lr_lambda(current_epoch):
        if current_epoch < start_epoch:
            return 1.0
        progress = (current_epoch - start_epoch + 1) / decay_span
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


sched_G = make_linear_decay(opt_G, start_epoch=EPOCHS//2, total_epochs=EPOCHS)
sched_D = make_linear_decay(opt_D, start_epoch=EPOCHS//2, total_epochs=EPOCHS)

# --------------------------------
# 4) Logging
# --------------------------------
log = {k: [] for k in ["loss_G","loss_D","L1_hole","L1_valid","D_real_mean","D_fake_mean","grad_G","grad_D"]}

def plot_losses(save_path):
    plt.figure(figsize=(10,5))
    plt.title("Evolución de pérdidas durante el entrenamiento de Inpainting-GAN")
    plt.plot(log["loss_D"], label="Discriminator Loss", color="red")
    plt.plot(log["loss_G"], label="Generator Loss", color="blue")
    plt.plot(log["L1_hole"], label="L1 Hole", color="green")
    plt.plot(log["L1_valid"], label="L1 Valid", color="orange")
    plt.xlabel("Época"); plt.ylabel("Pérdida media"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# Fija un batch para seguimiento visual
fixed_data = next(iter(loader))
cond_batch, gt_batch, mask_batch = fixed_data  # desempaquetar el batch
fixed_batch = gt_batch[:8].to(DEVICE)          # nos quedamos con las imágenes reales

# --------------------------------------------
# Perceptual loss usando VGG16 preentrenada
# --------------------------------------------
from torchvision.models import vgg16

# --- Cargar VGG16 preentrenada y congelar ---
vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval().to(DEVICE)
for p in vgg.parameters():
    p.requires_grad = False

# --- Normalización para VGG (ImageNet) ---
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)

def to_vgg_space(x):
    # tus tensores están en [-1,1]; pásalos a [0,1] y normaliza como ImageNet
    x01 = (x.clamp(-1, 1) + 1) / 2
    return (x01 - IMNET_MEAN) / IMNET_STD

def perceptual_loss(pred, gt):
    return F.l1_loss(vgg(to_vgg_space(pred)), vgg(to_vgg_space(gt)))


# --------------------------------
# 5) Bucle de entrenamiento
# --------------------------------
print("🚀 Entrenando Inpainting-GAN...")
for epoch in range(1, EPOCHS+1):
    loop = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)

    G.train(); D.train()
    # acumuladores de pérdidas por epoch
    epoch_loss_G = epoch_loss_D = epoch_l1_hole = epoch_l1_valid = 0.0
    epoch_D_real = epoch_D_fake = epoch_grad_G = epoch_grad_D = 0.0
    steps = 0

    for cond, gt, mask in loop:
        cond, gt, mask = cond.to(DEVICE), gt.to(DEVICE), mask.to(DEVICE)

        # --------- Forward G ---------
        pred = G(cond)                          # (B,3,H,W)
        comp_fake = pred*(1-mask) + gt*mask     # compuesta (fake)
        
        # --------- Entrena D (LSGAN) ---------
        opt_D.zero_grad(set_to_none=True)
        D_real = D(cond, gt)
        D_fake = D(cond, comp_fake.detach())
        loss_D = 0.5 * ( (D_real - 1).pow(2).mean() + (D_fake - 0).pow(2).mean() )
        loss_D.backward()
        opt_D.step()

        # --------- Entrena G ---------
        opt_G.zero_grad(set_to_none=True)
        D_fake_for_G = D(cond, comp_fake)

        # Pérdida adversarial SOLO en el hueco (redimensionamos máscara al mapa de D)
        mask_d = F.interpolate(mask, size=D_fake_for_G.shape[-2:], mode="nearest")
        hole_d = 1.0 - mask_d
        adv_elem = (D_fake_for_G - 1).pow(2)           # LSGAN element-wise (MSE)
        adv_loss = (adv_elem * hole_d).sum() / hole_d.sum().clamp_min(1.0)

        # L1 (hueco y contexto)
        l1_hole  = F.l1_loss(pred*(1-mask), gt*(1-mask))
        l1_valid = F.l1_loss(pred*mask,      gt*mask)

        # ---------- Pérdida total del Generador ----------
        if epoch <= WARMUP_EPOCHS:
            # Warm-up: solo reconstrucción L1
            loss_G = 6.0*l1_hole + 1.0*l1_valid
        else:
            # Entrenamiento completo con los nuevos pesos
            loss_G = (LAMBDA_ADV*adv_loss
                    + LAMBDA_HOLE*l1_hole
                    + LAMBDA_VALID*l1_valid
                    + LAMBDA_VGG*perceptual_loss(pred, gt))

        loss_G.backward()
        opt_G.step()

        loop.set_postfix({
            "loss_D": f"{loss_D.item():.3f}",
            "loss_G": f"{loss_G.item():.3f}"
        })

        # ---- Métricas por paso (para medias de epoch) ----
        with torch.no_grad():
            D_real_mean = torch.sigmoid(D_real).mean().item()
            D_fake_mean = torch.sigmoid(D_fake).mean().item()
        grad_G = sum((p.grad.norm().item() for p in G.parameters() if p.grad is not None)) / max(1, sum(1 for p in G.parameters() if p.grad is not None))
        grad_D = sum((p.grad.norm().item() for p in D.parameters() if p.grad is not None)) / max(1, sum(1 for p in D.parameters() if p.grad is not None))

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        epoch_l1_hole += l1_hole.item()
        epoch_l1_valid += l1_valid.item()
        epoch_D_real += D_real_mean
        epoch_D_fake += D_fake_mean
        epoch_grad_G += grad_G
        epoch_grad_D += grad_D
        steps += 1

        # ---- Medias de epoch ----
        for k, v in {
            "loss_G": epoch_loss_G/steps, "loss_D": epoch_loss_D/steps,
            "L1_hole": epoch_l1_hole/steps, "L1_valid": epoch_l1_valid/steps,
            "D_real_mean": epoch_D_real/steps, "D_fake_mean": epoch_D_fake/steps,
            "grad_G": epoch_grad_G/steps, "grad_D": epoch_grad_D/steps
        }.items():
            log[k].append(v)

    # ---- Samples visuales (fijos) ----
    G.eval()
    with torch.no_grad():
        Bf = fixed_batch.size(0)
        
        # Generamos nuevas máscaras usando mi módulo externo
        mask_f = torch.stack([random_circle_mask(IMG_SIZE, min_radius=12, max_radius=40) for _ in range(Bf)]).to(DEVICE)
        
        # Aplicamos las máscaras con tu función
        cond_rgb = apply_mask(fixed_batch, mask_f)
        cond_f = torch.cat([cond_rgb, mask_f], dim=1)

        # Pasamos por el generador
        pred_f = G(cond_f)
        comp_f = pred_f * (1 - mask_f) + fixed_batch * mask_f

        # Creamos el grid: cond (RGB), mask (visualizada en gris), pred, comp, gt
        grid = torch.cat([
            cond_f[:, :3],
            mask_f.repeat(1, 3, 1, 1) * 2 - 1,
            pred_f, comp_f, fixed_batch
        ], dim=0)

        save_image(grid, f"{run_dir}/samples/epoch_{epoch:03d}.png", nrow=Bf, normalize=True, value_range=(-1, 1))
        
    # ---- Plot pérdidas y guardar checkpoints cada N ----
    plot_losses(f"{run_dir}/losses_plot.png")
    if epoch % 20 == 0 or epoch == EPOCHS:
        torch.save(G.state_dict(), f"{run_dir}/checkpoints/G_epoch{epoch:03d}.pth")
        torch.save(D.state_dict(), f"{run_dir}/checkpoints/D_epoch{epoch:03d}.pth")

    # ---- Scheduler (decaimiento lineal a partir de la mitad) ----
    sched_G.step()
    sched_D.step()

    print(f"[{epoch:03d}/{EPOCHS}] "
        f"D: {log['loss_D'][-1]:.3f} | G: {log['loss_G'][-1]:.3f} | "
        f"L1(h): {log['L1_hole'][-1]:.3f} L1(v): {log['L1_valid'][-1]:.3f} | "
        f"Dreal: {log['D_real_mean'][-1]:.2f} Dfake: {log['D_fake_mean'][-1]:.2f} | "
        f"LR_G: {sched_G.get_last_lr()[0]:.6f} LR_D: {sched_D.get_last_lr()[0]:.6f}")

    # Guardar log CSV al final
    pd.DataFrame(log).to_csv(f"{run_dir}/training_log.csv", index=False)
    print(f"✅ Listo. Resultados en: {run_dir}")
