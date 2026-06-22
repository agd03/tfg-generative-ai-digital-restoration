# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm  # barra de progreso en terminal

import os
from torchvision.utils import save_image
from datetime import datetime

from inpaint_dataset import InpaintingImageDataset
from data_split import DataConfig, make_dataloaders
from model_inpainting import UNET
from patchgan_discriminator import PatchGANDiscriminator
from typing import Optional

import random
import numpy as np

# Para guardar la evolución de las pérdidas
import matplotlib.pyplot as plt
import json

'''
Resumen visual

              (gt) Imagen real completa
                     (3,H,W)
                        │
                        ▼
                 Genero máscara
              (mask) (1=visible, 0=hueco)
                        │
                        ▼
          masked_img = gt * mask
          (3,H,W) imagen con agujero
                        │
                        ▼
    cond = concat(masked_img, mask)
         → (4,H,W) entrada al generador

'''

def denorm(x):
    """
    Des-normaliza tensores de [-1,1] a [0,1] para poder guardarlos como imagen.
    x: tensor (B, C, H, W)
    """
    return (x * 0.5 + 0.5).clamp(0, 1)

def prepare_for_vgg(x_01, device, resize_to=None):
    """
    x_01: (B,3,H,W) en [0,1]
    Devuelve x normalizado con estadísticas de ImageNet (lo que espera VGG).
    Opcionalmente redimensiona (p.ej. a 224x224).
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x_01 - mean) / std

    if resize_to is not None:
        x = F.interpolate(x, size=resize_to, mode="bilinear", align_corners=False)

    return x


# Entender y consultar con tutor!!
'''
La pérdida adversarial se calcula como una media ponderada de la BCE por parche, donde cada término se pondera por la máscara reescalada al espacio del discriminador, lo que permite ignorar completamente regiones fuera del hueco y manejar máscaras suaves sin umbralización.
'''
def masked_bce_logits(pred, target, hole, eps=1e-8):
    """
    pred:   (B,1,h,w) logits del PatchGAN
    target: (B,1,h,w) labels (0 o 1)
    hole:   (B,1,H,W) 1 en hueco, 0 fuera (ojo: H,W > h,w)
    """
    # bajar hole a tamaño del mapa PatchGAN
    hole_patch = F.interpolate(hole, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    loss_map = F.binary_cross_entropy_with_logits(pred, target, reduction="none")  # (B,1,h,w)

    # media SOLO donde hole_patch=1
    return (loss_map * hole_patch).sum() / (hole_patch.sum() + eps)


def vgg_perceptual_loss(vgg_feat, x, y, layers=(4, 9, 16), mask=None):
    """
    x,y: (B,3,H,W) en rango [0,1] o [-1,1] (pero consistente)
    layers: índices de capas en VGG.features donde extraer activaciones
    mask: (B,1,H,W) opcional; si se da, aplica en el espacio de imagen antes de extraer features
    """
    if mask is not None:
        x = x * mask
        y = y * mask

    feats_x = []
    feats_y = []
    h_x, h_y = x, y
    for i, layer in enumerate(vgg_feat):
        h_x = layer(h_x)
        h_y = layer(h_y)
        if i in layers:
            feats_x.append(h_x)
            feats_y.append(h_y)

    loss = 0.0
    for fx, fy in zip(feats_x, feats_y):
        loss = loss + F.l1_loss(fx, fy)
    return loss


def lpips_hole(lpips_fn, x, y, hole):
    """
    LPIPS espera tensores normalizados a [-1,1] normalmente.
    hole: (B,1,H,W) 1 en hueco
    """
    # aplicar hueco en imagen (3 canales)
    hole3 = hole.repeat(1, 3, 1, 1)
    return lpips_fn(x * hole3, y * hole3).mean()

def l1_boundary_loss(fake, gt, hole, kernel_size=3):
    """
    fake, gt: (B,3,H,W)
    hole:     (B,1,H,W) 1 en hueco
    kernel_size: tamaño del anillo (3 → fino, 5 → más ancho)
    """
    # Dilatación del hueco
    pad = kernel_size // 2
    hole_dilated = F.max_pool2d(hole, kernel_size, stride=1, padding=pad)

    # Banda = dilatado menos hueco original
    boundary = hole_dilated - hole  # (B,1,H,W), 1 solo en el borde

    # Expandir a 3 canales
    boundary3 = boundary.repeat(1, 3, 1, 1)

    # L1 solo en la banda
    diff = (fake - gt).abs() * boundary3
    return diff.sum() / (boundary3.sum() + 1e-8)

@torch.no_grad()
def evaluate(G, D, loader, device, vgg_feat=None, lpips_fn=None):
    G.eval(); D.eval()

    sum_D = sum_Gadv = 0.0
    sum_L1_g = 0.0
    sum_L1_h = 0.0
    sum_L1_b = 0.0
    sum_VGG = 0.0
    sum_LPIPS = 0.0
    n = 0

    for cond, gt, mask in loader:
        cond = cond.to(device, non_blocking=True)
        gt   = gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        hole = 1.0 - mask

        fake = G(cond)
        comp = gt * mask + fake * hole

        pred_real = D(gt)
        pred_fake = D(comp)

        loss_D_real = masked_bce_logits(pred_real, torch.ones_like(pred_real), hole)
        loss_D_fake = masked_bce_logits(pred_fake, torch.zeros_like(pred_fake), hole)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        pred_fake_for_G = D(comp)
        loss_G_adv = masked_bce_logits(pred_fake_for_G, torch.ones_like(pred_fake_for_G), hole)

        loss_L1_global = F.l1_loss(comp, gt)
        loss_L1_hole = F.l1_loss(fake * hole, gt * hole)
        loss_L1_boundary = l1_boundary_loss(comp, gt, hole, kernel_size=3)

        loss_vgg = None
        loss_lp = None

        fake_01 = denorm(fake)
        gt_01   = denorm(gt)

        if vgg_feat is not None:
            hole3 = hole.repeat(1, 3, 1, 1)
            fake_vgg = prepare_for_vgg(fake_01, device=device, resize_to=None)
            gt_vgg   = prepare_for_vgg(gt_01,   device=device, resize_to=None)
            loss_vgg = vgg_perceptual_loss(vgg_feat, fake_vgg, gt_vgg, layers=(4, 9, 16), mask=hole3)

        if lpips_fn is not None:
            loss_lp = lpips_hole(lpips_fn, fake, gt, hole)

        sum_D += loss_D.item()
        sum_Gadv += loss_G_adv.item()
        sum_L1_g += loss_L1_global.item()
        sum_L1_h += loss_L1_hole.item()
        sum_L1_b += loss_L1_boundary.item()
        if loss_vgg is not None: sum_VGG += loss_vgg.item()
        if loss_lp is not None:  sum_LPIPS += loss_lp.item()
        n += 1

    out = {
        "D_adv": sum_D / max(n, 1),
        "G_adv": sum_Gadv / max(n, 1),
        "L1_global": sum_L1_g / max(n, 1),
        "L1_hole": sum_L1_h / max(n, 1),
        "L1_boundary": sum_L1_b / max(n, 1),
        "VGG_hole": (sum_VGG / n) if (vgg_feat is not None and n > 0) else None,
        "LPIPS_hole": (sum_LPIPS / n) if (lpips_fn is not None and n > 0) else None,
    }

    G.train(); D.train()
    return out

def _plot_series(history, keys, out_path, title=None):
    plt.figure()
    for k in keys:
        if k in history and any(v is not None for v in history[k]):
            y = [float(v) if v is not None else float("nan") for v in history[k]]
            plt.plot(y, label=k)
    if title:
        plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Valor medio por época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibilidad más estricta (puede bajar rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    data_root: str = "data/afhq_cats",
    img_size: int = 128,

    # Entrenamiento
    batch_size: int = 16,
    num_epochs: int = 200,
    lr_G: float = 2e-4,
    lr_D: float = 5e-5,  # TTUR suave (D algo más bajo suele estabilizar)
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    augmentation: bool = True,
    seed: int = 42,

    # Split reproducible
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,

    # Máscaras (círculos)
    mask_seed: int = 1234,
    min_radius: int = 12,
    max_radius: int = 40,

    # Feathering de máscara (Gaussiana). 0 = máscara dura 0/1
    mask_blur_radius: float = 3.0,

    # Augmentation params (solo si augmentation=True)
    scale: tuple = (0.8, 1.0),
    ratio: tuple = (0.9, 1.1),
    p_hflip: float = 0.5,

    # Pérdidas / modos
    loss_mode: str = "gan_l1",   # "gan" | "gan_l1" | "gan_l1_vgg" | "gan_l1_lpips"
    lambda_recon: float = 3.0,   # peso L1 en hueco o boundary (según tu implementación actual)
    lambda_vgg: float = 0.1,
    lambda_lpips: float = 0.1,

    # Logging / run
    run_name: str = "inpaint_unet",
    gpu_id: Optional[int] = None
):
    
    # 0) Carpeta única para este entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{run_name}_{loss_mode}")
    samples_dir = os.path.join(run_dir, "samples")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(seed)

    # 1) Configuración de datos (DataConfig)
    data_cfg = DataConfig(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        split_seed=split_seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        dataset_kwargs_train=dict(
            size=img_size,
            min_radius=min_radius,
            max_radius=max_radius,
            mask_blur_radius=mask_blur_radius,
            augment=augmentation,
            scale=scale,
            ratio=ratio,
            p_hflip=p_hflip,
            split="train",
            mask_seed=mask_seed,
        ),
        dataset_kwargs_val=dict(
            size=img_size,
            min_radius=min_radius,
            max_radius=max_radius,
            mask_blur_radius=mask_blur_radius,
            augment=False,
            split="val",
            mask_seed=mask_seed,
        ),
        dataset_kwargs_test=dict(
            size=img_size,
            min_radius=min_radius,
            max_radius=max_radius,
            mask_blur_radius=mask_blur_radius,
            augment=False,
            split="test",
            mask_seed=mask_seed,
        ),
    )

    # 2) DataLoaders reproducibles
    train_loader, val_loader, test_loader, split = make_dataloaders(
        cfg=data_cfg,
        run_dir=run_dir,
        dataset_cls=InpaintingImageDataset
    )

    # Guardamos un resumen completo de la configuración del experimento
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        # --------------------
        # DATA / DATASET
        # --------------------
        f.write(f"data_root = {data_root}\n")
        f.write(f"img_size = {img_size}\n")
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"num_workers = {num_workers}\n")
        f.write(f"pin_memory = {pin_memory}\n")
        f.write(f"persistent_workers = {persistent_workers}\n")
        f.write(f"gpu_id_requested_nvidia_smi = {gpu_id}\n")
        f.write(f"seed = {seed}\n")

        # --------------------
        # SPLIT (REPRODUCIBLE)
        # --------------------
        f.write(f"split_seed = {split_seed}\n")
        f.write(f"train_frac = {train_frac}\n")
        f.write(f"val_frac = {val_frac}\n")
        f.write(f"test_frac = {test_frac}\n")

        # NUEVO: split guardado en run_dir
        f.write(f"split_file = {os.path.join(run_dir, 'split.json')}\n")
        f.write(f"n_total = {len(split.train_idx) + len(split.val_idx) + len(split.test_idx)}\n")
        f.write(f"n_train = {len(split.train_idx)}\n")
        f.write(f"n_val = {len(split.val_idx)}\n")
        f.write(f"n_test = {len(split.test_idx)}\n")

        # --------------------
        # MASKS
        # --------------------
        f.write(f"mask_seed = {mask_seed}\n")
        f.write(f"mask_type = circular\n")
        f.write(f"min_radius = {min_radius}\n")
        f.write(f"max_radius = {max_radius}\n")
        f.write(f"mask_blur_radius = {mask_blur_radius}\n")

        # --------------------
        # DATA AUGMENTATION
        # --------------------
        f.write(f"augmentation = {augmentation}\n")
        if augmentation:
            f.write(f"scale = {scale}\n")
            f.write(f"ratio = {ratio}\n")
            f.write(f"p_hflip = {p_hflip}\n")

        # --------------------
        # TRAINING
        # --------------------
        f.write(f"num_epochs = {num_epochs}\n")
        f.write(f"lr_G = {lr_G}\n")
        f.write(f"lr_D = {lr_D}\n")

        # --------------------
        # LOSSES / OBJECTIVE
        # --------------------
        f.write(f"loss_mode = {loss_mode}\n")
        f.write(f"lambda_recon = {lambda_recon}\n")
        f.write(f"lambda_vgg = {lambda_vgg}\n")
        f.write(f"lambda_lpips = {lambda_lpips}\n")

        # --------------------
        # METRICS (EVALUATION)
        # --------------------
        f.write(f"measure_l1_hole = True\n")
        f.write(f"measure_l1_boundary = True\n")
        f.write(f"measure_lpips = True\n")
        f.write(f"measure_vgg = True\n")
        f.write("best_model_selected_by = validation_total_loss (same as training objective)\n")


    print(f"📂 Guardando este entrenamiento en: {run_dir}")

    # 1) Dispositivo: GPU si hay, si no CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️  Usando dispositivo:", device)
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print("GPU activa (PyTorch):", torch.cuda.get_device_name(idx))

    # device = torch.device("cpu")
    # print("⚠️ Entrenando en CPU (GPU no disponible o demasiado pequeña).")

    # 3) Modelos: generador (UNET) y discriminador (CNN)
    G = UNET(in_channels=4, out_channels=3).to(device)              # in: [masked(3)+mask(1)]
    D = PatchGANDiscriminator(in_channels=3).to(device)
    
    # *) Módulos pérdida perceptual (VGG/LPIPS) si aplica
    use_l1    = loss_mode in ["gan_l1", "gan_l1_vgg", "gan_l1_lpips"]
    use_vgg   = loss_mode == "gan_l1_vgg"
    use_lpips = loss_mode == "gan_l1_lpips"

    #measure_lpips = True

    vgg_feat = None
    lpips_fn = None

    if use_vgg:   # True si quieres medir VGG siempre como métrica
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        vgg_feat = vgg

    #if measure_lpips: # True si quieres medir LPIPS siempre como métrica
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").eval().to(device)

    # 5) Optimizadores
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # 6) Historial de pérdidas (diccionario de listas)
    history = {
        "D_adv": [],
        "G_adv": [],
        "L1_global": [],
        "L1_hole": [],
        "L1_boundary": [],
        "VGG_hole": [],
        "LPIPS_hole": [],
        "G_total": []
    }

    best_metric = float("inf")
    best_epoch = -1


    # 6) Bucle de entrenamiento
    for epoch in range(num_epochs):
        G.train()
        D.train()

        # Pérdidas acumuladas por época
        loss_D_epoch = 0.0
        loss_G_adv_epoch = 0.0

        loss_L1_global_epoch = 0.0
        loss_L1_hole_epoch = 0.0
        loss_L1_boundary_epoch = 0.0

        loss_VGG_epoch = 0.0
        loss_LPIPS_epoch = 0.0

        loss_G_total_epoch = 0.0


        # tqdm envuelve al dataloader y muestra una barra por época
        progress_bar = tqdm(
            train_loader,
            desc=f"Época {epoch+1}/{num_epochs}",
            leave=True
        )

        for step, (cond, gt, mask) in enumerate(progress_bar):
            # cond: (B,4,H,W)  -> [imagen_enmascarada (3), máscara (1)]
            # gt:   (B,3,H,W)  -> imagen completa real
            # mask: (B,1,H,W)  -> 1 visible / 0 agujero

            cond = cond.to(device, non_blocking=True)
            gt   = gt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            hole = 1 - mask  # 1 en hueco, 0 fuera

            # =====================================
            # 1) Actualizar el DISCRIMINADOR (D)
            # =====================================
            optim_D.zero_grad()

            # Fake para D: sin gradiente
            with torch.no_grad():
                fake_D = G(cond)          # generamos imagen fake (B,3,H,W)
                comp_fake_D = gt * mask + fake_D * hole  # solo cambia hueco

            pred_real = D(gt)                 # D ve la imagen real
            pred_fake = D(comp_fake_D)

            loss_D_real = masked_bce_logits(pred_real, torch.ones_like(pred_real), hole)
            loss_D_fake = masked_bce_logits(pred_fake, torch.zeros_like(pred_fake), hole)

            # Pérdida total de D (media de ambas)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optim_D.step()

            # =====================================
            # 2) Actualizar el GENERADOR (G)
            # =====================================
            optim_G.zero_grad()

            fake_G = G(cond)                        # nueva pasada, ahora SÍ con gradiente
            comp_fake_G = gt * mask + fake_G * hole  # imagen compuesta

            pred_fake_for_G = D(comp_fake_G)             # D ve el fake para la pérdida adversarial

            loss_G_adv = masked_bce_logits(pred_fake_for_G, torch.ones_like(pred_fake_for_G), hole)

            # Métricas base
            loss_L1_global = F.l1_loss(comp_fake_G, gt)
            loss_L1_hole = F.l1_loss(fake_G * hole, gt * hole)
            loss_L1_boundary = l1_boundary_loss(comp_fake_G, gt, hole, kernel_size=3)

            # Perceptuales (como métricas, y opcionalmente como parte de la loss)
            loss_vgg = None
            loss_lpips = None

            # IMPORTANTE: define un criterio consistente de rango para perceptual.
            # Si tu training está en [-1,1], para VGG normalmente conviene llevar a [0,1].
            fake_01 = denorm(fake_G)
            gt_01   = denorm(gt)

            if vgg_feat is not None:
                # máscara 3 canales
                hole3 = hole.repeat(1, 3, 1, 1)

                # Preparar para VGG (normalización ImageNet)
                fake_vgg = prepare_for_vgg(fake_01, device=device, resize_to=None)  # o (224,224)
                gt_vgg   = prepare_for_vgg(gt_01,   device=device, resize_to=None)  # o (224,224)

                # VGG perceptual (tu función ya acepta mask)
                loss_vgg = vgg_perceptual_loss(vgg_feat, fake_vgg, gt_vgg, layers=(4, 9, 16), mask=hole3)

            if lpips_fn is not None:
                # LPIPS suele funcionar directo en [-1,1]
                loss_lpips = lpips_hole(lpips_fn, fake_G, gt, hole)

            # Construcción de la loss final (según experimento)
            loss_G = loss_G_adv

            if use_l1:
                loss_G = loss_G + lambda_recon * loss_L1_hole

            if use_vgg and loss_vgg is not None:
                loss_G = loss_G + lambda_vgg * loss_vgg

            if use_lpips and loss_lpips is not None:
                loss_G = loss_G + lambda_lpips * loss_lpips


            loss_G.backward()
            optim_G.step()

            # Guardar una muestra visual del primer batch de la época
            if step == 0:
                with torch.no_grad():
                    # Reutilizamos lo ya calculado en el paso de entrenamiento
                    fake_vis = fake_G.detach().cpu()
                    gt_vis   = gt.detach().cpu()
                    cond_vis = cond.detach().cpu()
                    mask_vis = mask.detach().cpu()
                    hole_vis = (1.0 - mask_vis)

                    # La imagen enmascarada está en los 3 primeros canales de cond
                    masked_vis = cond_vis[:, :3, :, :]

                    # Des-normalizamos a [0,1]
                    gt_vis     = denorm(gt_vis)
                    masked_vis = denorm(masked_vis)
                    fake_vis   = denorm(fake_vis)

                    # Composición final (lo que realmente “sale” del sistema)
                    comp_vis = gt_vis * mask_vis + fake_vis * hole_vis

                    # Apilamos: [gt, masked, comp]
                    grid = torch.cat([gt_vis, masked_vis, comp_vis], dim=0)

                    save_image(
                        grid,
                        os.path.join(samples_dir, f"epoch_{epoch+1:03d}.png"),
                        nrow=gt_vis.size(0),
                    )

            # Acumulamos las pérdidas para el historial
            loss_D_epoch += loss_D.item()
            loss_G_adv_epoch += loss_G_adv.item()

            loss_L1_global_epoch += loss_L1_global.item()
            loss_L1_hole_epoch += loss_L1_hole.item()
            loss_L1_boundary_epoch += loss_L1_boundary.item()

            if loss_vgg is not None:
                loss_VGG_epoch += loss_vgg.item()

            if loss_lpips is not None:
                loss_LPIPS_epoch += loss_lpips.item()

            loss_G_total_epoch += loss_G.item()

            # Actualizamos el texto de la barra de progreso
            postfix = {
                "Loss_D": f"{loss_D.item():.3f}",
                "G_adv": f"{loss_G_adv.item():.3f}",
                "L1_h": f"{loss_L1_hole.item():.3f}",
                "L1_b": f"{loss_L1_boundary.item():.3f}",
            }

            if loss_vgg is not None:
                postfix["VGG"] = f"{loss_vgg.item():.3f}"

            if loss_lpips is not None:
                postfix["LPIPS"] = f"{loss_lpips.item():.3f}"

            progress_bar.set_postfix(postfix)

        # Media de las pérdidas por época
        n_steps = len(train_loader)

        history["D_adv"].append(loss_D_epoch / n_steps)
        history["G_adv"].append(loss_G_adv_epoch / n_steps)

        history["L1_global"].append(loss_L1_global_epoch / n_steps)
        history["L1_hole"].append(loss_L1_hole_epoch / n_steps)
        history["L1_boundary"].append(loss_L1_boundary_epoch / n_steps)

        if use_vgg:
            history["VGG_hole"].append(loss_VGG_epoch / n_steps)
        else:
            history["VGG_hole"].append(None)

        #if measure_lpips:
        history["LPIPS_hole"].append(loss_LPIPS_epoch / n_steps)
        #else:
        #    history["LPIPS_hole"].append(None)

        history["G_total"].append(loss_G_total_epoch / n_steps)

        # ==========================
        # VALIDACIÓN (por época)
        # ==========================
        val_metrics = evaluate(G, D, val_loader, device, vgg_feat=vgg_feat, lpips_fn=lpips_fn)

        # Guardar series de validación en history (con sufijo _val)
        for k, v in val_metrics.items():
            key = f"{k}_val"
            history.setdefault(key, [])
            history[key].append(v)

        # --------------------------
        # CHECKPOINTS "LAST"
        # --------------------------
        torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_last.pth"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, "D_last.pth"))

        # --------------------------
        # CHECKPOINTS "BEST"
        # --------------------------
        # Criterio BEST = misma loss que optimizas, pero en VALIDACIÓN
        val_loss_total = val_metrics["G_adv"]

        if loss_mode in ("gan_l1", "gan_l1_vgg", "gan_l1_lpips"):
            val_loss_total += lambda_recon * val_metrics["L1_hole"]

        if loss_mode == "gan_l1_vgg":
            assert val_metrics.get("VGG_hole") is not None, (
                "loss_mode='gan_l1_vgg' pero VGG_hole es None en validación. "
                "Revisa que vgg_feat se cargue y que evaluate(...) reciba vgg_feat."
            )
            val_loss_total += lambda_vgg * val_metrics["VGG_hole"]

        if loss_mode == "gan_l1_lpips":
            assert val_metrics.get("LPIPS_hole") is not None, (
                "loss_mode='gan_l1_lpips' pero LPIPS_hole es None en validación. "
                "Revisa que lpips_fn esté cargado y que evaluate(...) lo reciba."
            )
            val_loss_total += lambda_lpips * val_metrics["LPIPS_hole"]


        history.setdefault("G_total_val", []).append(float(val_loss_total))

        if val_loss_total < best_metric:
            best_metric = val_loss_total
            best_epoch = epoch + 1
            torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_best.pth"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, "D_best.pth"))
            with open(os.path.join(run_dir, "best.txt"), "w") as bf:
                bf.write(f"best_epoch = {best_epoch}\n")
                bf.write("best_criterion = validation_total_loss\n")
                bf.write(f"loss_mode = {loss_mode}\n")
                bf.write(f"best_value = {best_metric}\n")


        # Guardar checkpoints al final de cada época
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch_{epoch+1:03d}.pth"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch_{epoch+1:03d}.pth"))

    # ==========================
    # TEST FINAL (con G_best)
    # ==========================
    best_g_path = os.path.join(ckpt_dir, "G_best.pth")
    best_d_path = os.path.join(ckpt_dir, "D_best.pth")

    if os.path.exists(best_g_path):
        G.load_state_dict(torch.load(best_g_path, map_location=device))
    if os.path.exists(best_d_path):
        D.load_state_dict(torch.load(best_d_path, map_location=device))

    test_metrics = evaluate(G, D, test_loader, device, vgg_feat=vgg_feat, lpips_fn=lpips_fn)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


    # Guardar el historial de pérdidas en un archivo JSON
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Graficar las pérdidas / métricas

    # 1) Adversarial (train + val)
    _plot_series( history,
        ["D_adv", "G_adv", "D_adv_val", "G_adv_val"],
        os.path.join(run_dir, "adv_train_val.png"),
        title="Adversarial"
    )

    # 2) Reconstrucción (train + val)
    _plot_series(
        history,
        ["L1_global", "L1_hole", "L1_global_val", "L1_hole_val"],
        os.path.join(run_dir, "l1_train_val.png"),
        title="L1 (global vs hole)"
    )

    # 3) Perceptual (train + val) — solo si hay datos
    _plot_series( history,
        ["LPIPS_hole", "VGG_hole", "LPIPS_hole_val", "VGG_hole_val"],
        os.path.join(run_dir, "perceptual_train_val.png"),
        title="Perceptual"
    )

    # 4) Total (solo train)
    _plot_series(
        history,
        ["G_total", "G_total_val"],
        os.path.join(run_dir, "g_total_train_val.png"),
        title="G_total (train vs val)"
    )

    _plot_series(
        history,
        ["L1_boundary", "L1_boundary_val"],
        os.path.join(run_dir, "l1_boundary_train_val.png"),
        title="L1 boundary (diagnostic)"
    )


if __name__ == "__main__":
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser(description="Entrenamiento GAN Inpainting")
    parser.add_argument("--data_root", type=str, default="data/afhq_cats")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr_G", type=float, default=2e-4)
    #parser.add_argument("--lr_D", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--mask_seed", type=int, default=1234)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    parser.add_argument("--loss_mode", type=str, default="gan_l1",
                        choices=["gan", "gan_l1", "gan_l1_vgg", "gan_l1_lpips"])
    #parser.add_argument("--lambda_recon", type=float, default=3.0)
    parser.add_argument("--lambda_vgg", type=float, default=0.1)
    #parser.add_argument("--lambda_lpips", type=float, default=0.1)

    #parser.add_argument("--run_name", type=str, default="inpaint_unet_lambda3")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pin_memory", action="store_true", help="Desactiva pin_memory (por defecto True)")      # override
    parser.add_argument("--no_persistent_workers", action="store_true", help="Desactiva persistent_workers (por defecto True)") # por defecto False si no lo pasas
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Desactiva data augmentation (por defecto está activado)"
    )

    parser.add_argument("--gpu_id", type=int, default=0,
                        help="ID de GPU según nvidia-smi")

    args = parser.parse_args()

    augmentation = not args.no_augmentation

    # GPU selection (mapeo nvidia-smi -> CUDA_VISIBLE_DEVICES -> cuda:0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # pin_memory default: si no pasas nada, lo activamos (recomendado)
    pin_memory = not args.no_pin_memory

    train(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_G=args.lr_G,
        #lr_D=args.lr_D,
        loss_mode=args.loss_mode,
        #lambda_recon=args.lambda_recon,
        lambda_vgg=args.lambda_vgg,
        #lambda_lpips=args.lambda_lpips,
        #run_name=args.run_name,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers = not args.no_persistent_workers,
        augmentation=augmentation,
        gpu_id=args.gpu_id,
        split_seed=args.split_seed,
        mask_seed=args.mask_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )

