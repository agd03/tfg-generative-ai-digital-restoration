
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # barra de progreso en terminal

import os
from torchvision.utils import save_image
from datetime import datetime

from inpaint_dataset import InpaintingCatsDataset
from model_inpainting import UNET
from discriminator import Discriminator

# Para guardar la evolución de las pérdidas
import matplotlib.pyplot as plt
import json

import argparse

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

def train(
    data_root="data/afhq_cats",
    img_size=128,
    #batch_size=16,
    batch_size=64,
    num_epochs=200,
    lr_G=2e-4,
    lr_D=2e-4,
    #lambda_recon=10.0,   # peso de la L1 en la zona del agujero
    lambda_recon=3.0,
    run_name="inpaint_unet_lambda3"
):
    # 0) Carpeta única para este entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{run_name}")
    samples_dir = os.path.join(run_dir, "samples")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Guardamos un pequeño resumen de la configuración
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"data_root = {data_root}\n")
        f.write(f"img_size = {img_size}\n")
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"num_epochs = {num_epochs}\n")
        f.write(f"lr_G = {lr_G}\n")
        f.write(f"lr_D = {lr_D}\n")
        f.write(f"lambda_recon = {lambda_recon}\n")

    print(f"📂 Guardando este entrenamiento en: {run_dir}")

    # 1) Dispositivo: GPU si hay, si no CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        print("GPU activa (PyTorch):", torch.cuda.get_device_name(gpu_index))

    # 2) Dataset y DataLoader
    dataset = InpaintingCatsDataset(root=data_root, size=img_size)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 3) Modelos: generador (UNET) y discriminador (MLP)
    G = UNET(in_channels=4, out_channels=3).to(device)              # in: [masked(3)+mask(1)]
    D = Discriminator(img_size=img_size, in_channels=3).to(device)  # ve imágenes completas (3,H,W)

    # 4) Pérdidas
    bce_loss = nn.BCELoss()   # para la parte adversarial
    l1_loss  = nn.L1Loss()    # para la reconstrucción en la zona del agujero

    # 5) Optimizadores
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # 6) Historial de pérdidas (diccionario de listas)
    history = {
        "D": [],
        "G_adv": [],
        "G_L1": [],
        "G_total": [],
    }

    # 6) Bucle de entrenamiento
    for epoch in range(num_epochs):
        G.train()
        D.train()

        # Pérdidas acumuladas en la época inicializadas a 0
        loss_D_epoch = 0.0
        loss_G_adv_epoch = 0.0
        loss_G_L1_epoch = 0.0
        loss_G_total_epoch = 0.0

        # tqdm envuelve al dataloader y muestra una barra por época
        progress_bar = tqdm(
            dataloader,
            desc=f"Época {epoch+1}/{num_epochs}",
            leave=True
        )

        for step, (cond, gt, mask) in enumerate(progress_bar):
            # cond: (B,4,H,W)  -> [imagen_enmascarada (3), máscara (1)]
            # gt:   (B,3,H,W)  -> imagen completa real
            # mask: (B,1,H,W)  -> 1 visible / 0 agujero

            cond = cond.to(device)
            gt   = gt.to(device)
            mask = mask.to(device)

            bs = gt.size(0)

            # Etiquetas reales (1) y falsas (0) para D
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # =====================================
            # 1) Actualizar el DISCRIMINADOR (D)
            # =====================================
            optim_D.zero_grad()

            # a) Pérdida con imágenes reales
            pred_real = D(gt)                     # D ve la imagen real completa
            loss_D_real = bce_loss(pred_real, real_labels)

            # b) Pérdida con imágenes falsas (generadas)
            with torch.no_grad():                 # no queremos gradientes para G aquí
                fake = G(cond)                    # generamos imagen fake
            pred_fake = D(fake.detach())
            loss_D_fake = bce_loss(pred_fake, fake_labels)

            # c) Pérdida total de D (media de ambas)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optim_D.step()

            # =====================================
            # 2) Actualizar el GENERADOR (G)
            # =====================================
            optim_G.zero_grad()

            fake = G(cond)                        # nueva pasada, ahora SÍ con gradiente
            pred_fake_for_G = D(fake)             # D ve el fake para la pérdida adversarial

            # a) Pérdida adversarial: queremos que D(fake) ≈ 1
            loss_G_adv = bce_loss(pred_fake_for_G, real_labels)

            # b) Pérdida de reconstrucción SOLO en la zona del agujero
            #    mask = 1 visible, 0 agujero  ⇒ (1 - mask) = 1 en el agujero
            hole = 1 - mask
            loss_G_recon = l1_loss(fake * hole, gt * hole)

            # c) Pérdida total del generador
            loss_G = loss_G_adv + lambda_recon * loss_G_recon
            loss_G.backward()
            optim_G.step()

            # Guardar una muestra visual del primer batch de la época
            if step == 0:
                with torch.no_grad():
                    # fake ya lo tenemos calculado más arriba
                    fake_vis = fake.detach().cpu()
                    gt_vis   = gt.detach().cpu()
                    cond_vis = cond.detach().cpu()

                    # la imagen enmascarada está en los 3 primeros canales de cond
                    masked_vis = cond_vis[:, :3, :, :]

                    # Des-normalizamos a [0,1]
                    gt_vis      = denorm(gt_vis)
                    masked_vis  = denorm(masked_vis)
                    fake_vis    = denorm(fake_vis)

                    # Apilamos: [gt (fila 1), masked (fila 2), fake (fila 3)]
                    # concatenamos en batch: primero todos gt, luego todos masked, luego todos fake
                    grid = torch.cat([gt_vis, masked_vis, fake_vis], dim=0)

                    save_image(
                        grid,
                        os.path.join(samples_dir, f"epoch_{epoch+1:03d}.png"),
                        nrow=gt_vis.size(0),
                    )
            
            # Acumulamos las pérdidas para el historial
            loss_D_epoch        += loss_D.item()
            loss_G_adv_epoch    += loss_G_adv.item()
            loss_G_L1_epoch     += loss_G_recon.item()
            loss_G_total_epoch  += loss_G.item()

            # Actualizamos el texto de la barra con las pérdidas actuales
            progress_bar.set_postfix({
                "Loss_D": f"{loss_D.item():.3f}",
                "G_adv": f"{loss_G_adv.item():.3f}",
                "G_L1": f"{loss_G_recon.item():.3f}"
            })

        # Media de las pérdidas por época
        n_steps = len(dataloader)
        history["D"].append(loss_D_epoch / n_steps)
        history["G_adv"].append(loss_G_adv_epoch / n_steps)
        history["G_L1"].append(loss_G_L1_epoch / n_steps)
        history["G_total"].append(loss_G_total_epoch / n_steps)

        # Guardar checkpoints al final de cada época
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch_{epoch+1:03d}.pth"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch_{epoch+1:03d}.pth"))

    # Guardar el historial de pérdidas en un archivo JSON
    with open(os.path.join(run_dir, "losses.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Graficar las pérdidas
    plt.figure()
    plt.plot(history["D"],      label="D")
    plt.plot(history["G_adv"],  label="G_adv")
    plt.plot(history["G_L1"],   label="G_L1")
    plt.plot(history["G_total"],label="G_total")
    plt.xlabel("Época")
    plt.ylabel("Pérdida media")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "losses.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento GAN Inpainting")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/afhq_cats",  # <-- tu valor actual
        help="Ruta al dataset para entrenamiento"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,  # <-- comportamiento actual
        help="ID de la GPU a utilizar (según nvidia-smi)"
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.set_device(0)

    print("torch.cuda.is_available():", torch.cuda.is_available())

    train(data_root=args.data_root)

