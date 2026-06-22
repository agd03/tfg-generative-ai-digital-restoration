import torch
from torchvision.utils import save_image, make_grid
import os

CUDA_VISIBLE_DEVICES=0

# --- Inicialización de pesos (estilo DCGAN) ---
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias.data)

# --- Generador de ruido ---
def sample_noise_map(batch_size, z_ch=256, fmap=16, device="cuda"):
    # Genera un tensor de ruido (mapa) con estructura espacial.
    return torch.randn(batch_size, z_ch, fmap, fmap, device=device)

# --- Guardar imágenes del generador ---
def save_generated_images(generator, fixed_noise, step, out_dir="runs/samples"):
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise).cpu()
    grid = make_grid(fake_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, os.path.join(out_dir, f"step_{step:06d}.png"))
    generator.train()
