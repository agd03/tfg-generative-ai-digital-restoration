from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model_inpainting import UNET


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "G_epoch_200.pth"
RECORTES_DIR = BASE_DIR / "recortes128"
OUTPUT_PATH = BASE_DIR / "oberranna_unet_muestras_finales.png"

IMAGE_FILES = [
	"recorte01.jpg",
	"recorte02.jpg",
	"recorte03.jpg",
	"recorte04.jpg",
	"recorte05.jpg",
]

# Máscaras fijas: (cx, cy, radio) en píxeles, para imágenes 128x128
# Cámbialas si quieres recolocar los círculos.
MASK_SPECS = [
	(92, 34, 20),
	(82, 70, 22),
	(36, 46, 22),
	(44, 78, 20),
	(34, 70, 21),
]

IMG_SIZE = 128
GREY_VALUE = 0.6


# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------

def load_generator():
	model = UNET(in_channels=4, out_channels=3).to(DEVICE)

	checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

	if isinstance(checkpoint, dict):
		if "state_dict" in checkpoint:
			state_dict = checkpoint["state_dict"]
		elif "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
		else:
			state_dict = checkpoint
	else:
		state_dict = checkpoint

	model.load_state_dict(state_dict, strict=True)
	model.eval()

	return model


def make_circle_mask(size, cx, cy, radius):
	yy, xx = np.ogrid[:size, :size]
	dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

	mask = np.ones((size, size), dtype=np.float32)
	mask[dist2 <= radius ** 2] = 0.0

	return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


def tensor_to_numpy_img(tensor_img):
	tensor_img = tensor_img.detach().cpu().clamp(0.0, 1.0)
	return tensor_img.permute(1, 2, 0).numpy()


def build_masked_visual(image_tensor, mask_tensor, grey_value=0.6):
	mask_rgb = mask_tensor.repeat(3, 1, 1)
	grey = torch.full_like(image_tensor, grey_value)
	masked_vis = image_tensor * mask_rgb + grey * (1.0 - mask_rgb)
	return masked_vis.clamp(0.0, 1.0)


def load_image(image_path):
	transform = transforms.Compose([
		transforms.Resize((IMG_SIZE, IMG_SIZE)),
		transforms.ToTensor(),
	])

	image = Image.open(image_path).convert("RGB")
	return transform(image)


# --------------------------------------------------
# INFERENCIA
# --------------------------------------------------

def reconstruct_image(model, image_tensor, mask_tensor):
	image_tensor = image_tensor.to(DEVICE)
	mask_tensor = mask_tensor.to(DEVICE)

	masked_input = image_tensor * mask_tensor
	model_input = torch.cat([masked_input, mask_tensor], dim=0).unsqueeze(0)  # (1, 4, H, W)

	with torch.no_grad():
		pred = model(model_input).squeeze(0)

	composed = image_tensor * mask_tensor + pred * (1.0 - mask_tensor)
	composed = composed.clamp(0.0, 1.0)

	return masked_input.cpu(), pred.cpu().clamp(0.0, 1.0), composed.cpu()


# --------------------------------------------------
# FIGURA
# --------------------------------------------------

def create_grid():
	model = load_generator()

	fig, axes = plt.subplots(
		3,
		5,
		figsize=(15, 9)
	)

	row_labels = [
		"Imagen enmascarada",
		"Reconstrucción",
		"Imagen original",
	]

	for col, (image_name, mask_spec) in enumerate(zip(IMAGE_FILES, MASK_SPECS)):
		image_path = RECORTES_DIR / image_name

		if not image_path.exists():
			raise FileNotFoundError(f"No existe el archivo: {image_path}")

		image_tensor = load_image(image_path)
		mask_tensor = make_circle_mask(IMG_SIZE, *mask_spec)

		_, _, composed = reconstruct_image(
			model,
			image_tensor,
			mask_tensor
		)

		masked_visual = build_masked_visual(
			image_tensor,
			mask_tensor,
			grey_value=GREY_VALUE
		)

		images_to_show = [
			masked_visual,
			composed,
			image_tensor,
		]

		for row in range(3):
			ax = axes[row, col]
			ax.imshow(tensor_to_numpy_img(images_to_show[row]))
			ax.set_xticks([])
			ax.set_yticks([])

			for spine in ax.spines.values():
				spine.set_visible(False)

	for row, label in enumerate(row_labels):
		axes[row, 0].set_ylabel(
			label,
			fontsize=16,
			rotation=90,
			labelpad=28,
			va="center"
		)

	plt.tight_layout()
	plt.savefig(
		OUTPUT_PATH,
		dpi=300,
		bbox_inches="tight"
	)
	plt.close(fig)

	print(f"Cuadrícula guardada en: {OUTPUT_PATH}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
	print(f"Dispositivo utilizado: {DEVICE}")
	create_grid()
