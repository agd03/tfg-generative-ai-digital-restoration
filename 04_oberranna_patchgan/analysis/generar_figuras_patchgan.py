from pathlib import Path
import csv
import json
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms

from model_inpainting import UNET


# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------

ROOT_DIR = Path.cwd()
RECORTES_DIR = ROOT_DIR / "recortes128"
OUTPUT_DIR = ROOT_DIR / "figuras_patchgan"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

SAMPLE_IMAGES = [
	"recorte02.jpg",
	"recorte03.jpg",
	"recorte04.jpg",
	"recorte05.jpg",
]

# (centro_x, centro_y, radio)
MASK_SPECS = [
	(82, 70, 22),
	(42, 52, 22),
	(74, 80, 20),
	(34, 70, 21),
]

# Valores de respaldo si no aparecen en config.txt.
BLUR_FALLBACKS = {
	"EXP_01": 0.0,
	"EXP_02": 0.0,
	"EXP_07": 0.0,
	"EXP_08": 0.0,
	"EXP_09": 0.0,
	"EXP_10": 2.0,
	"EXP_11": 3.0,
	"EXP_13A": 3.0,
	"EXP_13B": 3.0,
}

EXPERIMENTS = {
	"EXP_01": {
		"folder": "exp01",
		"config": "01_config.txt",
		"best": "01_best.txt",
		"test_metrics": "01_test_metrics.json",
		"checkpoint": "G_epoch_021.pth",
	},
	"EXP_02": {
		"folder": "exp02",
		"config": "02_config.txt",
		"best": "02_best.txt",
		"test_metrics": "02_test_metrics.json",
		"checkpoint": "G_epoch_021.pth",
	},
	"EXP_07": {
		"folder": "exp07",
		"config": "07_config.txt",
		"best": "07_best.txt",
		"test_metrics": "07_test_metrics.json",
		"checkpoint": "G_epoch_027.pth",
	},
	"EXP_08": {
		"folder": "exp08",
		"config": "08_config.txt",
		"best": "08_best.txt",
		"test_metrics": "08_test_metrics.json",
		"checkpoint": "G_epoch_030.pth",
	},
	"EXP_09": {
		"folder": "exp09",
		"config": "09_config.txt",
		"best": "09_best.txt",
		"test_metrics": "09_test_metrics.json",
		"checkpoint": "G_epoch_026.pth",
	},
	"EXP_10": {
		"folder": "exp10",
		"config": "10_config.txt",
		"best": "10_best.txt",
		"test_metrics": "10_test_metrics.json",
		"checkpoint": "G_epoch_025.pth",
	},
	"EXP_11": {
		"folder": "exp11",
		"config": "11_config.txt",
		"best": "11_best.txt",
		"test_metrics": "11_test_metrics.json",
		"checkpoint": "G_epoch_058.pth",
	},
	"EXP_13A": {
		"folder": "exp13",
		"config": "13A_config.txt",
		"best": "13A_best.txt",
		"test_metrics": "13A_test_metrics.json",
		"checkpoint": "G_epoch_023.pth",
	},
	"EXP_13B": {
		"folder": "exp13",
		"config": "13B_config.txt",
		"best": "13B_best.txt",
		"test_metrics": "13B_test_metrics.json",
		"checkpoint": "G_epoch_058.pth",
	},
}

EXPERIMENT_ORDER = [
	"EXP_01",
	"EXP_02",
	"EXP_07",
	"EXP_08",
	"EXP_09",
	"EXP_10",
	"EXP_11",
	"EXP_13A",
	"EXP_13B",
]

ANNOTATION_OFFSETS = {
	"EXP_01": (6, 5),
	"EXP_02": (6, 5),
	"EXP_07": (6, 5),
	"EXP_08": (6, 5),
	"EXP_09": (6, 5),
	"EXP_10": (6, 5),
	"EXP_11": (10, 10),
	"EXP_13A": (6, 5),
	"EXP_13B": (6, -15),
}

CONFIG_CACHE = {}


# --------------------------------------------------
# LECTURA DE ARCHIVOS
# --------------------------------------------------

def experiment_dir(label):
	return ROOT_DIR / EXPERIMENTS[label]["folder"]


def experiment_path(label, key):
	return experiment_dir(label) / EXPERIMENTS[label][key]


def read_key_value_file(path):
	values = {}

	if not path.exists():
		return values

	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()

		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		values[key.strip()] = value.strip()

	return values


def get_config(label):
	if label not in CONFIG_CACHE:
		CONFIG_CACHE[label] = read_key_value_file(
			experiment_path(label, "config")
		)

	return CONFIG_CACHE[label]


def get_config_float(label, key, default):
	config = get_config(label)

	if key not in config:
		return float(default)

	try:
		return float(config[key])
	except ValueError:
		return float(default)


def get_blur_radius(label):
	return get_config_float(
		label,
		"mask_blur_radius",
		BLUR_FALLBACKS[label],
	)


def read_json(path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def find_metric(data, candidates):
	if not isinstance(data, dict):
		return None

	normalised = {
		str(key).lower(): value
		for key, value in data.items()
	}

	for candidate in candidates:
		value = normalised.get(candidate.lower())

		if value is None:
			continue

		try:
			return float(value)
		except (TypeError, ValueError):
			continue

	return None


def read_best_epoch(label):
	best_path = experiment_path(label, "best")

	if best_path.exists():
		for line in best_path.read_text(encoding="utf-8").splitlines():
			match = re.match(r"\s*best_epoch\s*=\s*(\d+)", line)

			if match:
				return int(match.group(1))

	checkpoint_name = EXPERIMENTS[label]["checkpoint"]
	match = re.search(r"epoch_(\d+)", checkpoint_name)

	if match:
		return int(match.group(1))

	return None


# --------------------------------------------------
# CARGA DE MODELOS E INFERENCIA
# --------------------------------------------------

def denorm(tensor):
	return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def tensor_to_image(tensor):
	tensor = tensor.detach().cpu().clamp(0.0, 1.0)
	return tensor.permute(1, 2, 0).numpy()


def load_image_tensor(image_path):
	transform = transforms.Compose([
		transforms.Resize((IMG_SIZE, IMG_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize([0.5] * 3, [0.5] * 3),
	])

	image = Image.open(image_path).convert("RGB")
	return transform(image)


def build_circle_mask(center_x, center_y, radius, blur_radius):
	mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
	draw = ImageDraw.Draw(mask)

	draw.ellipse(
		(
			center_x - radius,
			center_y - radius,
			center_x + radius,
			center_y + radius,
		),
		fill=0,
	)

	if blur_radius > 0:
		mask = mask.filter(
			ImageFilter.GaussianBlur(radius=float(blur_radius))
		)

	mask_array = np.asarray(mask, dtype=np.float32) / 255.0
	return torch.from_numpy(mask_array).unsqueeze(0)


def load_state_dict(checkpoint_path):
	try:
		checkpoint = torch.load(
			checkpoint_path,
			map_location=DEVICE,
			weights_only=True,
		)
	except TypeError:
		checkpoint = torch.load(
			checkpoint_path,
			map_location=DEVICE,
		)

	if isinstance(checkpoint, dict):
		if "G" in checkpoint:
			checkpoint = checkpoint["G"]
		elif "state_dict" in checkpoint:
			checkpoint = checkpoint["state_dict"]
		elif "model_state_dict" in checkpoint:
			checkpoint = checkpoint["model_state_dict"]

	if not isinstance(checkpoint, dict):
		raise ValueError(
			f"Formato de checkpoint no reconocido: {checkpoint_path}"
		)

	return {
		key.replace("module.", "", 1): value
		for key, value in checkpoint.items()
	}


def get_model(label, model_cache):
	if label in model_cache:
		return model_cache[label]

	checkpoint_path = experiment_path(label, "checkpoint")

	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"No se encontró el checkpoint de {label}: {checkpoint_path}"
		)

	model = UNET(
		in_channels=4,
		out_channels=3,
	).to(DEVICE)

	model.load_state_dict(
		load_state_dict(checkpoint_path),
		strict=True,
	)

	model.eval()
	model_cache[label] = model

	return model


def create_mask(label, mask_spec):
	return build_circle_mask(
		center_x=mask_spec[0],
		center_y=mask_spec[1],
		radius=mask_spec[2],
		blur_radius=get_blur_radius(label),
	)


def make_input_visual(gt, mask):
	return denorm(gt * mask)


def reconstruct(model, gt, mask):
	gt_device = gt.to(DEVICE)
	mask_device = mask.to(DEVICE)

	masked = gt_device * mask_device
	condition = torch.cat(
		[masked, mask_device],
		dim=0,
	).unsqueeze(0)

	with torch.no_grad():
		fake = model(condition).squeeze(0)

	hole = 1.0 - mask_device
	composed = gt_device * mask_device + fake * hole

	return denorm(composed).cpu()


# --------------------------------------------------
# FIGURAS DE MUESTRAS CONTROLADAS
# --------------------------------------------------

def resolve_column_image(column, gt, mask_spec, model_cache):
	kind = column["kind"]

	if kind == "original":
		return denorm(gt)

	label = column["experiment"]
	mask = create_mask(label, mask_spec)

	if kind == "input":
		return make_input_visual(gt, mask)

	if kind == "output":
		model = get_model(label, model_cache)
		return reconstruct(model, gt, mask)

	raise ValueError(f"Tipo de columna no reconocido: {kind}")


def create_comparison_grid(filename, columns):
	model_cache = {}

	fig, axes = plt.subplots(
		nrows=len(SAMPLE_IMAGES),
		ncols=len(columns),
		figsize=(
			2.6 * len(columns),
			2.45 * len(SAMPLE_IMAGES),
		),
		squeeze=False,
	)

	for row, (image_name, mask_spec) in enumerate(
		zip(SAMPLE_IMAGES, MASK_SPECS)
	):
		image_path = RECORTES_DIR / image_name

		if not image_path.exists():
			raise FileNotFoundError(
				f"No existe el recorte: {image_path}"
			)

		gt = load_image_tensor(image_path)

		for col, column in enumerate(columns):
			ax = axes[row, col]

			image = resolve_column_image(
				column,
				gt,
				mask_spec,
				model_cache,
			)

			ax.imshow(tensor_to_image(image))
			ax.set_xticks([])
			ax.set_yticks([])

			for spine in ax.spines.values():
				spine.set_visible(False)

			if row == 0:
				ax.set_title(
					column["label"],
					fontsize=14,
					pad=8,
				)
				
	ax.margins(x=0.05, y=0.08)			

	fig.tight_layout()

	output_path = OUTPUT_DIR / filename
	fig.savefig(
		output_path,
		dpi=300,
		bbox_inches="tight",
	)
	plt.close(fig)

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	print(f"Figura creada: {output_path}")


# --------------------------------------------------
# MÉTRICAS Y DIAGRAMA L1-LPIPS
# --------------------------------------------------

def classify_experiment(label):
	if label.startswith("EXP_13"):
		return "Dos fases"

	if get_blur_radius(label) > 0:
		return "Máscara suavizada"

	return "Máscara dura"


def collect_test_metrics():
	rows = []

	for label in EXPERIMENT_ORDER:
		metrics_path = experiment_path(label, "test_metrics")

		if not metrics_path.exists():
			print(f"Aviso: no existe {metrics_path}")
			continue

		metrics = read_json(metrics_path)

		l1_hole = find_metric(
			metrics,
			[
				"L1_hole",
				"l1_hole",
				"loss_l1_hole",
				"hole_l1",
			],
		)

		lpips_hole = find_metric(
			metrics,
			[
				"LPIPS_hole",
				"lpips_hole",
				"LPIPS",
				"lpips",
			],
		)

		if l1_hole is None or lpips_hole is None:
			print(
				f"Aviso: faltan L1_hole o LPIPS_hole en {label}"
			)
			continue

		rows.append(
			{
				"experiment": label,
				"best_epoch": read_best_epoch(label),
				"l1_hole": l1_hole,
				"lpips_hole": lpips_hole,
				"blur_radius": get_blur_radius(label),
				"group": classify_experiment(label),
			}
		)

	return rows


def write_metrics_csv(rows):
	output_path = OUTPUT_DIR / "patchgan_resumen_metricas.csv"

	with open(
		output_path,
		"w",
		encoding="utf-8",
		newline="",
	) as file:
		writer = csv.writer(file)

		writer.writerow(
			[
				"Experimento",
				"Época seleccionada",
				"L1_hole",
				"LPIPS_hole",
				"Radio de suavizado",
				"Familia",
			]
		)

		for row in rows:
			writer.writerow(
				[
					row["experiment"],
					row["best_epoch"],
					row["l1_hole"],
					row["lpips_hole"],
					row["blur_radius"],
					row["group"],
				]
			)

	print(f"CSV creado: {output_path}")


def create_l1_lpips_scatter(rows):
	if not rows:
		print("No hay métricas suficientes para crear el diagrama.")
		return

	fig, ax = plt.subplots(figsize=(10, 7.5))

	group_order = [
		"Máscara dura",
		"Máscara suavizada",
		"Dos fases",
	]

	markers = {
		"Máscara dura": "o",
		"Máscara suavizada": "s",
		"Dos fases": "^",
	}

	for group in group_order:
		group_rows = [
			row
			for row in rows
			if row["group"] == group
		]

		if not group_rows:
			continue

		ax.scatter(
			[row["l1_hole"] for row in group_rows],
			[row["lpips_hole"] for row in group_rows],
			s=105,
			marker=markers[group],
			label=group,
		)

		for row in group_rows:
			offset = ANNOTATION_OFFSETS.get(
				row["experiment"],
				(6, 6),
			)

			fontweight = (
				"bold"
				if row["experiment"] == "EXP_08"
				else "normal"
			)

			ax.annotate(
				row["experiment"],
				(
					row["l1_hole"],
					row["lpips_hole"],
				),
				xytext=offset,
				textcoords="offset points",
				fontsize=10,
				fontweight=fontweight,
			)

	ax.set_xlabel(
		"L1 en región oculta sobre prueba",
		fontsize=14,
	)
	ax.set_ylabel(
		"Distancia LPIPS sobre prueba",
		fontsize=14,
	)
	ax.set_title(
		"Relación entre L1 y distancia perceptual LPIPS",
		fontsize=16,
	)

	ax.grid(True, alpha=0.3)
	ax.legend(
		frameon=False,
		fontsize=11,
	)

	fig.tight_layout()

	output_path = OUTPUT_DIR / "patchgan_l1_lpips_prueba.png"
	fig.savefig(
		output_path,
		dpi=300,
		bbox_inches="tight",
	)
	plt.close(fig)

	print(f"Figura creada: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
	if not RECORTES_DIR.exists():
		raise FileNotFoundError(
			f"No existe la carpeta de recortes: {RECORTES_DIR}"
		)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	print(f"Directorio de trabajo: {ROOT_DIR}")
	print(f"Dispositivo: {DEVICE}")

	# 1. Efecto de eliminar la pérdida L1.
	create_comparison_grid(
		"patchgan_01_l1_vs_sin_l1.png",
		[
			{
				"kind": "original",
				"label": "Original",
			},
			{
				"kind": "input",
				"experiment": "EXP_01",
				"label": "Imagen\nenmascarada",
			},
			{
				"kind": "output",
				"experiment": "EXP_01",
				"label": "EXP_01\nGAN + L1",
			},
			{
				"kind": "output",
				"experiment": "EXP_02",
				"label": "EXP_02\nsolo GAN",
			},
		],
	)

	# 2. Incorporación de LPIPS.
	create_comparison_grid(
		"patchgan_02_lpips_07_vs_08.png",
		[
			{
				"kind": "original",
				"label": "Original",
			},
			{
				"kind": "input",
				"experiment": "EXP_07",
				"label": "Imagen\nenmascarada",
			},
			{
				"kind": "output",
				"experiment": "EXP_07",
				"label": "EXP_07\nsin LPIPS",
			},
			{
				"kind": "output",
				"experiment": "EXP_08",
				"label": "EXP_08\nLPIPS = 0.1",
			},
		],
	)

	# 3. Máscara dura frente a máscaras suavizadas.
	create_comparison_grid(
		"patchgan_03_mascaras_09_10_11.png",
		[
			{
				"kind": "original",
				"label": "Original",
			},
			{
				"kind": "input",
				"experiment": "EXP_09",
				"label": "Entrada\ndura",
			},
			{
				"kind": "output",
				"experiment": "EXP_09",
				"label": "EXP_09",
			},
			{
				"kind": "input",
				"experiment": "EXP_10",
				"label": "Máscara suave\nr = 2",
			},
			{
				"kind": "output",
				"experiment": "EXP_10",
				"label": "EXP_10",
			},
			{
				"kind": "input",
				"experiment": "EXP_11",
				"label": "Máscara suave\nr = 3",
			},
			{
				"kind": "output",
				"experiment": "EXP_11",
				"label": "EXP_11",
			},
		],
	)

	# 4. Resumen cuantitativo global.
	rows = collect_test_metrics()
	create_l1_lpips_scatter(rows)
	write_metrics_csv(rows)

	print("\nProceso terminado.")
	print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()

