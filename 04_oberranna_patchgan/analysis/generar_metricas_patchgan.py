from pathlib import Path
import json
import re

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "figuras_patchgan_curvas"

EXPERIMENTS = {
	"EXP_01": {
		"folder": "exp01",
		"metrics": "01_metrics.json",
		"best": "01_best.txt",
	},
	"EXP_02": {
		"folder": "exp02",
		"metrics": "02_metrics.json",
		"best": "02_best.txt",
	},
	"EXP_07": {
		"folder": "exp07",
		"metrics": "07_metrics.json",
		"best": "07_best.txt",
	},
	"EXP_08": {
		"folder": "exp08",
		"metrics": "08_metrics.json",
		"best": "08_best.txt",
	},
	"EXP_09": {
		"folder": "exp09",
		"metrics": "09_metrics.json",
		"best": "09_best.txt",
	},
	"EXP_10": {
		"folder": "exp10",
		"metrics": "10_metrics.json",
		"best": "10_best.txt",
	},
	"EXP_11": {
		"folder": "exp11",
		"metrics": "11_metrics.json",
		"best": "11_best.txt",
	},
}

METRICS = {
	"L1_hole_val": {
		"candidates": [
			"L1_hole_val",
			"l1_hole_val",
			"loss_l1_hole_val",
		],
		"title": "L1 en región oculta",
		"ylabel": "L1 de validación",
	},
	"L1_boundary_val": {
		"candidates": [
			"L1_boundary_val",
			"l1_boundary_val",
			"loss_l1_boundary_val",
		],
		"title": "L1 en el borde de la máscara",
		"ylabel": "L1 de validación",
	},
	"LPIPS_hole_val": {
		"candidates": [
			"LPIPS_hole_val",
			"lpips_hole_val",
			"LPIPS_val",
			"lpips_val",
		],
		"title": "Distancia perceptual LPIPS",
		"ylabel": "LPIPS de validación",
	},
}

COMPARISONS = [
	{
		"filename": "patchgan_curvas_01_vs_02.png",
		"experiments": ["EXP_01", "EXP_02"],
		"metrics": [
			"L1_hole_val",
			"LPIPS_hole_val",
		],
	},
	{
		"filename": "patchgan_curvas_07_vs_08.png",
		"experiments": ["EXP_07", "EXP_08"],
		"metrics": [
			"L1_hole_val",
			"LPIPS_hole_val",
		],
	},
	{
		"filename": "patchgan_curvas_09_10_11.png",
		"experiments": ["EXP_09", "EXP_10", "EXP_11"],
		"metrics": [
			"L1_hole_val",
			"L1_boundary_val",
			"LPIPS_hole_val",
		],
	},
]


# --------------------------------------------------
# LECTURA DE ARCHIVOS
# --------------------------------------------------

def experiment_path(experiment, key):
	info = EXPERIMENTS[experiment]
	return ROOT_DIR / info["folder"] / info[key]


def read_json(path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def read_best_epoch(experiment):
	path = experiment_path(experiment, "best")

	if not path.exists():
		return None

	for line in path.read_text(encoding="utf-8").splitlines():
		match = re.match(r"\s*best_epoch\s*=\s*(\d+)", line)

		if match:
			return int(match.group(1))

	return None


def load_validation_series(experiment, metric_key):
	metrics_path = experiment_path(experiment, "metrics")

	if not metrics_path.exists():
		raise FileNotFoundError(
			f"No existe el archivo: {metrics_path}"
		)

	history = read_json(metrics_path)

	normalised = {
		str(key).lower(): value
		for key, value in history.items()
	}

	for candidate in METRICS[metric_key]["candidates"]:
		values = normalised.get(candidate.lower())

		if not isinstance(values, list):
			continue

		try:
			return np.asarray(
				[
					np.nan if value is None else float(value)
					for value in values
				],
				dtype=float,
			)
		except (TypeError, ValueError):
			continue

	raise KeyError(
		f"No se encontró la serie {metric_key} en {metrics_path}"
	)


# --------------------------------------------------
# VISUALIZACIÓN
# --------------------------------------------------

def selected_value(series, epoch):
	if epoch is None:
		return None

	index = epoch - 1

	if index < 0 or index >= len(series):
		return None

	return series[index]


def draw_metric_curve(axis, experiments, metric_key):
	handles = []
	labels = []

	for experiment in experiments:
		series = load_validation_series(
			experiment,
			metric_key,
		)

		epochs = np.arange(1, len(series) + 1)
		best_epoch = read_best_epoch(experiment)

		line, = axis.plot(
			epochs,
			series,
			linewidth=2.1,
		)

		best_value = selected_value(
			series,
			best_epoch,
		)

		if best_value is not None:
			axis.scatter(
				[best_epoch],
				[best_value],
				s=50,
				color=line.get_color(),
				zorder=3,
			)

		if best_epoch is None:
			legend_label = experiment
		else:
			legend_label = (
				f"{experiment} "
				f"(checkpoint: época {best_epoch})"
			)

		handles.append(line)
		labels.append(legend_label)

	axis.set_title(
		METRICS[metric_key]["title"],
		fontsize=15,
	)
	axis.set_ylabel(
		METRICS[metric_key]["ylabel"],
		fontsize=13,
	)
	axis.grid(
		True,
		alpha=0.3,
	)
	axis.tick_params(
		axis="both",
		labelsize=11,
	)
	axis.margins(
		x=0.02,
		y=0.12,
	)

	return handles, labels


def create_comparison_figure(comparison):
	metric_keys = comparison["metrics"]
	experiments = comparison["experiments"]

	fig, axes = plt.subplots(
		len(metric_keys),
		1,
		figsize=(10, 4.2 * len(metric_keys)),
		sharex=True,
		squeeze=False,
	)

	axes = axes[:, 0]
	legend_handles = []
	legend_labels = []

	for index, metric_key in enumerate(metric_keys):
		handles, labels = draw_metric_curve(
			axes[index],
			experiments,
			metric_key,
		)

		if index == 0:
			legend_handles = handles
			legend_labels = labels

	axes[-1].set_xlabel(
		"Época",
		fontsize=13,
	)

	fig.legend(
		legend_handles,
		legend_labels,
		loc="upper center",
		ncol=len(experiments),
		frameon=False,
		fontsize=11,
		bbox_to_anchor=(0.5, 0.995),
	)

	fig.tight_layout(
		rect=(0, 0, 1, 0.93),
	)

	output_path = OUTPUT_DIR / comparison["filename"]

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
	OUTPUT_DIR.mkdir(
		parents=True,
		exist_ok=True,
	)

	for comparison in COMPARISONS:
		create_comparison_figure(comparison)

	print("\nProceso terminado.")
	print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
