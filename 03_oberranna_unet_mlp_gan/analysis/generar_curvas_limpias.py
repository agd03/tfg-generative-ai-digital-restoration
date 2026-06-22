import json
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURACIÓN
# ==========================================

LOSSES_FILE = "losses.json"
OUTPUT_FILE = "oberranna_curvas_limpias.png"


# ==========================================
# CARGA DE DATOS
# ==========================================

with open(LOSSES_FILE, "r", encoding="utf-8") as f:
	data = json.load(f)

required_keys = ["G_L1", "G_adv"]
missing = [k for k in required_keys if k not in data]

if missing:
	raise KeyError(
		f"Faltan claves en {LOSSES_FILE}: {missing}. "
		f"Claves disponibles: {list(data.keys())}"
	)

g_l1 = np.asarray(data["G_L1"], dtype=float)
g_adv = np.asarray(data["G_adv"], dtype=float)

if len(g_l1) != len(g_adv):
	raise ValueError(
		f"Las longitudes no coinciden: G_L1={len(g_l1)}, G_adv={len(g_adv)}"
	)

epochs = np.arange(1, len(g_l1) + 1)


# ==========================================
# FIGURA
# ==========================================

fig, axes = plt.subplots(
	2,
	1,
	figsize=(12, 8.5),
	sharex=True
)

ax_l1, ax_adv = axes

# Panel superior: G_L1
ax_l1.plot(
	epochs,
	g_l1,
	linewidth=2.3,
	label="L1 en región oculta"
)
ax_l1.set_ylabel("L1 en región oculta", fontsize=18)
ax_l1.tick_params(axis="both", labelsize=14)
ax_l1.grid(True, alpha=0.3)
ax_l1.legend(fontsize=13, frameon=False, loc="upper right")

# Panel inferior: G_adv
ax_adv.plot(
	epochs,
	g_adv,
	linewidth=2.3,
	label="Pérdida adversarial del generador"
)
ax_adv.set_xlabel("Época", fontsize=18)
ax_adv.set_ylabel("Pérdida adversarial del generador", fontsize=18)
ax_adv.tick_params(axis="both", labelsize=14)
ax_adv.grid(True, alpha=0.3)
ax_adv.legend(fontsize=13, frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(
	OUTPUT_FILE,
	dpi=300,
	bbox_inches="tight"
)
plt.close(fig)

print(f"Gráfica guardada en: {OUTPUT_FILE}")
