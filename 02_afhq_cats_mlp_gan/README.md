
---

## `02_afhq_cats_mlp_gan/README.md`

```md
# Segunda serie AFHQ Cats: U-Net con discriminador MLP

Esta carpeta contiene los experimentos de inpainting realizados sobre AFHQ Cats mediante una U-Net como generador y un discriminador global de tipo MLP.

Esta familia se desarrolló después de la primera serie cGAN para estudiar una formulación más específica del problema de reconstrucción de regiones ocultas.

## Estructura

- `src/`: implementación del generador, discriminador, dataset, máscaras, entrenamiento y pruebas.
- `configs/`: configuraciones de los tres experimentos conservados.
- `checkpoints_selected/`: pesos finales seleccionados del generador.
- `results/`: curvas de pérdidas, métricas y muestras de cada experimento.
- `analysis/`: figuras y trazabilidad de la comparación final.
- `experimentos_descartados/`: resultados intermedios no utilizados en el análisis final.

## Experimentos incluidos

- `afhq_mlp_01`
- `afhq_mlp_02`
- `afhq_mlp_03`

Cada experimento dispone de su configuración, curvas de entrenamiento, muestras generadas y checkpoint final.

## Datos

Las imágenes seleccionadas para evaluación se encuentran en:

```text
../afhq_cats_selected/
