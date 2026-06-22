
---

## `04_oberranna_patchgan/README.md`

```md
# Oberranna: experimentos con PatchGAN

Esta carpeta contiene la serie principal de experimentos de inpainting sobre Oberranna basada en una U-Net generadora y un discriminador PatchGAN.

La serie evalúa diferentes configuraciones de pérdidas, máscaras, duración de entrenamiento y estrategias de entrenamiento en varias fases.

## Estructura

- `src/`: código común para preparación de datos, máscaras, modelo, discriminador PatchGAN y entrenamiento.
- `experiments/`: configuraciones, divisiones de datos, métricas, curvas, muestras y checkpoints de cada experimento.
- `analysis/`: scripts para generar las figuras y resúmenes comparativos.
- `results/`: figuras comparativas finales de la serie experimental.

## Experimentos incluidos

- `exp01`
- `exp02`
- `exp07`
- `exp08`
- `exp09`
- `exp10`
- `exp11`
- `exp12`
- `exp13`

El experimento `exp13` se divide en dos fases:

- `exp13A`
- `exp13B`

Cada experimento contiene su configuración, división de datos, métricas, curvas de entrenamiento y checkpoint seleccionado.

## Datos

Los recortes utilizados se encuentran en:

```text
../recortes128_selected/

## Resultados

Las comparativas finales se encuentran en:

results/figuras_patchgan/
results/figuras_patchgan_curvas/
