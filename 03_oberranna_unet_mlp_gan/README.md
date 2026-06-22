
---

## `03_oberranna_unet_mlp_gan/README.md`

```md
# Oberranna: U-Net con discriminador MLP

Esta carpeta contiene la aplicación al caso de estudio de Oberranna de la familia de modelos basada en una U-Net generadora y un discriminador global MLP.

El objetivo de este experimento fue evaluar la capacidad del modelo para reconstruir regiones ocultas en los recortes de pinturas murales utilizados en el trabajo.

## Estructura

- `src/`: código de entrenamiento, modelo generador, discriminador, máscaras y pruebas.
- `checkpoints/`: checkpoint final del generador.
- `results/`: curvas de pérdidas, muestras generadas y figuras finales.
- `analysis/`: scripts para generar las curvas limpias y la rejilla de resultados.

## Datos

Los recortes utilizados se encuentran en:

```text
../recortes128_selected/

##Resultados principales

La carpeta results/ incluye:

-Curvas de pérdidas del entrenamiento.
-Muestras generadas en distintas épocas.
-Figura comparativa final de las reconstrucciones.
-Figura con curvas de entrenamiento procesadas.

## Nota

Este experimento se apoya en la familia evaluada previamente sobre AFHQ Cats, pero utiliza exclusivamente los recortes del caso de estudio de Oberranna.
