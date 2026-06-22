# Primera serie de experimentos cGAN sobre AFHQ Cats

Esta carpeta contiene la primera familia de experimentos realizada sobre imágenes de gatos del dataset AFHQ. El objetivo fue evaluar distintas configuraciones de modelos adversariales condicionados para la reconstrucción de regiones enmascaradas.

La serie se utilizó como fase exploratoria antes de aplicar modelos de inpainting al caso de estudio de Oberranna.

## Estructura

- `src/`: funciones comunes para carga de datos, generación de máscaras y utilidades.
- `experiments/`: código y configuración de cada experimento.
- `checkpoints_selected/`: checkpoint seleccionado del generador para cada configuración.
- `results/`: logs, curvas de pérdidas y muestras generadas.

## Experimentos incluidos

- `afhq_01_configuracion_inicial`
- `afhq_02_reduccion_lambda_adv`
- `afhq_03_bloques_residuales_dilatados`
- `afhq_04_warmup_vgg16`
- `afhq_05_ajuste_adversarial_perceptual`
- `afhq_06_entrenamiento_prolongado_lr_decay`
- `afhq_07_perdida_bordes_sobel`

Cada carpeta de `experiments/` contiene el script de entrenamiento, el modelo utilizado y la configuración correspondiente.

## Datos

Las imágenes seleccionadas para evaluación se encuentran en:

```text
../afhq_cats_selected/

El archivo ../afhq_cats.zip contiene el material de partida asociado al dataset.
