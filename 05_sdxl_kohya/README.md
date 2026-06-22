
---

## `05_sdxl_kohya/README.md`

```md
# SDXL y LoRA con Kohya

Esta carpeta contiene los resultados seleccionados de los experimentos realizados con Stable Diffusion XL y ajuste LoRA mediante Kohya.

El objetivo de esta línea fue explorar la generación de variantes visuales compatibles con la apariencia de las pinturas murales de Oberranna.

## Estructura

- `samples/`: imágenes generadas durante y al final del entrenamiento.
- `curves/`: evolución de la pérdida y de la tasa de aprendizaje.
- `comparative/`: comparativas visuales de resultados generados para distintos motivos del mural.

## Contenido

La carpeta incluye:

- Muestras generadas en distintas épocas del entrenamiento.
- Curva media de pérdida por época.
- Evolución de la tasa de aprendizaje de la U-Net.
- Comparativas visuales para los motivos del ángel, el león y la abadía.

## Reproducibilidad

La configuración y el seguimiento de los experimentos se realizaron mediante la interfaz de TensorBoard, por lo que no se utilizaron scripts independientes para automatizar el entrenamiento.

Esta carpeta conserva las muestras, curvas y comparativas exportadas durante la experimentación. No incluye los pesos LoRA finales ni el dataset curado utilizado para el ajuste, por lo que permite revisar los resultados obtenidos, aunque no reproducir de forma autónoma el entrenamiento completo.

