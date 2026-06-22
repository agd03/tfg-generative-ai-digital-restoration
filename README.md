## `README.md` — raíz

# Código entregable del TFG

Este directorio reúne el código, las configuraciones, los checkpoints seleccionados y los resultados de los experimentos desarrollados en el Trabajo de Fin de Grado.

## Estructura

* `01_afhq_cgan/`: primera serie de experimentos adversariales sobre AFHQ Cats.
* `02_afhq_cats_mlp_gan/`: experimentos de inpainting con U-Net y discriminador global MLP sobre AFHQ Cats.
* `03_oberranna_unet_mlp_gan/`: aplicación de la familia U-Net + MLP al caso de estudio de Oberranna.
* `04_oberranna_patchgan/`: experimentos PatchGAN sobre los recortes de Oberranna.
* `05_sdxl_kohya/`: resultados de los experimentos con SDXL y LoRA mediante Kohya.
* `afhq_cats_selected/`: imágenes de evaluación seleccionadas de AFHQ Cats.
* `recortes128_selected/`: recortes de Oberranna utilizados en los experimentos de inpainting.
* `afhq_cats.zip`, `fotos-Burg Oberranna.zip` y `recortes_1024_oberranna.zip`: datos de partida archivados.

Cada familia incluye su propio `README.md` con información específica sobre su organización.

## Entorno

Los experimentos se desarrollaron en Python con PyTorch. Los scripts conservan parte de su estructura original y utilizan rutas relativas, por lo que se recomienda ejecutarlos desde la carpeta de cada familia experimental.

Los resultados almacenados permiten revisar las configuraciones evaluadas, pero el entrenamiento puede no reproducir exactamente las mismas salidas debido a la aleatoriedad inherente al proceso.

---

## `01_afhq_cgan/README.md`

# AFHQ Cats: primera serie cGAN

Primera serie de experimentos adversariales sobre imágenes de AFHQ Cats con máscaras sintéticas. Su objetivo fue analizar distintas configuraciones de entrenamiento y arquitectura antes de aplicar modelos de inpainting al caso de estudio patrimonial.

## Contenido

* `src/`: utilidades comunes para carga de datos, generación de máscaras y visualización.
* `experiments/`: código, configuración y modelo de cada una de las siete configuraciones evaluadas.
* `checkpoints_selected/`: checkpoint del generador seleccionado para cada experimento.
* `results/`: logs de entrenamiento, curvas de pérdidas y muestras generadas.

## Datos

Las imágenes de evaluación seleccionadas se encuentran en `../afhq_cats_selected/`. El archivo `../afhq_cats.zip` contiene el material de partida asociado al dataset.

---

## `02_afhq_cats_mlp_gan/README.md`

# AFHQ Cats: U-Net + discriminador MLP

Segunda serie de experimentos sobre AFHQ Cats. Se evaluó un modelo de inpainting basado en una U-Net como generador y un discriminador global de tipo MLP.

## Contenido

* `src/`: código del modelo, discriminador, carga de datos, máscaras, entrenamiento y pruebas.
* `configs/`: configuraciones de las tres ejecuciones conservadas.
* `checkpoints_selected/`: pesos finales del generador para cada experimento.
* `results/`: curvas de pérdidas, métricas y muestras generadas.
* `analysis/`: figuras y trazabilidad de la comparación entre configuraciones.
* `experimentos_descartados/`: resultados intermedios no utilizados en el análisis final.

## Datos

Las imágenes de evaluación se encuentran en `../afhq_cats_selected/`.

---

## `03_oberranna_unet_mlp_gan/README.md`

# Oberranna: U-Net + discriminador MLP

Aplicación al caso de estudio de Oberranna de la familia experimental U-Net + discriminador global MLP evaluada previamente sobre AFHQ Cats.

## Contenido

* `src/`: código de entrenamiento, modelo generador, discriminador, generación de máscaras y pruebas.
* `checkpoints/`: checkpoint final del generador.
* `results/`: pérdidas, muestras generadas y figuras finales.
* `analysis/`: scripts para generar curvas limpias y rejillas de resultados.

## Datos

Los cinco recortes empleados se encuentran en `../recortes128_selected/`.

## Nota

La configuración de esta ejecución está integrada en el código de entrenamiento. Esta carpeta no incluye actualmente un fichero de configuración independiente.

---

## `04_oberranna_patchgan/README.md`

# Oberranna: experimentos PatchGAN

Serie principal de experimentos PatchGAN sobre los recortes de Oberranna. Se comparan distintas configuraciones de pérdidas, máscaras, duración de entrenamiento y estrategias de entrenamiento en fases.

## Contenido

* `src/`: código común para preparación de datos, máscaras, modelo, discriminador PatchGAN y entrenamiento.
* `experiments/`: configuraciones, divisiones de datos, métricas, curvas y checkpoints de cada experimento.
* `analysis/`: scripts para generar figuras y resúmenes comparativos.
* `results/`: figuras finales utilizadas para analizar la serie experimental.

La carpeta `experiments/exp13/` contiene dos fases diferenciadas: `exp13A` y `exp13B`.

## Datos

Los cinco recortes empleados se encuentran en `../recortes128_selected/`. Si algún script conserva una ruta local anterior, deberá actualizarse antes de ejecutarlo.

---

## `05_sdxl_kohya/README.md`

# SDXL y LoRA con Kohya

Resultados de los experimentos de ajuste LoRA sobre SDXL para generar variantes visuales compatibles con las pinturas murales de Oberranna.

## Contenido

* `samples/`: imágenes generadas durante y al final del entrenamiento.
* `curves/`: evolución de la pérdida y de la tasa de aprendizaje.
* `comparative/`: comparativas visuales generadas para distintos motivos del mural.

## Limitación de reproducibilidad

Esta carpeta contiene los resultados seleccionados, pero no incluye actualmente los scripts de Kohya, la configuración de entrenamiento, los pesos LoRA finales ni el dataset curado utilizado para el ajuste.

Por tanto, su contenido permite revisar los resultados experimentales, pero no reproducir de forma autónoma el entrenamiento LoRA. Para completar la entrega reproducible, será necesario añadir esos elementos.

