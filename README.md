# Clasificación automática de calidad y tamaño de frutas

Proyecto final del curso **Algoritmos y Programación III** (2026-2), Ingeniería de Sistemas, Universidad Icesi, Cali, Colombia.

#### Estado del proyecto: Activo

- **Completado:** clasificación buena/mala con modelos clásicos y CNN, 45 fotos propias anotadas, tamaño en centímetros, tabla de decisiones éticas y matriz de impactos.
- **En curso:** etiquetado de recortes, clase `regular`, evaluación en las fotos propias, interfaz con cámara, diagramas, informe y video.

## Integrantes

| Nombre | Código | Rol principal |
|---|---|---|
| Sebastián Mejía | _Por completar_ | Responsable técnico: datos, modelos y aplicación |
| _Por completar_ | | Informe e investigación |
| _Por completar_ | | Ética, impacto, diagramas y video |

**Profesor:** Milton Sarria.

## Objetivo

En plazas de mercado y agroindustrias, la clasificación de frutas por calidad y tamaño se hace a mano. Es un proceso lento y subjetivo que genera pérdidas económicas y desperdicio de alimentos.

Este proyecto desarrolla un sistema de visión por computador que, a partir de una fotografía de **una fruta individual sobre fondo uniforme**, predice:

1. su **clase de calidad**: `buena`, `regular` o `mala`;
2. su **tamaño**: pequeño, mediano o grande, además del diámetro estimado.

El objetivo es ofrecer un criterio objetivo y reproducible que apoye a pequeños productores y comerciantes, y que reduzca el desperdicio por clasificaciones erróneas.

## Métodos utilizados

- Metodología **CRISP-DM**.
- Visión por computador:
  - segmentación por color en CIELab con k-means y Otsu;
  - GrabCut;
  - watershed.
- Ingeniería de características: color, forma, textura e indicadores de defecto.
- Aprendizaje automático: regresión logística, SVM (RBF), Random Forest y XGBoost, con búsqueda en rejilla y validación cruzada agrupada.
- Aprendizaje profundo: CNN pequeña entrenada desde cero.
- Evaluación:
  - F1-macro, recall de la clase `mala` y AUC-ROC;
  - prueba de McNemar;
  - curvas de calibración;
  - Grad-CAM;
  - estudio de ablación.

## Tecnologías

- **Lenguaje:** Python 3.11.
- **Procesamiento de imágenes:** OpenCV, scikit-image.
- **Modelos:** scikit-learn, XGBoost, PyTorch.
- **Datos y gráficas:** NumPy, pandas, Matplotlib, seaborn.
- **Notebooks:** Jupyter.
- **Interfaz:** Streamlit.
- **Pruebas:** pytest.

## Descripción del proyecto

**Datos.**
- Dataset público *Fruit Quality Classification* (Kaggle): 6 frutas (manzana, banano, guayaba, limón, naranja y granada) etiquetadas como buenas o malas. Después de la limpieza quedan 17 264 imágenes.
- Recortes individuales obtenidos al segmentar la carpeta `mixed_quality`.
- Fotos propias del grupo: banano, manzana y guayaba, en tres calidades, tomadas con un protocolo controlado.

**Preguntas que guían el trabajo.**
- ¿Qué tan bien separan las calidades el color, la forma y la textura frente a una CNN que aprende sus propias características?
- ¿Los modelos generalizan a fotos tomadas con otra cámara y en otras condiciones?
- ¿Se puede estimar el tamaño físico a partir de una sola foto?

**Retos encontrados.**
- Kaggle solo trae dos calidades. La clase `regular` se obtiene con umbrales sobre $P(\text{mala} \mid x)$, calibrados con recortes y fotos etiquetadas por el grupo.
- Cerca del 30 % de las imágenes son casi duplicadas. Se agruparon por firma perceptual y por ráfagas de captura, y la partición train/val/test respeta esos grupos para evitar fuga de información.
- Los fondos de Kaggle no son uniformes. Por eso la segmentación modela el fondo a partir del borde de cada imagen.
- La nitidez de la imagen funciona como un posible atajo, porque refleja la cámara y no el defecto. Se analizó con un estudio de ablación.
- XGBoost está sobreconfiado: sus probabilidades caen casi siempre cerca de 0 o de 1, lo que dificulta definir la zona `regular`.

**Resultados actuales (test, 2 540 imágenes, buena/mala).**

| Modelo | F1-macro | Recall (mala) | AUC |
|---|---|---|---|
| XGBoost (57 características) | **0.964** | 0.950 | 0.990 |
| CNN desde cero (72 434 parámetros) | 0.940 | 0.939 | 0.983 |

## Notebooks y entregables destacados

- [Análisis exploratorio](notebooks/02_eda.ipynb)
- [Modelos clásicos: búsqueda de hiperparámetros, ablación y errores](notebooks/04_modelos_clasicos.ipynb)
- [CNN y comparación con XGBoost: McNemar, calibración y Grad-CAM](notebooks/05_cnn_y_comparacion.ipynb)
- [Arquitectura del sistema](docs/arquitectura.md)
- [Documentación por fase CRISP-DM](docs/fases/)
- Informe IEEE: `reports/informe/` (en elaboración)
- Video de presentación: _enlace por completar_

## Inicio rápido

```bash
git clone https://github.com/KOTE3/proyectoFinalAPOIII.git
cd proyectoFinalAPOIII
python -m venv .venv
.venv\Scripts\activate              # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

- Instalación detallada, versiones probadas y reproducción completa: [`docs/instalacion.md`](docs/instalacion.md).
- Organización de los datos: [`data/README.md`](data/README.md).
- Etiquetado de recortes: [`docs/ETIQUETADO.md`](docs/ETIQUETADO.md).
- Reparto de tareas y protocolo de fotografías: [`docs/PLAN_DE_TRABAJO.md`](docs/PLAN_DE_TRABAJO.md).
- Traspaso del trabajo técnico, datos compartidos y tareas pendientes: [`docs/TRASPASO.md`](docs/TRASPASO.md).

## Estructura del repositorio

```
.
├── docs/
│   ├── enunciado/            # Enunciado del proyecto y competencias
│   ├── fases/                # Documentación de cada fase CRISP-DM (evidencias)
│   ├── arquitectura.md       # Diseño del sistema y de los modelos
│   ├── instalacion.md        # Instalación y reproducción
│   ├── ETIQUETADO.md         # Guía de etiquetado de recortes
│   ├── PLAN_DE_TRABAJO.md    # Reparto de tareas y protocolo de fotografías
│   └── TRASPASO.md           # Estado entregado y especificación de lo pendiente
├── data/                     # (no versionado, salvo etiquetas) ver data/README.md
│   ├── raw/kaggle/           # Dataset Fruit Quality Classification (Kaggle)
│   ├── raw/propias/          # Fotos recolectadas por el grupo
│   ├── interim/segmentadas/  # Frutas recortadas de mixed_quality
│   ├── processed/            # Manifiesto, características y recortes para la CNN
│   └── labels/               # Anotaciones manuales (CSV, sí versionado)
├── src/fruitq/               # Paquete: segmentación, características, tamaño, datos, CNN
├── scripts/                  # Flujo de datos y entrenamiento por línea de comandos
├── notebooks/                # EDA, modelado y evaluación (numerados por fase)
├── models/                   # Modelos finales versionados (XGBoost y CNN)
├── reports/
│   ├── cnn/                  # Historias de entrenamiento, predicciones y resúmenes de la CNN
│   ├── figures/              # Gráficas vectoriales (PDF) y PNG
│   └── informe/              # Informe final IEEE (máx. 7 páginas)
├── app/                      # Interfaces Streamlit
├── tests/                    # Pruebas unitarias
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

**Equivalencia con la estructura sugerida del curso:**
- `reports/` cumple el papel de `experiments/`:
  - `reports/cnn/` corresponde a `logs/`;
  - `reports/figures/` y las tablas CSV de `reports/` corresponden a `results/`.
- `models/` corresponde a `checkpoints/`.
- El código fuente está en `src/fruitq/` y los puntos de entrada de entrenamiento y evaluación están en `scripts/`.

### Mapa CRISP-DM → repositorio

| Fase | Documento | Código / notebooks |
|---|---|---|
| 1. Comprensión del negocio | `docs/fases/01_comprension_negocio.md` | — |
| 2. Comprensión de los datos | `docs/fases/02_comprension_datos.md` | `notebooks/02_eda.ipynb`, `scripts/download_data.py` |
| 3. Preparación de los datos | `docs/fases/03_preparacion_datos.md` | `scripts/prepare_dataset.py`, `scripts/segment_mixed.py`, `scripts/extract_features.py`, `scripts/cache_tiles.py`, `src/fruitq/` |
| 4. Modelado | `docs/fases/04_modelado.md` | `notebooks/04_modelos_clasicos.ipynb`, `scripts/train_cnn.py` |
| 5. Evaluación | `docs/fases/05_evaluacion.md` | `notebooks/05_cnn_y_comparacion.ipynb` |
| 6. Despliegue | `docs/fases/06_despliegue.md` | `app/` |
| Ética e impacto | `docs/fases/07_etica_impactos.md` | — |

## Referencias de terceros

- Dataset: ryandpark, *Fruit Quality Classification*, Kaggle. https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- Fuente original del dataset: V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications", *Data in Brief*, vol. 40, art. 107686, 2022. doi:10.1016/j.dib.2021.107686. El artículo se publica con licencia CC BY 4.0, que exige citar a sus autores.
- Bibliotecas: OpenCV, scikit-image, scikit-learn, XGBoost, PyTorch, Streamlit, NumPy, pandas y Matplotlib (ver `requirements.txt`).

Todo código o dato de terceros usado en el proyecto se cita aquí y en el informe.

## Licencia

El código se distribuye bajo la licencia MIT (ver [`LICENSE`](LICENSE)). El dataset de Kaggle y las bibliotecas conservan sus propias licencias.
