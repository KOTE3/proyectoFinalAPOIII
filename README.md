# Clasificación automática de calidad y tamaño de frutas

Proyecto final — Algoritmos y Programación III (2026-2), Universidad Icesi.

Sistema de visión por computador que, a partir de una fotografía de **una fruta individual sobre fondo uniforme**, predice:

1. **Clase de calidad** (≥ 3 categorías definidas por el grupo — ver [`docs/fases/01_comprension_negocio.md`](docs/fases/01_comprension_negocio.md)).
2. **Tamaño relativo** (pequeño / mediano / grande) a partir del diámetro equivalente normalizado.

El trabajo sigue la metodología **CRISP-DM**; el repositorio está organizado según sus fases.

## Integrantes

| Nombre | Código | Rol principal |
|---|---|---|
| _Por completar_ | | |
| _Por completar_ | | |
| _Por completar_ | | |

## Estructura del repositorio

```
.
├── docs/
│   ├── enunciado/            # Enunciado del proyecto y competencias
│   └── fases/                # Documentación de cada fase CRISP-DM (evidencias)
├── data/                     # (no versionado, salvo etiquetas) ver data/README.md
│   ├── raw/kaggle/           # Dataset Fruit Quality Classification (Kaggle)
│   ├── raw/propias/          # Fotos recolectadas por el grupo
│   ├── interim/segmentadas/  # Frutas recortadas de mixed_quality
│   ├── processed/            # Matrices de características, splits
│   └── labels/               # Anotaciones manuales (CSV, sí versionado)
├── notebooks/                # EDA, modelado y evaluación (numerados por fase)
├── src/fruitq/               # Paquete reutilizable: segmentación, tamaño, características
├── scripts/                  # Scripts de línea de comandos (descarga, segmentación…)
├── models/                   # Modelos entrenados (no versionados)
├── app/                      # Interfaz gráfica (Streamlit)
├── reports/
│   ├── figures/              # Gráficas vectoriales (PDF/SVG)
│   └── informe/              # Informe final IEEE (máx. 7 páginas)
└── tests/                    # Pruebas unitarias
```

### Mapa CRISP-DM → repositorio

| Fase | Documento | Código / notebooks |
|---|---|---|
| 1. Comprensión del negocio | `docs/fases/01_comprension_negocio.md` | — |
| 2. Comprensión de los datos | `docs/fases/02_comprension_datos.md` | `notebooks/02_eda.ipynb`, `scripts/download_data.py` |
| 3. Preparación de los datos | `docs/fases/03_preparacion_datos.md` | `src/fruitq/segmentation.py`, `src/fruitq/features.py`, `scripts/segment_mixed.py` |
| 4. Modelado | `docs/fases/04_modelado.md` | `notebooks/04_*` |
| 5. Evaluación | `docs/fases/05_evaluacion.md` | `notebooks/05_evaluacion.ipynb` |
| 6. Despliegue | `docs/fases/06_despliegue.md` | `app/` |

## Instalación

Requiere Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Datos

1. Crear un token de API en Kaggle (*Settings → API → Create New Token*) y guardarlo, solo el texto `KGAT_...`, en el archivo `~/.kaggle/access_token` (Windows: `C:\Users\<usuario>\.kaggle\access_token`, sin extensión). **Nunca** subir el token al repositorio.
2. Descargar el dataset:

```bash
python scripts/download_data.py
```

3. Recortar cada fruta de las imágenes de `mixed_quality`:

```bash
python scripts/segment_mixed.py --input data/raw/kaggle --output data/interim/segmentadas
```

Ver [`data/README.md`](data/README.md) para el protocolo de recolección y etiquetado de las fotos propias.

## Referencias de terceros

- Dataset: ryandpark, *Fruit Quality Classification*, Kaggle (verificar autor y licencia en la página). https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- Bibliotecas: OpenCV, scikit-learn, XGBoost, PyTorch, Streamlit (ver `requirements.txt`).

Todo código o dato de terceros usado en el proyecto debe citarse aquí y en el informe.
