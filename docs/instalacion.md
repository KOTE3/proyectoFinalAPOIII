# Instalación y reproducción

## 1. Requisitos

- Python 3.11.
- Unos 5 GB libres en disco: el dataset de Kaggle ocupa 3,2 GB y los datos procesados 0,8 GB.
- No se requiere GPU: todo el entrenamiento se hizo en CPU.

## 2. Entorno

```bash
git clone https://github.com/KOTE3/proyectoFinalAPOIII.git
cd proyectoFinalAPOIII

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# PyTorch en versión CPU (más liviana que la versión con CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

### Versiones probadas

Los modelos de `models/` se guardaron con estas versiones. Cargar `clasico_mejor.joblib` con versiones muy distintas de scikit-learn o de XGBoost puede fallar o emitir advertencias.

| Paquete | Versión |
|---|---|
| Python | 3.11 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| opencv-python | 5.0.0 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.9.1 |
| xgboost | 3.2.0 |
| torch | 2.15.0 (CPU) |
| streamlit | 1.66.1 |

## 3. Uso sin reentrenar

Los dos modelos finales están versionados en `models/`:

| Archivo | Modelo |
|---|---|
| `clasico_mejor.joblib` | Pipeline `StandardScaler` + XGBoost, con la lista de características |
| `cnn_lr1e-3_do0.5.pt` | Pesos de la CNN y su configuración |

Con eso basta para ejecutar las pruebas y las aplicaciones:

```bash
pytest                                  # pruebas unitarias
streamlit run app/etiquetar.py          # herramienta de etiquetado de recortes
streamlit run app/app.py                # aplicación de calidad y tamaño con cámara
```

## 4. Reproducción completa

### 4.1 Credencial de Kaggle

1. En Kaggle, ir a *Settings → API → Create New Token*.
2. Guardar solo el texto del token (`KGAT_...`) en `~/.kaggle/access_token`. En Windows la ruta es `C:\Users\<usuario>\.kaggle\access_token`, sin extensión.
3. El token **nunca** se sube al repositorio.

### 4.2 Flujo de datos y modelos

Los pasos se ejecutan en orden. Los tiempos son aproximados, medidos en un portátil de 4 núcleos.

| Paso | Comando | Produce | Tiempo |
|---|---|---|---|
| 1. Descarga | `python scripts/download_data.py` | `data/raw/kaggle/` | Depende de la conexión |
| 2. Limpieza y partición | `python scripts/prepare_dataset.py` | `data/processed/manifest.csv` | Varios minutos |
| 3. Recortes de `mixed_quality` | `python scripts/segment_mixed.py --input data/raw/kaggle` | `data/interim/segmentadas/` | Varios minutos |
| 4. Características | `python scripts/extract_features.py` | `data/processed/features.parquet` | ~15 min |
| 5. Recortes 128×128 para la CNN | `python scripts/cache_tiles.py` | `data/processed/tiles128.npy` | ~14 min |
| 6. EDA | `python scripts/build_notebooks.py eda --run` | `notebooks/02_eda.ipynb`, figuras `eda_*` | ~5 min |
| 7. Modelos clásicos | `python scripts/build_notebooks.py clasicos --run` | `models/clasico_mejor.joblib`, figuras `modelos_*` | ~15 min |
| 8. CNN | `python scripts/train_cnn.py --tag lr1e-3_do0.5 --lr 1e-3 --dropout 0.5 --epochs 20` | `models/cnn_lr1e-3_do0.5.pt`, `reports/cnn/` | ~35 min |
| 9. Comparación CNN vs. XGBoost | `python scripts/build_notebooks.py cnn --run` | `notebooks/05_cnn_y_comparacion.ipynb`, figuras `cnn_*` | ~3 min |
| 10. Tamaño (requiere las fotos propias) | `python scripts/build_notebooks.py tamano --run` | `notebooks/06_tamano.ipynb`, sección `tamano` de `models/umbrales.json`, figuras `tamano_*` | ~1 min |
| 11. Tres clases (requiere las fotos propias) | `python scripts/build_notebooks.py calidad --run` | `notebooks/07_tres_clases.ipynb`, sección `calidad` de `models/umbrales.json`, figuras `calidad_*` | ~1 min |
| 12. Diagramas del informe | `python scripts/build_diagrams.py` | figuras `diagrama_crispdm` y `diagrama_arquitectura` | segundos |

- Todos los procesos aleatorios usan la semilla `SEED = 42` (`src/fruitq/config.py`).
- En Windows, si la consola muestra errores de codificación, se define antes `set PYTHONIOENCODING=utf-8`.

## 5. Fotos propias

El protocolo de toma de fotografías y el formato de las anotaciones están en [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md) y en [`../data/README.md`](../data/README.md). Las imágenes se guardan en `data/raw/propias/`, que no se versiona.
