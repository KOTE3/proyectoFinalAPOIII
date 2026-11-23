# Traspaso del proyecto

Documento de entrega del trabajo técnico a los integrantes que continúan el proyecto. Resume:
- qué está terminado;
- qué datos se comparten fuera de GitHub;
- cómo poner en marcha el entorno;
- la especificación de cada tarea pendiente;
- las convenciones que el código y los documentos deben respetar.

## 1. Estado entregado

| Componente | Estado | Referencia |
|---|---|---|
| EDA, limpieza y partición train/val/test sin fuga (17 264 imágenes de Kaggle) | Terminado | `docs/fases/02_*.md`, `docs/fases/03_*.md`, `notebooks/02_eda.ipynb` |
| Segmentación de Kaggle y recortes de `mixed_quality` (4 716) | Terminado | `src/fruitq/segmentation.py`, `scripts/segment_mixed.py` |
| Segmentación de las fotos propias (fondo blanco; excluye la sombra), verificada en las 45 fotos | Terminado | `fruit_on_uniform_background` en `src/fruitq/segmentation.py` |
| 57 características manuales | Terminado | `src/fruitq/features.py` |
| Modelos clásicos con búsqueda de hiperparámetros | Terminado (buena/mala) | `notebooks/04_modelos_clasicos.ipynb`, `docs/fases/04_modelado.md` |
| CNN desde cero y comparación con XGBoost (McNemar, calibración, Grad-CAM) | Terminado (buena/mala) | `notebooks/05_cnn_y_comparacion.ipynb`, `docs/fases/05_evaluacion.md` |
| Modelos finales versionados | Terminado | `models/clasico_mejor.joblib`, `models/cnn_lr1e-3_do0.5.pt` |
| Herramienta de etiquetado de recortes | Terminada | `app/etiquetar.py`, `docs/ETIQUETADO.md` |
| Arquitectura e instalación | Documentadas | `docs/arquitectura.md`, `docs/instalacion.md` |
| 45 fotografías propias (5 frutas × 3 estados × 3 tipos) y sus anotaciones | Terminado | `data/labels/anotaciones_propias.csv`, `docs/PLAN_DE_TRABAJO.md` §4 |

**Resultados en test (2 540 imágenes, buena/mala):**

| Modelo | F1-macro | AUC |
|---|---|---|
| XGBoost | 0.964 | 0.990 |
| CNN | 0.940 | 0.983 |

La diferencia es significativa según la prueba de McNemar (p < 10⁻⁴).

## 2. Datos que se comparten fuera de GitHub

Las imágenes y los datos procesados no están en el repositorio (`.gitignore`). Se comparten por la carpeta del grupo en Google Drive y se copian en las rutas indicadas.

| Archivo | Tamaño | Ruta de destino | Para qué |
|---|---|---|---|
| Fotos propias (`*.jpg`) | ~3 MB c/u | `data/raw/propias/` | Clase `regular`, tamaño y evaluación en datos propios |
| `recortes_segmentados.zip` | 19 MB | Descomprimir en la raíz del repositorio (crea `data/interim/segmentadas/`) | Etiquetado de recortes |
| `manifest.csv` y `features.parquet` | 10 MB | `data/processed/` | Trabajar con los modelos clásicos sin descargar Kaggle |

El dataset de Kaggle (3,2 GB) solo hace falta para volver a leer sus imágenes: reentrenar la CNN, rehacer el EDA o regenerar el Grad-CAM. Se descarga con `python scripts/download_data.py`, usando una credencial propia (`docs/instalacion.md` §4.1). En ese caso, `tiles128.npy` (850 MB) se regenera con `python scripts/cache_tiles.py`.

Todas las rutas guardadas en CSV y parquet son relativas a la raíz del proyecto (por ejemplo, `data/raw/kaggle/...`), así que los archivos funcionan en cualquier computador.

## 3. Puesta en marcha

1. Clonar el repositorio y crear el entorno según `docs/instalacion.md` §2.
2. Copiar los archivos de la sección 2.
3. Verificar con `pytest`: deben pasar todas las pruebas.

## 4. Tareas pendientes

Se presentan en orden de dependencia.

### 4.1 Fotos propias (terminado)

- Las 45 fotos están en Drive y se copian en `data/raw/propias/`.
- Sus anotaciones están en `data/labels/anotaciones_propias.csv` y se leen con `fruitq.data.read_annotations`.
- Formato del CSV: `docs/PLAN_DE_TRABAJO.md` §4.7.
- Las tres fotos de una misma fruta comparten `id_fruta`. Cualquier partición o validación cruzada con estas fotos debe agrupar por esa columna.
- Si se agregan fotos con otro celular, se nombran con el prefijo `prueba_`.

### 4.2 Etiquetado de recortes

- Cada integrante etiqueta unos 200 recortes con `streamlit run app/etiquetar.py` (`docs/ETIQUETADO.md`).
- Además, cada uno etiqueta 50 recortes de la parte de otro integrante, para calcular el acuerdo entre etiquetadores (κ de Cohen).
- Resultado: archivos `data/labels/recortes_<nombre>.csv`.

### 4.3 Clase `regular` (tres clases)

Kaggle solo tiene `buena` y `mala`. La tercera clase se obtiene con dos umbrales sobre la probabilidad del modelo:

$$\hat y = \begin{cases} \text{buena} & p < t_1 \\ \text{regular} & t_1 \le p < t_2 \\ \text{mala} & p \ge t_2 \end{cases}, \qquad p = P(\text{mala} \mid x).$$

1. Calcular $p$ con `models/clasico_mejor.joblib` (`modelo.predict_proba`, columnas en el orden de `features`) para:
   - los recortes etiquetados;
   - las fotos propias del conjunto `principal`.
2. XGBoost está sobreconfiado (`docs/fases/05_evaluacion.md`), así que conviene calibrar $p$ con regresión logística (Platt) o isotónica antes de elegir los umbrales. Comparar también con la CNN y con el ensamble, que tienen probabilidades más suaves.
3. Elegir $t_1 < t_2$ por búsqueda en rejilla, maximizando el F1-macro de 3 clases sobre los **recortes etiquetados**.
4. Evaluar una sola vez con las **fotos propias**: matriz de confusión 3×3, F1-macro y recall de `mala`. Repetir la evaluación con las fotos de `conjunto = prueba_camara`, que miden la generalización a otro celular.
5. Para respetar la escala con la que se entrenó el modelo, preparar así las fotos propias antes de extraer las características:
   - leer con `read_image_max(path, 1024)`;
   - segmentar con `fruit_on_uniform_background`;
   - recortar un cuadrado centrado en la fruta cuyo lado sea `d / 0.44`, donde `d` es el diámetro equivalente de la máscara y 0.44 la mediana de `d_norm` en Kaggle;
   - reducir ese cuadrado a 256 px;
   - extraer las características con `extract_features`.

   Sin este paso, la fruta ocupa menos píxeles que en Kaggle y las características de nitidez cambian.
6. Leer el CSV siempre con `fruitq.data.read_annotations`.
7. Documentar los resultados en `docs/fases/04_modelado.md` y `docs/fases/05_evaluacion.md`.

### 4.4 Tamaño

1. Para cada foto propia, calcular la longitud característica a partir de la máscara (`src/fruitq/size.py`):
   - manzana y guayaba: diámetro equivalente $d = 2\sqrt{A/\pi}$;
   - banano: largo, como el lado mayor de `cv2.minAreaRect`.
2. Conversión a centímetros. No se tomaron fotos con moneda: la cámara estuvo a distancia fija y cada fruta se midió con regla. Por eso el factor de escala se estima directamente:

   $$k = \operatorname{mediana}\left(rac{	ext{medida}_{cm}}{L_{px}}ight).$$

   Para no evaluar con los mismos datos con los que se ajusta, $k$ se calcula con validación cruzada agrupada por `id_fruta`.

   Resultado preliminar, con las imágenes reducidas a 1024 px por el lado mayor:
   - $k pprox 0.042$ cm/px, con un coeficiente de variación de 15 %;
   - error absoluto medio: 0.8 cm en manzana, 1.1 cm en guayaba y 1.6 cm en banano.

   Parte de la variación viene de pequeños cambios de altura de la cámara entre sesiones. Se puede reducir estimando un $k$ por día de captura.
3. Validar contra `medida_cm` con el error absoluto medio en cm, por fruta.
4. Asignar pequeño, mediano o grande por terciles de `medida_cm` dentro de cada fruta, y medir la exactitud de la clase de tamaño estimada.

### 4.5 Aplicación con cámara (`app/app.py`)

- **Entrada:** `st.camera_input` y `st.file_uploader`.
- **Proceso:**
  - segmentar con `fruit_on_uniform_background`; si no detecta fruta, usar `largest_fruit`;
  - preparar la imagen como en §4.3, paso 5;
  - calcular las características, $p$ con XGBoost y la clase con los umbrales $t_1, t_2$;
  - estimar el tamaño con §4.4.
- **Salida:**
  - imagen con el contorno de la fruta;
  - clase de calidad con su probabilidad;
  - tamaño en clase y en cm;
  - tiempo de procesamiento.
- Guardar los umbrales y el factor $k$ (§4.4) en un archivo versionado, por ejemplo `models/umbrales.json`, en lugar de escribirlos en el código.
- Documentar en `docs/fases/06_despliegue.md`, incluida la latencia medida.

### 4.6 Entregables escritos

El reparto de informe, diagramas, ética, impacto y video está en `docs/PLAN_DE_TRABAJO.md` §2, §6 y §7. Las cifras y figuras del informe se toman de `docs/fases/` y `reports/figures/`. Las figuras están en PDF, que es formato vectorial.

## 5. Convenciones del proyecto

- **Partición:** no se reentrena ni se eligen hiperparámetros con el conjunto de test de Kaggle. La partición está en `manifest.csv` (columna `split`).
- **Semilla:** todos los procesos aleatorios usan `SEED = 42`.
- **Rutas:** se guardan con `fruitq.io.project_path` y las imágenes se leen con `fruitq.io.read_image` o `read_image_max`, que resuelven la ruta relativa a la raíz del proyecto.
- **Notebooks:** los existentes se generan desde `scripts/build_notebooks.py`. Los nuevos se numeran por fase, por ejemplo `notebooks/06_tres_clases_y_tamano.ipynb`.
- **Documentación:** español técnico y formal. Cada resultado nuevo se registra en el documento de su fase en `docs/fases/`.
- **Git:**
  - hacer `git pull` antes de trabajar;
  - agregar solo los archivos propios;
  - no versionar imágenes, datos procesados, archivos `.zip` ni configuraciones locales de editores.
