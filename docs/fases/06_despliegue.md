# Fase 6 — Despliegue

Aplicación Streamlit en `app/app.py`. Especificación: `docs/TRASPASO.md` §4.5.

```bash
streamlit run app/app.py
```

## 1. Funciones

- **Entrada:** foto con la cámara del navegador (`st.camera_input`) o archivo de imagen (`st.file_uploader`: JPG, PNG, BMP o WEBP).
- **Tipo de fruta:** se elige en la barra lateral, porque el modelo de calidad no reconoce el tipo de fruta. Solo se usa para asignar la clase de tamaño.
- **Salida:**
  - imagen con el contorno de la fruta detectada;
  - clase de calidad (`buena`, `regular` o `mala`) y probabilidad calibrada de que la fruta esté mala;
  - longitud estimada en centímetros y clase de tamaño (pequeño, mediano o grande);
  - tiempo de procesamiento;
  - detalle: probabilidad sin calibrar, umbrales y longitud en píxeles.
- **Advertencia permanente:** el modelo de calidad no generaliza a fotos nuevas (F1-macro 0.62 en los recortes de Kaggle frente a 0.25 en las fotos propias) y el tamaño solo es válido con el protocolo de las fotos propias.

## 2. Procesamiento

El análisis está en `src/fruitq/predict.py` (`analizar`, `analizar_bytes`), con los mismos pasos y parámetros que los notebooks de evaluación:

| Paso | Función | Detalle |
|---|---|---|
| 1. Decodificación | `fruitq.io.decode_image` | Bytes JPG o PNG a imagen BGR |
| 2. Escala de trabajo | `fruitq.io.resize_max` | Lado mayor reducido a 1024 px, como en `notebooks/06_tamano.ipynb` y `notebooks/07_tres_clases.ipynb` |
| 3. Segmentación | `fruit_on_uniform_background`; si no detecta fruta, `largest_fruit` | Fruta principal y su contorno |
| 4. Preparación para el modelo | `fruitq.quality.kaggle_view` | Cuadrado de lado $d / 0.436$ centrado en la fruta, reducido a 256 px (escala de entrenamiento) |
| 5. Calidad | `extract_features` y XGBoost (`models/clasico_mejor.joblib`) | $p = P(\text{mala} \mid x)$; clase con $t_1, t_2$ |
| 6. Probabilidad mostrada | Calibración de Platt | $P = \sigma(a \cdot \operatorname{logit} p + b)$; no cambia la clase porque conserva el orden |
| 7. Tamaño | `characteristic_length` y `size_class` | $L_{px}$ a la escala de 1024 px; $L_{cm} = k \cdot L_{px}$; clase por terciles de la fruta elegida |

Si la imagen llega con menos de 1024 px por el lado mayor (por ejemplo, desde la cámara del navegador), no se amplía: la longitud medida se lleva a la escala de 1024 px multiplicándola por $1024 / \text{lado mayor}$.

**Parámetros.** Se leen de `models/umbrales.json` y no se escriben en el código:

| Sección | Parámetros | Valor actual |
|---|---|---|
| `calidad` | $t_1$, $t_2$ | 0.178, 0.986 |
| `calidad` | Platt $a$, $b$ | 0.363, −1.828 |
| `tamano` | $k$ (cm/px), lado de trabajo | 0.0404, 1024 px |
| `tamano` | terciles en cm | banano 13.8 / 14.5; guayaba 6.97 / 7.10; manzana 7.5 / 8.0 |

Si el notebook de tres clases cambia el puntaje elegido a uno distinto de XGBoost, `cargar_parametros` lo rechaza con un mensaje explícito, en lugar de aplicar los umbrales a un puntaje que no corresponde.

## 3. Verificación

- **Pruebas unitarias** (`tests/test_predict.py`, 6 pruebas): clases válidas en una foto sintética, tamaño independiente de la resolución de entrada, fruta sin terciles, imagen sin fruta, decodificación de bytes y monotonía de la calibración. La suite completa tiene 27 pruebas.
- **Reproducción de los notebooks** (`scripts/medir_app.py`): en las 45 fotos propias, la aplicación obtiene la misma probabilidad que `reports/tabla_calidad_fotos_propias.csv` ($|\Delta p| \le 5 \cdot 10^{-5}$, el redondeo de la tabla), la misma longitud que `reports/tabla_tamano_fotos_propias.csv` ($|\Delta L_{px}| \le 5 \cdot 10^{-4}$) y la misma clase de calidad en las 45 fotos. Las 45 se segmentan con `fruit_on_uniform_background`.

## 4. Latencia medida

Medida con `python scripts/medir_app.py` sobre las 45 fotos propias (entre 1.2 y 12.2 MP; mediana 12 MP), con 3 repeticiones por foto y la mediana de cada foto. Equipo: Linux x86_64, 2 núcleos (Intel Xeon a 2.1 GHz), Python 3.13. El tiempo va desde la decodificación de la imagen hasta el resultado; no incluye la carga del modelo, que ocurre una sola vez al iniciar la aplicación (`st.cache_resource`). Tabla por foto: `reports/tabla_latencia_app.csv`.

| Entrada | Mediana | Percentil 90 | Máximo |
|---|---|---|---|
| Fotos propias originales (≈ 12 MP) | 129 ms | 160 ms | 182 ms |
| Las mismas fotos reducidas a 640 px (tamaño típico de la cámara del navegador) | 16 ms | 21 ms | 23 ms |

Desglose por etapa en las fotos originales (mediana por foto):

| Etapa | Tiempo |
|---|---|
| Decodificación del JPEG | 72 ms |
| Reducción a 1024 px | 21 ms |
| Segmentación (`fruit_on_uniform_background`) | 21 ms |
| Recorte a la escala de entrenamiento y 57 características | 7 ms |
| XGBoost | 1 ms |
| Tamaño | < 1 ms |

- Leer y reducir una foto de 12 MP ocupa más del 70 % del tiempo; el modelo, alrededor del 1 %.
- La segmentación de las fotos propias es mucho más rápida que la de Kaggle (unos 270 ms, `docs/arquitectura.md` §6), porque decide por croma respecto al fondo y no usa GrabCut.
- Con cualquiera de las dos entradas, el tiempo es muy inferior al de tomar la foto, así que la aplicación responde de forma interactiva.
- La cifra depende del equipo. Para medirla en otro computador basta ejecutar `python scripts/medir_app.py` con las fotos propias en `data/raw/propias/`.

## 5. Limitaciones

- **Calidad:** el modelo usa la nitidez y el brillo de las fotos de Kaggle como atajo y no generaliza a las fotos propias (`docs/fases/05_evaluacion.md`). Por eso la aplicación muestra la advertencia en todo momento.
- **Tamaño:** la escala $k$ se calibró con un celular a 30 cm sobre cartulina blanca. Con otra cámara, otra distancia u otra lente, la medida en centímetros no es válida. La cámara del computador no cumple ese protocolo.
- **Tipo de fruta:** lo indica el usuario. Las clases de tamaño solo existen para banano, guayaba y manzana; para limón, naranja y granada se muestra solo la medida en centímetros.
- **Segmentación:** supone una sola fruta sobre fondo uniforme. Con fondos complejos se usa `largest_fruit`, que puede tomar otra región; la imagen con el contorno permite comprobarlo.
- **Aviso al iniciar:** XGBoost puede mostrar en la terminal un aviso sobre el formato del modelo serializado. No afecta las predicciones: la comparación con los notebooks da la misma probabilidad en las 45 fotos.

## 6. Pendiente

- [x] Aplicación con cámara y carga de imagen
- [x] Medición de latencia
- [ ] Diagrama de arquitectura del sistema para el informe (integrante 3; el flujo está en `docs/arquitectura.md` §1)
- [ ] (Opcional) Simulación de línea de empaque con Raspberry Pi
