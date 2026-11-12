# Fase 4 — Modelado

## Modelos
| Tipo | Modelo | Entrada | Hiperparámetros buscados |
|---|---|---|---|
| Piso | Clase mayoritaria | — | — |
| Línea base | Regresión logística | 57 características | `C` ∈ {0.01, 0.1, 1, 10} |
| ML clásico | SVM (RBF) | 57 características | `C` ∈ {0.3, 1, 3, 10, 30}, `γ` ∈ {scale, 0.003, 0.01, 0.03} |
| ML clásico | Random Forest (300 árboles) | 57 características | `max_depth` ∈ {∞, 20, 10}, `min_samples_leaf` ∈ {1, 3, 10}, `max_features` ∈ {√d, 0.3} |
| ML clásico | XGBoost (400 árboles) | 57 características | `max_depth` ∈ {3, 5, 7}, `learning_rate` ∈ {0.05, 0.1} |
| Deep learning | CNN pequeña desde cero (72 434 parámetros) | Recorte enmascarado 128×128 | `lr` ∈ {1e-3, 3e-4}, `dropout` ∈ {0.3, 0.5} |

- **Tarea actual:** binaria buena (0) / mala (1). La clase `regular` se obtiene después con umbrales sobre P(mala | x), ver la sección "De 2 a 3 clases".
- **Búsqueda:** `GridSearchCV` + `StratifiedGroupKFold(k=5)` sobre train, con métrica `f1_macro`. Los grupos de casi duplicados nunca se separan entre pliegues.
- **Desbalanceo:** pesos de muestra $w_i = N / (K N_{e(i)})$ por estrato fruta×calidad.
- Notebook: [`notebooks/04_modelos_clasicos.ipynb`](../../notebooks/04_modelos_clasicos.ipynb). Características: `scripts/extract_features.py` (17 264 imágenes, 0 errores, 0,2 % sin fruta detectada).

## Registro de experimentos (2026-11-05)

| Modelo | Mejores hiperparámetros | F1 train (CV) | F1 val (CV) ± std | Tiempo de búsqueda |
|---|---|---|---|---|
| Regresión logística | C = 10 | 0.913 | 0.902 ± 0.019 | 0.2 min |
| SVM (RBF) | C = 10, γ = scale | 0.995 | 0.955 ± 0.004 | 2.2 min |
| Random Forest | max_depth = ∞, min_samples_leaf = 1, max_features = √d | 0.999 | 0.943 ± 0.014 | 8.3 min |
| **XGBoost** | max_depth = 7, learning_rate = 0.1 | 1.000 | **0.957 ± 0.010** | 0.7 min |

**Observaciones**
- La regresión logística casi no sobreajusta (0.913 vs 0.902), pero su frontera lineal limita su desempeño: es un modelo **sesgado**. Los modelos no lineales memorizan train (F1 ≈ 1) y aun así generalizan mejor (0.94–0.96): tienen más **varianza**, pero mucho menos sesgo.
- Dos óptimos quedaron en el **borde de la rejilla** (logística C = 10, XGBoost max_depth = 7). Conviene ampliar la rejilla para confirmar que no hay un óptimo más allá.

## Ablación: ¿qué información usa el modelo? (XGBoost, F1-macro en val)

| Características | F1 val |
|---|---|
| Todas (57) | 0.957 |
| Sin nitidez/textura (`laplacian_var`, `gradient_energy`) | 0.943 |
| Solo color (histogramas y momentos HSV/Lab) | 0.942 |
| Solo nitidez/textura (2) | 0.715 |

La importancia por permutación pone primero a `gradient_energy` y `laplacian_var`, pero al quitarlas el F1 cae solo 1,4 puntos: **el color basta para la mayor parte de la decisión**. Además, la nitidez cambia de dirección según la fruta: en granada y guayaba las buenas son más nítidas, y en banano y limón las malas lo son. Eso sugiere que refleja **la cámara o la sesión de captura** de cada carpeta más que el defecto físico (un posible *atajo*). Las fotos propias, tomadas con otra cámara, dirán si el modelo depende de él.

## CNN desde cero (`scripts/train_cnn.py`)

**Arquitectura** (`src/fruitq/cnn.py`): 3 bloques `[Conv 3×3 → BatchNorm → ReLU] × 2 → MaxPool 2×2` con 16, 32 y 64 filtros, Global Average Pooling, Dropout y una capa lineal de 2 salidas. Tiene **72 434 parámetros** y en CPU predice en unos 5 ms por imagen dentro de un lote.

**Entrada:** se usa solo la fruta principal, con el fondo en negro, centrada en un cuadrado de 128×128 (`scripts/cache_tiles.py`). Enmascarar el fondo evita que la red aprenda atajos del entorno.

**Entrenamiento**
- AdamW con `weight_decay = 1e-4` y decaimiento coseno de la tasa de aprendizaje, durante 20 épocas.
- Cada época tiene 6 000 muestras de train **balanceadas por estrato**, con probabilidad ∝ 1/N_estrato.
- Aumento de datos: las 8 simetrías del cuadrado, más brillo y contraste de ±20 %. **No** se cambia el tono ni la saturación, porque el color es la señal de calidad.
- Se guarda el modelo con el mejor F1-macro en val.

| Configuración | Mejor época | F1 val | F1 test | AUC test | Tiempo |
|---|---|---|---|---|---|
| lr = 1e-3, dropout = 0.3 | 18 | 0.931 | 0.946 | 0.985 | 35 min |
| **lr = 1e-3, dropout = 0.5** | 17 | **0.935** | 0.940 | 0.983 | 34 min |
| lr = 3e-4, dropout = 0.3 | 17 | 0.924 | 0.938 | 0.982 | 35 min |

La configuración se eligió por el F1 de **validación** (dropout 0.5). El F1 de test se muestra solo como referencia y no se usó para decidir.

**Curvas** (`reports/figures/cnn_curvas.pdf`)
- La pérdida de train y la de val bajan juntas y casi no se separan (0.19 frente a 0.16): **no hay sobreajuste**. La val queda incluso un poco por debajo de train porque en train la red ve imágenes aumentadas y con dropout activo.
- En la época 4 la configuración lr = 1e-3 / dropout 0.3 tiene un pico (F1 0.56). Es una inestabilidad típica de una tasa de aprendizaje alta en las primeras épocas, y se recupera en la siguiente.
- El F1 seguía subiendo en la época 20, así que con más épocas (o una red algo más ancha) podría mejorar. No se hizo por el costo en CPU (~35 min por corrida).
