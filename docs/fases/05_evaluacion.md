# Fase 5 — Evaluación

## Métricas
- **F1-macro** (métrica principal; robusta al desbalanceo).
- **Recall de `mala`** (costo de negocio: no dejar pasar fruta dañada).
- Exactitud, precisión y recall por clase, **matriz de confusión**, AUC-ROC.
- Tamaño: exactitud frente a la etiqueta manual.
- Tiempo de inferencia por imagen (requisito de tiempo real).

## Comparación en validación (ML clásico)

| Modelo | Accuracy | F1-macro | Recall (mala) | AUC | ms/imagen* |
|---|---|---|---|---|---|
| Clase mayoritaria | 0.610 | 0.379 | 0.000 | 0.500 | — |
| Regresión logística (línea base) | 0.917 | 0.913 | 0.920 | 0.964 | 0.002 |
| SVM (RBF) | 0.952 | 0.949 | 0.911 | 0.990 | 0.138 |
| Random Forest | 0.944 | 0.941 | 0.910 | 0.982 | 0.031 |
| **XGBoost** | **0.958** | **0.956** | **0.939** | 0.987 | 0.007 |

\* Solo la clasificación; la segmentación y la extracción de características toman ~270 ms por imagen y dominan el tiempo total.

## Modelo elegido: XGBoost, evaluado en test una sola vez (2 540 imágenes)

| | Precisión | Recall | F1 |
|---|---|---|---|
| buena | 0.966 | 0.976 | 0.971 |
| mala | 0.965 | 0.950 | 0.958 |
| **macro** | 0.965 | 0.963 | **0.964** |

Accuracy 0.965 · AUC 0.990. El F1 de test (0.964) coincide con el de validación (0.956) y con el de CV (0.957): **no hay señales de sobreajuste a la selección del modelo**.

**Exactitud por fruta (test)**

| Fruta | buena | mala |
|---|---|---|
| banano | 1.000 | 0.923 |
| granada | 0.996 | 0.959 |
| guayaba | 0.988 | 0.975 |
| limón | 1.000 | 0.981 |
| manzana | **0.896** | 0.951 |
| naranja | **0.919** | 0.927 |

## Análisis de errores (`reports/figures/modelos_errores.pdf`)
1. **Ruido de etiqueta a nivel de imagen:** la etiqueta es de la foto completa, pero las características vienen solo de la fruta más grande. En fotos de la carpeta "mala" con varias frutas, la más grande puede estar sana.
2. **Segmentación sobre fondos complejos** (banano sobre maleza): la máscara incluye hojas y altera el color medido.
3. **Manzana y naranja buenas** son las más confundidas: su piel rojiza o con brillos se parece a la de una fruta deteriorada.

## CNN frente a XGBoost (test, 2 540 imágenes) — `notebooks/05_cnn_y_comparacion.ipynb`

| Modelo | Accuracy | F1-macro | Recall (mala) | AUC |
|---|---|---|---|---|
| CNN (lr 1e-3, dropout 0.5) | 0.941 | 0.940 | 0.939 | 0.983 |
| **XGBoost** | **0.965** | **0.964** | 0.950 | 0.990 |
| Ensamble (promedio de probabilidades) | 0.964 | 0.963 | **0.955** | **0.991** |

**Prueba de McNemar** (¿la diferencia es real o es azar?). Solo cuenta las imágenes en las que los dos modelos no coinciden:
- solo acierta la CNN: $b = 31$;
- solo acierta XGBoost: $c = 92$;
- prueba binomial exacta: $p < 10^{-4}$.

**XGBoost es significativamente mejor.** Hay 57 imágenes que fallan los dos modelos: son los errores "duros", y muchos se deben a ruido de etiqueta o a una mala segmentación.

**Exactitud por fruta y calidad (test)**

| Fruta | buena CNN | buena XGB | mala CNN | mala XGB |
|---|---|---|---|---|
| banano | 0.969 | 1.000 | 0.852 | 0.923 |
| granada | 0.984 | 0.996 | **0.970** | 0.959 |
| guayaba | 0.963 | 0.988 | 0.938 | 0.975 |
| limón | 0.994 | 1.000 | 0.974 | 0.981 |
| manzana | 0.835 | 0.896 | 0.920 | 0.951 |
| naranja | 0.798 | 0.919 | **0.963** | 0.927 |

**¿Por qué gana el modelo clásico?**
1. Las 57 características ya resumen justo lo que importa (color y textura), y el modelo clásico las aprovecha con los 12 269 ejemplos de train.
2. Una CNN de 72 mil parámetros entrenada desde cero, en CPU y durante 20 épocas, tiene que aprender esos detectores sola, y las curvas muestran que todavía no había convergido.
3. Las características incluyen la nitidez de la imagen original, que puede ser un atajo de la cámara (ver la ablación en la fase 4). La CNN ve un recorte reducido a 128 px, donde esa señal casi desaparece.

**Calibración** (`cnn_calibracion.pdf`)
- Las probabilidades de la CNN son **más suaves**: el 10,9 % de las imágenes de test cae entre 0.2 y 0.8, frente al 4,4 % en XGBoost.
- La CNN es algo **sobreestimada** en la zona media: cuando dice 0.36, la fracción real de malas es 0.27.
- Para la clase `regular`, la CNN (o el ensamble) da más margen para los umbrales $t_1, t_2$.

**Grad-CAM** (`cnn_gradcam.pdf`)
- En las frutas malas acertadas, la red se activa sobre **manchas, magulladuras y bordes podridos**, es decir, mira el defecto y no el fondo.
- En los errores se ven dos causas:
  - una segmentación que incluye la mano o el tallo (la manzana "buena" con p = 0.90);
  - un brillo o una sombra en la piel de una naranja buena, que la red confunde con una mancha.

## Hallazgo para la 3.ª clase
XGBoost está **sobreconfiado**: P(mala | x) cae casi siempre cerca de 0 o de 1 (`modelos_prob_mala.pdf`), así que la zona intermedia queda casi vacía. Para que los umbrales $t_1, t_2$ den una clase `regular` útil hay dos opciones:
- **calibrar** las probabilidades (Platt o isotónica) con los recortes etiquetados;
- usar un modelo con probabilidades más suaves (SVM calibrada o la CNN).

Se decide cuando existan etiquetas de `regular`.

## Tamaño en las fotos propias (45 fotos) — `notebooks/06_tamano.ipynb`

Método en `docs/fases/04_modelado.md`. Todas las cifras son de **validación cruzada dejando una fruta fuera**: cada foto se estima con un $\hat k$ ajustado sin su fruta. Tabla por foto: `reports/tabla_tamano_fotos_propias.csv`.

### Longitud en centímetros

| Fruta | EAM | Sesgo medio | Error relativo medio | Error máximo |
|---|---|---|---|---|
| Banano | 1.49 cm | +0.76 cm | 10.4 % | 6.65 cm |
| Guayaba | 0.62 cm | −0.45 cm | 8.5 % | 1.47 cm |
| Manzana | 0.82 cm | +0.73 cm | 10.8 % | 2.80 cm |
| **Total** | **0.98 cm** | +0.35 cm | **9.9 %** | 6.65 cm |

- El 71 % de las fotos queda a 1 cm o menos de la medida con regla (`tamano_estimado_vs_real.pdf`).
- Las fotos se ordenan alrededor de la recta $\text{medida} = \hat k \cdot L_{px}$, salvo las de la sesión `banano buena` (`tamano_escala.pdf`).

### Fuente del error: sesiones de captura

Cada combinación fruta × calidad se fotografió en una sesión distinta:

| Sesión | Celular | Fecha | $L_{px}$ mediana | Sesgo medio |
|---|---|---|---|---|
| **banano buena** | moto g(9) play | 2026-11-18 | **419 px** | **+3.37 cm** |
| banano mala | moto g(9) play | 2026-11-18 | 332 px | −0.50 cm |
| banano regular | iPhone 13 | 2026-11-16 | 338 px | −0.59 cm |
| **manzana mala** | moto g(9) play | 2026-11-23 | 216 px | **+1.46 cm** |
| Otras cinco sesiones | — | — | — | entre −0.62 y +0.38 cm |

- Los bananos `buena` miden apenas 0.7 cm más que los `mala` con regla, pero ocupan un 26 % más de píxeles: **la cámara quedó más cerca de la fruta** en esa sesión. Esas cinco fotos concentran el error: sin ellas, el EAM del banano baja a 0.55 cm y el total a 0.68 cm.
- `banano buena` y `banano mala` se tomaron el mismo día con el mismo celular, así que **un $k$ por día o por celular no corrige el problema**; solo una distancia de cámara realmente fija lo evita.
- La mediana limita el efecto de esa sesión sobre $\hat k$, pero no puede corregir las fotos de la propia sesión.
- Las cinco fotos `manzana_regular` comprimidas (960×1280 px, sin metadatos) tienen un sesgo de +0.38 cm, comparable al de las demás: la compresión no afecta la medición del contorno.

### Clase de tamaño

| Fruta | $t_1$ | $t_2$ | Rango medido | Exactitud | Clase más frecuente |
|---|---|---|---|---|---|
| Banano | 13.83 cm | 14.50 cm | 1.5 cm | 60.0 % | 40.0 % |
| Guayaba | 6.97 cm | 7.10 cm | 2.3 cm | 60.0 % | 33.3 % |
| Manzana | 7.50 cm | 8.00 cm | 1.5 cm | 46.7 % | 46.7 % |
| **Total** | | | | **55.6 %** | 40.0 % |

- La exactitud supera a la de asignar siempre la clase más frecuente (40 %), y el 15.6 % de las fotos se confunde entre clases extremas (pequeño ↔ grande) (`tamano_clases.pdf`).
- **Limitación principal:** las cinco frutas de cada tipo tienen tamaños parecidos, así que los terciles quedan separados por menos que el error de la estimación. En guayaba, $t_2 - t_1 = 0.13$ cm, menos que la resolución de la regla. Con este conjunto, la longitud en cm es una salida más confiable que la clase.
- Los terciles se calculan con las mismas 45 fotos que se evalúan. Esto no introduce fuga en la estimación en cm, porque los terciles solo definen la clase real; pero describen el tamaño de *estas* frutas, no una norma comercial.

### Mejoras posibles
- Fijar la cámara en un soporte para que la distancia no cambie entre sesiones.
- Definir las clases con una norma de calibre comercial (por ejemplo, las NTC del ICONTEC) en lugar de terciles de una muestra pequeña.
- Incluir un objeto de tamaño conocido en cada foto, para que la escala no dependa de la distancia.

## Pendiente
- [x] CNN y comparación con los modelos clásicos
- [x] Curvas de entrenamiento y validación de la CNN (¿sobreajuste?)
- [x] Tamaño en centímetros y clase de tamaño en las fotos propias
- [ ] Desempeño en las fotos propias (generalización y prueba del atajo de nitidez)
- [ ] Comparación con resultados de la literatura
