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

## Pendiente
- [x] CNN y comparación con los modelos clásicos
- [x] Curvas de entrenamiento y validación de la CNN (¿sobreajuste?)
- [ ] Desempeño en las fotos propias (generalización y prueba del atajo de nitidez)
- [ ] Comparación con resultados de la literatura
