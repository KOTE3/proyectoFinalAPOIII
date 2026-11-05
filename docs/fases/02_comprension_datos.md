# Fase 2 — Comprensión de los datos

Notebook: `notebooks/02_eda.ipynb`

## Checklist del análisis exploratorio
- [x] Inventario Kaggle y segmentadas
- [ ] Inventario de fotos propias (cuando se recolecten)
- [x] Distribución de clases y **desbalanceo** (gráfica de barras, razón mayoritaria/minoritaria)
- [x] Resolución, formato y relación de aspecto de las imágenes
- [x] Calidad: imágenes borrosas (varianza del Laplaciano), sobre/subexpuestas, duplicadas
- [x] Variabilidad: fondos, iluminación, número de frutas por foto
- [x] Ejemplos visuales por clase (mosaico)
- [x] Distribución del diámetro normalizado por fruta (para definir los umbrales de tamaño)
- [ ] Diferencias entre datos de Kaggle y propios (posible *domain shift*)

## Hallazgos (dataset Kaggle, 2026-11-05)

| # | Hallazgo | Evidencia | Decisión para la fase 3 |
|---|---|---|---|
| 1 | 19 526 imágenes JPG, 6 frutas (manzana, banano, guayaba, lima, naranja, granada), 2 calidades (buena 11 664 / mala 6 788) + 1 074 imágenes `mixed` sin etiqueta por fruta | `eda_distribucion_clases.pdf` | Usar buena/mala como base y construir la 3.ª clase (`regular`) con anotación manual |
| 2 | **Desbalanceo**: `Pomegranate_Good` = 5 940 imágenes, ≈ 5,5× las demás clases (~1 100) | Razón máx/mín = 5,47 | Eliminar sus imágenes en gris y deduplicar; si persiste, aplicar pesos de clase o submuestreo |
| 3 | **1 188 imágenes en escala de grises**, todas en `Pomegranate_Good` (aumentos ya incluidos en el dataset) | Sección 2 del notebook | **Eliminarlas**: el color es la señal principal de calidad |
| 4 | **5 874 casi duplicados** (≈ 30 %, mismo average-hash 8×8) en 2 759 grupos: ráfagas de fotos tomadas con un segundo de diferencia y copias en gris; 19 grupos tienen etiquetas distintas | Verificación visual de grupos aleatorios | Partición **por grupo de hash** (`GroupShuffleSplit`) para evitar fuga de información entre train y test |
| 5 | Resolución mayoritaria de 256×256 o 256×192; 564 imágenes de 12–48 MP | `value_counts` de resolución | Redimensionar todas a un tamaño fijo (128 px para la CNN) |
| 6 | Fondos **muy heterogéneos** (pasto, baldosa, cartulinas de colores, manos) y **solo ~36 % con una única fruta**, también en buena y mala | `eda_mosaico.pdf`, `eda_frutas_por_imagen.pdf` | Segmentación robusta (modelo de fondo por borde + GrabCut + watershed) y enmascarar el fondo para que el modelo no aprenda atajos |
| 7 | Distancia de cámara no controlada: el diámetro normalizado de Kaggle no refleja el tamaño físico | `eda_tamano.pdf` | Calibrar los umbrales de tamaño con las **fotos propias** a distancia fija |
| 8 | 0,9 % de imágenes borrosas (varianza del Laplaciano < 50) | `eda_brillo_nitidez.pdf` | Mantenerlas (aportan robustez) y revisar si concentran errores |
| 9 | En el PCA global, buena y mala se solapan porque mezcla 6 frutas con colores distintos; las diferencias aparecen **dentro de cada fruta** (fracción oscura, desviación de V) | `eda_pca.pdf`, `eda_separabilidad.pdf` | Incluir la fruta como contexto, o evaluar por fruta; justifica probar modelos no lineales (RF, SVM-RBF, CNN) |

### Segmentación de `mixed_quality`
`scripts/segment_mixed.py` extrajo **4 716 recortes** de 1 074 imágenes (`data/interim/segmentadas/segmentacion.csv`). Funciona bien con frutas separadas sobre pasto o cartulina. Falla con **sombras sobre fondos de color**, **racimos de banano** (se parten o se unen) y **franjas oscuras pegadas al borde**. Por eso cada recorte se revisa a mano: columna `valido` (1/0) y su `calidad`.
