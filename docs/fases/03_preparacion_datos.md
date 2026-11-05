# Fase 3 — Preparación de los datos

## Pasos
0. **Limpieza y agrupación** (`scripts/prepare_dataset.py` → `data/processed/manifest.csv`): se eliminan las 1 188 imágenes en gris y se agrupan los casi duplicados (hash exacto y ráfagas de fotos). Ver los hallazgos 3 y 4 de la fase 2 y la sección de decisiones.
1. **Segmentación** de la fruta frente al fondo (`fruitq.segmentation`), en tres etapas:
   - (a) se modela el color del fondo con k-medias (k = 3) en CIELab sobre la franja del borde, se calcula D(x) = min_j ‖Lab(x) − c_j‖ y se umbraliza D con Otsu;
   - (b) se refina con GrabCut, limitado a una banda alrededor de la máscara;
   - (c) se separan las frutas que se tocan con watershed sobre la transformada de distancia.

   Primero se probó con umbral de Otsu sobre la saturación HSV: falló con pasto y cartulinas saturadas, que no son fondo "uniforme".
2. **Recorte de ejemplares individuales** de `mixed_quality` (`scripts/segment_mixed.py`) mediante componentes conexas.
3. **Estimación de tamaño** (`fruitq.size`).
4. **Extracción de características** para modelos clásicos (`fruitq.features`): histogramas HSV, momentos de color, forma, nitidez, proporción de manchas oscuras.
5. **Normalización**: `StandardScaler` ajustado **solo con el conjunto de entrenamiento**, $z = (x-\mu)/\sigma$.
6. **Partición estratificada por grupos** train/val/test (70/15/15) con semilla fija; los casi duplicados quedan siempre en el mismo subconjunto. Las fotos propias se reservan en parte para test, para medir la generalización a datos reales.
7. **Desbalanceo**: pesos de clase $w_k = N/(K\cdot N_k)$ y/o aumento de datos (rotación, volteo, brillo) solo en train.

## Resultado de la limpieza (Kaggle, buena/mala)

| Paso | Imágenes |
|---|---|
| Imágenes buena/mala en Kaggle | 18 452 |
| Tras eliminar imágenes en gris | 17 264 |
| Tras eliminar duplicados exactos (MD5) | 17 264 |
| Grupos con etiqueta contradictoria eliminados | 0 |
| **Imágenes utilizables** | **17 264** en **6 141 grupos** (33 % de los grupos tienen más de una imagen; el mayor tiene 245) |

Partición agrupada y estratificada por `fruta_calidad` (semilla 42): **train 12 269 · val 2 455 · test 2 540**, unos 14 % de test por estrato. La excepción es `naranja_mala`, con 21 % en test, porque una sesión de 245 fotos quedó completa en ese subconjunto. `check_no_leakage` verifica que ningún grupo aparezca en dos subconjuntos.

## Decisiones y justificación

### ¿Cómo definir "casi duplicado"? (evidencia de PI3)
Se compararon tres criterios sobre las 17 264 imágenes:

| Criterio | Grupos | Grupo más grande | Problema |
|---|---|---|---|
| Hamming(aHash) ≤ 4 con union-find | 5 498 | **8 261** | La transitividad A~B~C~… encadena fotos distintas |
| Hamming(aHash) ≤ 1 | 9 740 | 627 | Sigue encadenando |
| Número de secuencia consecutivo (IMG_8479, IMG_8480, …) | — | 521 | Une sesiones enteras (todo el banano bueno de una fecha) |
| aHash exacto entre carpetas | — | 815 | Colisiones entre frutas distintas: hashes "genéricos" |
| **Elegido:** aHash exacto dentro de la misma carpeta (si ≤ 3 imágenes lo comparten) **o** ráfaga ≤ 1 s según la hora del nombre del archivo | 6 141 | 245 | Grupos razonables; la partición queda balanceada |

Lección: la relación "casi duplicado" no es transitiva, pero las componentes conexas sí lo son. Por eso un umbral que parece estricto (d_H ≤ 4 de 64 bits) produce un grupo con casi la mitad del dataset. Hubo que restringir las aristas del grafo, no solo el umbral.

### Desbalanceo restante
`granada_buena` sigue teniendo 4 752 imágenes (≈ 5,5× las demás) incluso sin las imágenes en gris. Para el modelado:
- **Entrenamiento:** submuestreo de `granada_buena` en train a ~850 imágenes (tomando grupos completos) y/o pesos de clase $w_k = N/(K N_k)$ en la función de pérdida.
- **Evaluación:** F1-macro, que da el mismo peso a cada clase, y matriz de confusión por fruta.
- **Test:** no se toca; refleja la distribución original.
