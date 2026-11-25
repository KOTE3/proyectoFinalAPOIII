# Plan de trabajo del grupo

Documento de coordinación interna: estado del proyecto, reparto de tareas, protocolo de toma de fotografías y orden de ejecución.

## 1. Estado actual

| Componente | Estado | Dónde está |
|---|---|---|
| Estructura del repositorio por fases CRISP-DM | Completo | `docs/fases/` |
| Análisis exploratorio (EDA) del dataset de Kaggle | Completo | `notebooks/02_eda.ipynb` |
| Limpieza, eliminación de duplicados y partición train/val/test por grupos | Completo | `scripts/prepare_dataset.py` |
| Segmentación de la fruta y recorte de `mixed_quality` (4 716 recortes) | Completo | `src/fruitq/segmentation.py`, `scripts/segment_mixed.py` |
| Herramienta de etiquetado | Completa | `app/etiquetar.py`, `docs/ETIQUETADO.md` |
| Modelos clásicos (regresión logística, SVM, Random Forest, XGBoost) | Completo (2 clases) | `notebooks/04_modelos_clasicos.ipynb` |
| CNN desde cero y comparación con XGBoost | Completo (2 clases) | `notebooks/05_cnn_y_comparacion.ipynb` |
| Fotografías propias (45) y sus anotaciones | Completo | Sección 4 de este documento, `data/labels/anotaciones_propias.csv` |
| **Etiquetado de recortes (clase `regular`)** | **Parcial:** Sebastián y Giuseppe terminados; falta el integrante 2 | `docs/ETIQUETADO.md`, `data/labels/recortes_*.csv` |
| Estimación de tamaño en cm y clase de tamaño | Completo | `notebooks/06_tamano.ipynb`, `models/umbrales.json` |
| **Paso de 2 a 3 clases** | **Pendiente** (`docs/TRASPASO.md` §4.3) | — |
| Evaluación en las fotos propias (generalización y atajo de nitidez) | Pendiente | `docs/fases/05_evaluacion.md` |
| Aplicación Streamlit con cámara | Pendiente | `app/` |
| Diagrama CRISP-DM propio y diagrama de arquitectura | Pendiente | `reports/figures/` |
| Tabla de ética (PI1), matriz de impacto (PI2) | Completo | `docs/fases/07_etica_impactos.md` |
| Informe IEEE (máx. 7 páginas) | Pendiente | `reports/informe/` |
| Video (máx. 10 minutos) | Pendiente | — |

Resultado actual (test, 2 540 imágenes, clases buena/mala): XGBoost obtiene F1-macro 0.964 y la CNN 0.940. La diferencia es significativa según la prueba de McNemar (p < 10⁻⁴).

## 2. Reparto de tareas

> **Traspaso.** Las tareas técnicas pendientes (clase `regular`, tamaño, evaluación en fotos propias y aplicación con cámara) se especifican en [`TRASPASO.md`](TRASPASO.md) y las continúan los integrantes 2 y 3.

### Tareas compartidas (los tres integrantes)

| Tarea | Sebastián Mejía | Integrante 2 | Integrante 3 |
|---|---|---|---|
| Fotografías (sección 4) | **Conjunto principal**: 45 fotos de las 3 frutas | **Prueba de cámara (opcional)**: 5–10 fotos con su celular | **Prueba de cámara (opcional)**: 5–10 fotos con su celular |
| Etiquetado de recortes | Parte 1 | Parte 2 | Parte 3 |
| Recortes en común (acuerdo entre etiquetadores) | 50 de la parte 2 | 50 de la parte 3 | 50 de la parte 1 |
| Filas en `data/labels/anotaciones_propias.csv` | Las 45 del conjunto principal | Las de sus fotos de prueba | Las de sus fotos de prueba |

- Meta de etiquetado: **unos 200 recortes válidos por persona** (20–30 minutos). Los recortes aparecen en orden aleatorio, así que cualquier cantidad etiquetada es una muestra representativa; más etiquetas mejoran la estimación de la clase `regular`.
- Los criterios de calidad son los mismos para fotos y recortes (sección 4.6).

### Sebastián Mejía — responsable técnico

- Toma de las 45 fotografías del conjunto principal (sección 4).
- Integrar las fotos propias al flujo de datos y verificar su segmentación.
- Pasar de 2 a 3 clases (umbrales calibrados sobre P(mala | x) con los recortes etiquetados como `regular`).
- Calibrar el tamaño en centímetros con las medidas con regla.
- Evaluar los modelos en las fotos propias (generalización a otra cámara).
- Aplicación Streamlit: carga de imagen y captura con cámara, con calidad y tamaño como salida.
- Mantener el repositorio, revisar y unir los aportes del grupo.
- Informe: **Metodología**, **Resultados** y, junto con el integrante 2, **Análisis de resultados**.
- Video: demostración de la aplicación y explicación de los modelos.

### Integrante 2 — informe e investigación

- Revisión de literatura: 8–12 artículos sobre clasificación de calidad de frutas con visión por computador (bases sugeridas: IEEE Xplore, ScienceDirect, Google Scholar; el enunciado recomienda revisar conferencias como NeurIPS, ICML e ICLR). Para cada artículo, registrar: dataset, método, métrica y resultado. Ese registro alimenta la comparación con la literatura.
- Informe: **Título**, **Resumen**, **Introducción**, **Fundamentos teóricos** y **Referencias** en formato IEEE.
- Comparación con la literatura dentro de **Análisis de resultados**.
- Montaje del documento en la plantilla IEEE (`reports/informe/conference-template-letter.docx`):
  - máximo 7 páginas;
  - figuras en PDF (vectoriales), tomadas de `reports/figures/`;
  - referenciar explícitamente el dataset de Kaggle y las librerías usadas (el enunciado considera fraude omitir fuentes de terceros).
- Verificar el informe contra la rúbrica publicada en INTU antes de la entrega.
- Coordinar con los demás grupos el mecanismo común para compartir fotos y anotaciones (sección 2.3 del enunciado).

### Integrante 3 — ética, impacto, diagramas y video

- **Tabla de ética (PI1):** completar el borrador de `docs/fases/07_etica_impactos.md`. Debe cubrir sesgos del dataset, impacto en el empleo de los clasificadores manuales, privacidad y responsabilidad ante errores.
- **Matriz de impacto (PI2):** dimensiones social, económica, ambiental y global; para cada una, el impacto positivo, el negativo y una medida de mitigación.
- **Diagrama de flujo CRISP-DM propio:** debe mostrar las decisiones concretas del proyecto (por ejemplo, la segmentación, la partición por grupos y la comparación de modelos), no una copia del diagrama genérico. Herramienta sugerida: draw.io, con exportación a PDF o SVG.
- **Diagrama de arquitectura del sistema:** foto → segmentación → características / CNN → calidad y tamaño → interfaz.
- Informe: **Conclusiones y trabajo futuro**, junto con los demás integrantes.
- **Video:** guion, grabación y edición (máximo 10 minutos; estructura en la sección 6).
- Registrar el grupo y el enlace del repositorio en el documento que comparta el docente.

## 3. Orden de ejecución

1. **Fotografías (principal y de prueba) y etiquetado de recortes.** Son el bloqueo principal: sin ellos no existe la clase `regular` ni la evaluación en datos propios.
2. En paralelo:
   - revisión de literatura;
   - tabla de ética y matriz de impacto;
   - diagramas;
   - aplicación Streamlit.
3. Modelos de 3 clases, tamaño y evaluación en las fotos propias.
4. Redacción del informe, con todas las secciones en borrador.
5. Grabación del video con la aplicación terminada.
6. Revisión final contra la rúbrica de INTU.

La fecha de entrega se confirma en INTU. Las fotografías toman unos 10 días desde la compra de la fruta, porque cada fruta se fotografía mientras pasa de `buena` a `mala` (sección 4.4). Conviene comprarla cuanto antes.

## 4. Protocolo de toma de fotografías

### 4.1 Cantidad y distribución

**Conjunto principal (Sebastián Mejía): 45 fotos de 15 frutas físicas.**

Se compran **5 frutas de cada tipo** y cada una se fotografía **tres veces**, a medida que se deteriora: una foto como `buena`, otra como `regular` y otra como `mala`.

| Fruta | Frutas físicas | buena | regular | mala | Total |
|---|---|---|---|---|---|
| Manzana | 5 | 5 | 5 | 5 | 15 |
| Banano | 5 | 5 | 5 | 5 | 15 |
| Guayaba | 5 | 5 | 5 | 5 | 15 |
| **Total** | **15** | **15** | **15** | **15** | **45** |

- Las 5 frutas de cada tipo se eligen de tamaños distintos: 2 pequeñas, 2 medianas y 1 grande, o una combinación similar.
- 45 fotos están dentro del rango de 30–50 que pide el enunciado. Si una foto sale defectuosa, se repite el mismo día.
- **Registro de la captura.** Las 45 fotos se tomaron entre el 16 y el 23 de noviembre de 2026 con dos celulares (`TRASPASO.md` §4.1). Cada foto se etiquetó según el estado visible en ella.
- Las tres fotos de una fruta comparten el mismo `id_fruta`, de modo que nunca quedan repartidas entre entrenamiento y prueba.
- **Identificación de cada fruta.** Cada fruta se guarda sobre un papel con su número (`ban_01`, `ban_02`, …). Si se marca la fruta, la marca va en la cara que no mira a la cámara. Ningún papel, cinta ni marca debe aparecer en la foto.

**¿Por qué una sola persona toma el conjunto principal?**
- Todas las fotos comparten montaje, luz y criterio de calidad. Se usaron dos celulares; la distribución de fotos por celular está en `TRASPASO.md` §4.1.
- Hay un solo lote de fruta madurando y un solo montaje fijo durante la semana.

**Fotos de prueba de cámara (integrantes 2 y 3, opcional): 5–10 fotos cada uno.**
- El conjunto principal ya proviene de dos celulares que el modelo nunca vio durante el entrenamiento. Estas fotos solo amplían la prueba de generalización.
- Con su propio celular y el mismo protocolo, mezclando las tres frutas y las tres calidades.
- No se usan para entrenar ni para elegir modelos. Solo miden si el sistema generaliza a cámaras que nunca vio, una de las preguntas del análisis de resultados.
- Se nombran con el prefijo `prueba_`; el lector del CSV las marca como `conjunto = prueba_camara`.

**Justificación de la elección.**
- **Las tres frutas están en el dataset de Kaggle.** Las fotos propias miden si el modelo generaliza a otra cámara y a otras condiciones.
- **Son económicas y se consiguen todo el año.**
- **Se deterioran en pocos días.** Eso permite obtener las tres calidades a partir del mismo lote.
- **La manzana es la fruta en la que el modelo actual más se equivoca** (89,6 % de exactitud en manzana buena), por lo que sus fotos aportan más información.
- Si alguna no se consigue, la reemplazan la naranja o el limón.

### 4.2 Materiales

- Celular con cámara. Todo el conjunto principal se toma con **el mismo celular**.
- Cartulina blanca mate, de tamaño pliego o medio pliego. Debe ocupar **todo** el encuadre.
- Soporte para fijar la altura del celular: una caja o una pila de libros.
- Regla o cinta métrica.

### 4.3 Montaje y toma

1. **Lugar.** Mesa cerca de una ventana, de día. La luz debe ser indirecta: nada de sol directo sobre la fruta ni lámparas puntuales. Si aparece una sombra fuerte, colocar una hoja blanca del lado opuesto a la ventana para rellenarla.
2. **Fondo.** La cartulina cubre todo el encuadre. **No debe verse la mesa, el borde de la cartulina, manos ni otros objetos.** El programa estima el color del fondo a partir del borde de la foto; si en el borde aparece algo distinto a la cartulina, la segmentación falla.
3. **Altura fija.** El celular va paralelo a la mesa, mirando hacia abajo (vista cenital), con el lente a **30 cm** de la cartulina; medirlo con la regla. La altura se mantiene igual en todas las fotos: el tamaño de la fruta en píxeles solo es comparable si la distancia no cambia.
4. **Configuración de la cámara:**
   - lente principal (1×): ni zoom ni gran angular (0,5×);
   - **sin flash**;
   - modo retrato, filtros y modo belleza **desactivados**;
   - formato **JPG**. En iPhone: Ajustes → Cámara → Formatos → "Más compatible". El formato HEIC no es compatible con el programa.
5. **La fruta:**
   - limpia y seca (las gotas de agua producen reflejos);
   - sin adhesivos de supermercado (el modelo podría usarlos como pista falsa);
   - centrada y completa dentro de la foto, con al menos **3 cm de cartulina libre** hacia cada borde. Una fruta que toca el borde se descarta automáticamente;
   - el banano va acostado y completo, punta a punta;
   - en `regular` y `mala`, **la cara con el defecto mira hacia la cámara**;
   - **una sola fruta por foto.**
6. **Enfoque.** Tocar la fruta en la pantalla antes de disparar y revisar que la foto no esté movida.
7. **Segunda toma (opcional).** Si se toma otra foto de la misma fruta desde otro ángulo, ambas llevan el mismo `id_fruta`.
8. **Moneda de referencia (descartada).** Se había previsto incluir una moneda de 500 pesos como escala. Finalmente no se usó: la distancia de la cámara es fija y cada fruta se midió con regla, así que la escala píxel → cm se calibra con esas medidas (`docs/TRASPASO.md` §4.4).
9. **Medida con regla.** Después de cada foto, medir la fruta y anotar el valor en la columna `tamaño`. Esta medida es la referencia real para validar la estimación de tamaño.
   - **Manzana y guayaba:** el diámetro mayor.
   - **Banano:** el largo en línea recta de punta a punta.
10. **Privacidad.** No fotografiar personas, rostros ni datos de vendedores. En plazas o supermercados, pedir permiso antes de tomar fotos.

### 4.4 Cómo obtener las tres calidades

| Fruta | Foto `buena` | Foto `regular` | Foto `mala` |
|---|---|---|---|
| Banano | Día 0: amarillo, firme, pocas pecas | Día 3–4: muchas pecas cafés, zonas algo blandas | Día 6–8: cáscara mayormente negra, blando, rajado o con moho |
| Guayaba | Día 0: color uniforme, firme | Día 2–3: manchas pequeñas oscuras, algo blanda | Día 5–6: manchas negras grandes, zonas hundidas, podredumbre o moho |
| Manzana | Día 0: piel lisa y brillante, sin golpes | Día 1–2 después de un golpe leve contra una superficie dura: mancha menor de ~1 cm | Día 7–10 después de un golpe fuerte o un pinchazo pequeño en la piel: zonas cafés blandas o moho |

- Todas las frutas se mantienen a temperatura ambiente, fuera de la nevera.
- La manzana sana se deteriora en 2–3 semanas, demasiado lento para el plazo; el golpe y el pinchazo aceleran el proceso. Tras la foto `buena`, se golpea levemente. Tras la foto `regular`, se golpea con fuerza o se pincha la piel en la misma zona.
- Antes de cada foto, revisar el criterio de la sección 4.6. Si una fruta todavía no se ve `regular` o `mala`, se espera uno o dos días más. El día exacto importa menos que el estado visible.

### 4.5 Nombres de archivo

`<fruta>_<calidad>_<nn>.jpg`, donde `nn` es el número de la fruta física (01–05) y coincide con su `id_fruta`. Por ejemplo, la guayaba `gua_03` produce `guayaba_buena_03.jpg`, `guayaba_regular_03.jpg` y `guayaba_mala_03.jpg`. Una toma adicional desde otro ángulo lleva el sufijo `_b`: `guayaba_mala_03_b.jpg`. Las fotos de prueba de cámara llevan el prefijo `prueba_` y el nombre del autor: `prueba_<autor>_<fruta>_<calidad>_<nn>.jpg`. Todo en minúsculas, sin tildes ni espacios.

### 4.6 Criterios de calidad (comunes a fotos y recortes)

| Clase | Criterio |
|---|---|
| `buena` | Sin defectos visibles, color uniforme, apta para la venta |
| `regular` | Defectos leves: manchas pequeñas, golpe superficial, madurez irregular. Se vendería con descuento |
| `mala` | Podredumbre, moho, golpes profundos o deformación severa. No se vendería |

En caso de duda entre dos clases, se elige la peor. Si la duda persiste, se escribe "duda" en `notas` y se decide en grupo.

### 4.7 Registro en `data/labels/anotaciones_propias.csv`

Una fila por foto:

| Columna | Contenido | Ejemplo |
|---|---|---|
| `archivo` | Nombre del archivo en `data/raw/propias/` | `manzana_regular_04.jpg` |
| `id_fruta` | Identificador de la fruta física: 3 letras + número; igual en sus tres fotos | `man_04` |
| `fruta` | manzana / banano / guayaba | `manzana` |
| `calidad` | buena / regular / mala | `regular` |
| `tamaño` | Medida con regla, con unidad: diámetro mayor (manzana, guayaba) o largo de punta a punta (banano) | `7.5 cm` |

- `fruitq.data.read_annotations` convierte `tamaño` en la columna numérica `medida_cm`.
- La clase pequeño/mediano/grande se calcula por terciles de `medida_cm` dentro de cada fruta.
- Las fotos tomadas con otro celular se nombran con el prefijo `prueba_`. El lector las marca como `conjunto = prueba_camara`, y al resto como `principal`.

**Edición en Excel.**
- El archivo usa `;` como separador y codificación UTF-8 con BOM. Así Excel, con configuración regional en español, lo abre directamente en columnas y muestra bien las tildes.
- Al guardar, elegir **"CSV UTF-8 (delimitado por comas)"**.
- La medida puede escribirse con coma o punto decimal (`7,5 cm` o `7.5 cm`).
- El código lo lee con `fruitq.data.read_annotations`, que acepta separador `;` o `,`, codificación UTF-8 o Windows-1252, y coma o punto decimal.

### 4.8 Entrega de las fotos

- Subir las fotos **originales** a la carpeta compartida del grupo en Google Drive: las del conjunto principal en la subcarpeta de cada fruta y las de prueba en `prueba_camara/`.
- **No enviarlas por WhatsApp:** comprime las imágenes y elimina sus metadatos.
- Las fotos no se suben a GitHub, porque pesan demasiado. Solo se sube el CSV de anotaciones.

### 4.9 Verificación antes de entregar

- [ ] Conjunto principal: 45 fotos, 15 por fruta (5 frutas físicas × 3 estados).
- [ ] Cada fruta tiene sus tres fotos con el mismo `id_fruta` y en ninguna aparece el papel o la marca de identificación.
- [ ] En ninguna foto se ve la mesa, el borde de la cartulina, manos ni otros objetos.
- [ ] La fruta no toca el borde de la foto.
- [ ] Ninguna foto está movida, tiene flash o reflejos fuertes.
- [ ] Todas se tomaron desde 30 cm; el conjunto principal, con un solo celular.
- [ ] Están en formato JPG y los nombres siguen el formato de la sección 4.5.
- [ ] Hay una fila completa en el CSV por cada foto, con su `tamaño` medido con regla.

## 5. Flujo de trabajo con Git

1. Antes de empezar: `git pull`.
2. Agregar solo los archivos propios. Por ejemplo: `git add data/labels/recortes_<nombre>.csv`. No usar `git add -A` sobre cambios de otros.
3. `git commit -m "<descripción breve>"` y luego `git push`.
4. Si `git push` es rechazado, hacer `git pull` y repetir.
5. No subir imágenes, archivos `.zip`, modelos de prueba ni configuraciones locales de editores. Todo eso ya está en `.gitignore`; de `models/` solo se versionan los dos modelos finales.
6. Cada integrante envía su nombre completo y su código para la tabla de integrantes del `README.md`.

## 6. Video (máximo 10 minutos)

| Bloque | Duración | Responsable |
|---|---|---|
| Contexto del problema e impacto | 1,5 min | Integrante 3 |
| Datos: Kaggle, fotos propias, segmentación y EDA | 2 min | Integrante 2 |
| Modelos, resultados y comparación | 3 min | Sebastián Mejía |
| Demostración de la aplicación con cámara | 2 min | Sebastián Mejía |
| Ética, conclusiones y trabajo futuro | 1,5 min | Integrante 3 |

## 7. Distribución sugerida del informe (7 páginas)

| Sección | Extensión | Responsable |
|---|---|---|
| Título y resumen | 0,25 p. | Integrante 2 |
| Introducción | 0,75 p. | Integrante 2 |
| Fundamentos teóricos | 1 p. | Integrante 2 |
| Metodología (con diagrama CRISP-DM) | 1,5 p. | Sebastián Mejía + diagrama del integrante 3 |
| Resultados | 1,25 p. | Sebastián Mejía |
| Análisis de resultados (incluye literatura, ética e impacto) | 1,25 p. | Sebastián Mejía, integrantes 2 y 3 |
| Conclusiones y trabajo futuro | 0,5 p. | Todos |
| Referencias | 0,5 p. | Integrante 2 |
