# Datos

Las imágenes **no se versionan** en Git (pesan demasiado). Solo se versionan las anotaciones en `labels/`.
Para compartir las fotos con la clase usar la carpeta común acordada (Drive / repositorio): _enlace por completar_.

## Fuentes

| Carpeta | Origen | Cómo se obtiene |
|---|---|---|
| `raw/kaggle/` | Fruit Quality Classification (Kaggle) | `python scripts/download_data.py` |
| `raw/propias/` | Fotos tomadas por el grupo | Protocolo de abajo |
| `processed/manifest.csv` | Lista limpia de imágenes con fruta, calidad, grupo y split | `python scripts/prepare_dataset.py` |
| `interim/segmentadas/` | Frutas individuales recortadas de `mixed_quality` | `python scripts/segment_mixed.py` |

## Qué fotografiar (plan del grupo)

**Frutas:** se eligen **3 de las 6 que trae Kaggle**, para que las fotos propias complementen al dataset y sirvan para medir la generalización: **banano, manzana y guayaba**. Son baratas, se consiguen todo el año en Colombia y se dañan rápido, así que es fácil tener las tres calidades. La naranja y el limón sirven de reemplazo; la granada es difícil de conseguir.

**Cantidad: 45 fotos de 15 frutas físicas.** Son 5 frutas de cada tipo, y cada una se fotografía en sus tres estados (`buena` → `regular` → `mala`) a medida que se deteriora.

| | Frutas físicas | buena | regular | mala | Total |
|---|---|---|---|---|---|
| Banano | 5 | 5 | 5 | 5 | 15 |
| Manzana | 5 | 5 | 5 | 5 | 15 |
| Guayaba | 5 | 5 | 5 | 5 | 15 |

- Las 5 frutas de cada tipo se eligen de tamaños distintos (pequeñas, medianas y grandes).
- Las tres fotos de una misma fruta llevan el **mismo `id_fruta`**, para que nunca queden repartidas entre train y test.
- En 2–3 fotos por fruta se pone una **moneda de 500** junto a la fruta. Su diámetro se mide con regla y se anota, para convertir píxeles a centímetros.
- Además, los integrantes 2 y 3 toman 5–10 fotos cada uno con su propio celular (`conjunto = prueba_camara`). Esas fotos solo se usan para evaluar la generalización a otras cámaras.

## Protocolo de recolección

Resumen; el protocolo completo (montaje, configuración de la cámara, obtención de las tres calidades y verificación) está en [`docs/PLAN_DE_TRABAJO.md`](../docs/PLAN_DE_TRABAJO.md#4-protocolo-de-toma-de-fotografías).

1. **Una fruta por foto**, centrada, sobre fondo **uniforme y sin textura** (cartulina blanca o negra).
2. **Distancia fija** de la cámara (~30 cm, el celular apoyado en una caja o en un libro) para que el tamaño en píxeles sea comparable entre fotos.
3. Luz difusa, sin flash directo ni sombras fuertes (cerca de una ventana, de día).
4. Nombre de archivo: `<fruta>_<calidad>_<nn>.jpg`, con `nn` igual al número de la fruta física. Por ejemplo, `guayaba_mala_03.jpg` corresponde a `gua_03`.
5. **No fotografiar personas, caras, placas ni datos de los vendedores.** Pedir permiso antes de tomar fotos en plazas o supermercados.

## Definición de etiquetas

> Definirlas como grupo **antes** de etiquetar y mantenerlas consistentes con la clase.

**Calidad** (propuesta, ajustar):

| Clase | Criterio |
|---|---|
| `buena` | Sin defectos visibles, color uniforme, apta para venta |
| `regular` | Defectos leves (manchas pequeñas, golpe superficial, madurez irregular) |
| `mala` | Podredumbre, moho, golpes profundos o deformación severa |

**Tamaño**: `pequeno` / `mediano` / `grande`. Se calcula automáticamente con `fruitq.size` (terciles del diámetro equivalente normalizado **por tipo de fruta**) y se revisa manualmente.

## Formato de `labels/anotaciones_propias.csv`

| Columna | Descripción |
|---|---|
| `archivo` | Ruta relativa a `data/raw/propias/` |
| `id_fruta` | Identificador de la fruta física (p. ej. `ban_03`); igual en varias fotos de la misma fruta |
| `fruta` | manzana / banano / guayaba |
| `calidad` | buena / regular / mala |
| `tamano` | pequeno / mediano / grande; se calcula por terciles de `medida_cm` dentro de cada fruta |
| `medida_cm` | Medida con regla: diámetro mayor (manzana, guayaba) o largo de punta a punta (banano) |
| `defecto` | ninguno / mancha / golpe / podredumbre / moho / arrugas / otro |
| `fuente` | casa / supermercado / plaza |
| `celular` | Marca y modelo del celular usado |
| `moneda` | si / no: la foto incluye la moneda de 500 pesos de referencia |
| `conjunto` | principal (entrenamiento y evaluación) / prueba_camara (solo prueba de generalización a otro celular) |
| `autor` | Integrante que tomó la foto |
| `fecha` | AAAA-MM-DD |
| `notas` | Observaciones libres; diámetro medido de la moneda cuando aparece |
