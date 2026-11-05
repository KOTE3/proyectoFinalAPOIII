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

**Cantidad: 63 fotos (3 frutas × 3 calidades × 7 frutas distintas)**

| | buena | regular | mala | Total |
|---|---|---|---|---|
| Banano | 7 | 7 | 7 | 21 |
| Manzana | 7 | 7 | 7 | 21 |
| Guayaba | 7 | 7 | 7 | 21 |

- Dentro de cada celda, variar el tamaño: unas pequeñas, unas medianas y unas grandes.
- **Cada foto es una fruta física distinta.** Si se toman 2 ángulos de la misma fruta, ambas fotos llevan el mismo `id_fruta` (así nunca quedan una en train y otra en test).
- Para conseguir `mala` sin gastar: comprar algunas frutas maduras o golpeadas (en la plaza son más baratas) y dejar unas cuantas 3–5 días fuera de la nevera. Fotografiar el mismo lote el día 0 (`buena`), el día 2–3 (`regular`) y el día 5 (`mala`) también vale; en ese caso se usa el **mismo `id_fruta`** en todos los días, para que la misma fruta no quede repartida entre train y test.
- En **5–10 fotos** poner una **moneda de 500** junto a la fruta (medir su diámetro con regla y anotarlo) para convertir píxeles a centímetros.

## Protocolo de recolección

1. **Una fruta por foto**, centrada, sobre fondo **uniforme y sin textura** (cartulina blanca o negra).
2. **Distancia fija** de la cámara (~30 cm, el celular apoyado en una caja o en un libro) para que el tamaño en píxeles sea comparable entre fotos.
3. Luz difusa, sin flash directo ni sombras fuertes (cerca de una ventana, de día).
4. Nombre de archivo: `<fruta>_<calidad>_<nnn>.jpg`, p. ej. `guayaba_mala_007.jpg`.
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
| `fruta` | manzana, tomate, banano, … |
| `calidad` | buena / regular / mala |
| `tamano` | pequeno / mediano / grande |
| `defecto` | ninguno / mancha / golpe / podredumbre / moho / otro |
| `fuente` | casa / supermercado / plaza |
| `autor` | Integrante que tomó la foto |
| `fecha` | AAAA-MM-DD |
| `notas` | Observaciones libres |
