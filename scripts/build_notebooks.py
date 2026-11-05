"""Genera (y opcionalmente ejecuta) los notebooks del proyecto a partir de celdas en código.

Mantener los notebooks como código facilita revisarlos en Git. Uso:
    python scripts/build_notebooks.py           # genera
    python scripts/build_notebooks.py --run     # genera y ejecuta (guarda salidas)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


def eda_notebook() -> nbf.NotebookNode:
    cells = [
        md("""
# Fase 2 — Comprensión de los datos (EDA)

Análisis exploratorio del dataset **Fruit Quality Classification** (Kaggle, `ryandpark`).

Objetivos:
1. Inventario y distribución de clases (calidad × fruta) → **desbalanceo**.
2. Propiedades de las imágenes: resolución, brillo, nitidez.
3. **Duplicados** exactos y casi duplicados (riesgo de fuga de datos entre train y test).
4. Variabilidad visual: fondos, número de frutas por imagen.
5. Distribución del tamaño (diámetro normalizado) por fruta.
6. Primera mirada a la separabilidad buena vs. mala con características de color.

Todas las figuras se guardan en `reports/figures/` en **PDF (vectorial)** para el informe.
"""),
        code("""
import sys, hashlib
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm

from fruitq.config import RAW_KAGGLE, FIGURES, PROCESSED, SEED
from fruitq.io import index_folder, read_image
from fruitq.segmentation import find_fruits, largest_fruit
from fruitq.size import normalized_diameter
from fruitq.features import extract_features, feature_names

FIGURES.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

def savefig(name):
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
"""),
        md("## 1. Inventario y distribución de clases"),
        code("""
df = index_folder(RAW_KAGGLE).rename(columns={"level_0": "carpeta", "level_1": "subcarpeta"})
calidad_map = {"Good Quality_Fruits": "buena", "Bad Quality_Fruits": "mala", "Mixed Qualit_Fruits": "mixta"}
df["calidad"] = df["carpeta"].map(calidad_map)
fruit_map = {"Lemon": "Lime"}  # en mixed se llama Lemon; en good/bad, Lime
df["fruta"] = df["subcarpeta"].str.replace(r"_(Good|Bad)$", "", regex=True).replace(fruit_map)
tabla = pd.crosstab(df["fruta"], df["calidad"], margins=True, margins_name="Total")
tabla
"""),
        code("""
cnt = df[df.calidad != "mixta"].groupby(["fruta", "calidad"]).size().unstack()
ax = cnt.plot(kind="bar", figsize=(8, 4), color={"buena": "#4C9A2A", "mala": "#B23A48"}, edgecolor="black")
ax.set_ylabel("Número de imágenes"); ax.set_xlabel(""); ax.set_title("Imágenes por fruta y calidad")
plt.xticks(rotation=0); savefig("eda_distribucion_clases"); plt.show()

n = df[df.calidad != "mixta"].groupby(["fruta", "calidad"]).size()
print(f"Razón de desbalanceo (máx/mín entre las 12 clases fruta×calidad): {n.max() / n.min():.2f}")
print(f"Buena vs mala global: {(df.calidad=='buena').sum()} vs {(df.calidad=='mala').sum()}")
"""),
        md("""
**Observaciones.** Las clases están balanceadas (~1100 imágenes) **excepto `Pomegranate_Good`**, que tiene ~5 veces más imágenes. Esto sesga la calidad global hacia `buena` y hay que tratarlo (submuestreo de granada buena o pesos de clase). El dataset solo trae **dos** niveles de calidad (buena / mala) más una carpeta `mixta` sin etiqueta por fruta. Para llegar a las **3 clases** que exige el enunciado se debe construir la clase intermedia (`regular`) con la anotación manual de los recortes de `mixed_quality` y de las fotos propias.
"""),
        md("## 2. Propiedades de las imágenes: resolución, brillo y nitidez"),
        code("""
def img_stats(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    ahash = "".join("1" if v else "0" for v in (small > small.mean()).ravel())
    return {
        "h": img.shape[0], "w": img.shape[1],
        "brillo": float(gray.mean()),
        "nitidez": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "md5": hashlib.md5(raw.tobytes()).hexdigest(),
        "ahash": ahash,
        # Imagen en escala de grises guardada como RGB: los 3 canales casi iguales
        "gris": bool(np.abs(img[:, :, 0].astype(int) - img[:, :, 1]).mean() < 2
                     and np.abs(img[:, :, 1].astype(int) - img[:, :, 2]).mean() < 2),
    }

cache = PROCESSED / "eda_image_stats.csv"
if cache.exists():
    stats = pd.read_csv(cache, dtype={"ahash": str})
else:
    stats = pd.DataFrame([img_stats(p) for p in tqdm(df["path"], desc="Leyendo imágenes")])
    stats.insert(0, "path", df["path"].values)
    stats.to_csv(cache, index=False)
df = df.merge(stats, on="path")
df["resolucion"] = df["w"].astype(str) + "x" + df["h"].astype(str)
df["resolucion"].value_counts().head(8)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
sub = df[df.calidad != "mixta"]
sns.boxplot(data=sub, x="fruta", y="brillo", hue="calidad", ax=axes[0],
            palette={"buena": "#4C9A2A", "mala": "#B23A48"}, fliersize=1)
axes[0].set_title("Brillo medio (0–255)"); axes[0].set_xlabel("")
sns.boxplot(data=sub, x="fruta", y="nitidez", hue="calidad", ax=axes[1],
            palette={"buena": "#4C9A2A", "mala": "#B23A48"}, fliersize=1, log_scale=True)
axes[1].set_title("Nitidez (varianza del Laplaciano, escala log)"); axes[1].set_xlabel("")
for ax in axes: ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); savefig("eda_brillo_nitidez"); plt.show()

umbral_borrosa = 50
print(f"Imágenes potencialmente borrosas (var. Laplaciano < {umbral_borrosa}): "
      f"{(df.nitidez < umbral_borrosa).sum()} ({(df.nitidez < umbral_borrosa).mean():.1%})")
"""),
        code("""
print("Imágenes en escala de grises por subcarpeta:")
print(df[df.gris].groupby("subcarpeta").size().to_string())
"""),
        md("""
**Hallazgo.** Todas las imágenes en escala de grises están en `Pomegranate_Good`. Son copias aumentadas de fotos a color (ver sección 3), y explican buena parte de su exceso de imágenes. Como el **color es la señal principal de calidad** (manchas, podredumbre), se **eliminan** en la preparación de datos.
"""),
        md("## 3. Duplicados (riesgo de fuga de información)"),
        code("""
dup_exact = df[df.duplicated("md5", keep=False)]
dup_near = df[df.duplicated("ahash", keep=False)]
print(f"Duplicados exactos (mismo MD5): {df.duplicated('md5').sum()} imágenes sobrantes")
print(f"Casi duplicados (mismo average-hash 8x8): {df.duplicated('ahash').sum()} imágenes sobrantes")
print()
print("Casi duplicados por subcarpeta:")
print(df[df.duplicated("ahash")].groupby("subcarpeta").size().sort_values(ascending=False).head(12).to_string())

cross = df.groupby("ahash")["calidad"].nunique()
print(f"\\nGrupos de casi duplicados con etiquetas DISTINTAS: {(cross > 1).sum()}")
"""),
        md("""
**Por qué importa.** Si una imagen y su copia (o una versión rotada o recortada) quedan una en *train* y otra en *test*, la métrica de test sobreestima la generalización (*data leakage*). La partición se debe hacer **por grupo de hash** (`GroupShuffleSplit`), no imagen por imagen. El average-hash solo detecta copias casi idénticas; las versiones aumentadas (rotadas o volteadas) requieren hashes invariantes. Esto se revisa visualmente abajo.
"""),
        code("""
grp = df[df.duplicated("ahash", keep=False)].groupby("ahash")
ejemplos = [g for _, g in grp if len(g) >= 2][:6]
if ejemplos:
    fig, axes = plt.subplots(len(ejemplos), 2, figsize=(4, 2 * len(ejemplos)))
    axes = np.atleast_2d(axes)
    for i, g in enumerate(ejemplos):
        for j in range(2):
            r = g.iloc[j]
            axes[i, j].imshow(cv2.cvtColor(read_image(r.path), cv2.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{r.subcarpeta}", fontsize=7); axes[i, j].axis("off")
    plt.tight_layout(); plt.show()
"""),
        md("## 4. Variabilidad visual"),
        code("""
rng = np.random.default_rng(SEED)
grupos = sorted(df.subcarpeta.unique())
fig, axes = plt.subplots(len(grupos), 6, figsize=(9, 1.6 * len(grupos)))
for i, s in enumerate(grupos):
    paths = df[df.subcarpeta == s].sample(6, random_state=SEED).path
    for j, p in enumerate(paths):
        axes[i, j].imshow(cv2.cvtColor(cv2.resize(read_image(p), (128, 128)), cv2.COLOR_BGR2RGB))
        axes[i, j].axis("off")
    axes[i, 0].set_title(s, fontsize=8, loc="left")
plt.tight_layout(); savefig("eda_mosaico"); plt.show()
"""),
        md("""
**Observaciones.** El supuesto del enunciado ("una fruta sobre fondo uniforme") **no se cumple** en buena parte del dataset. Hay manos sosteniendo la fruta, pasto, baldosas, cartulinas de colores y varias frutas por foto, también en las carpetas buena y mala. Consecuencias:
- Un modelo podría aprender **atajos** del fondo (p. ej. "pasto ⇒ mixta") en lugar de la fruta. Por eso se segmenta y se enmascara el fondo antes de extraer características.
- La segmentación con un umbral simple no basta. Se usa un modelo del color del fondo tomado del borde de la imagen, más GrabCut y watershed (ver `src/fruitq/segmentation.py`).
"""),
        code("""
muestra = df[df.calidad != "mixta"].groupby("subcarpeta").sample(40, random_state=SEED)
filas = []
for r in tqdm(muestra.itertuples(), total=len(muestra), desc="Segmentando muestra"):
    img = read_image(r.path)
    regs = find_fruits(img)
    fila = {"path": r.path, "fruta": r.fruta, "calidad": r.calidad, "n_frutas": len(regs)}
    if regs:
        main = regs[0]
        fila["d_norm"] = normalized_diameter(main.area, img.shape)
        f = extract_features(main.crop, main.mask)
        fila.update(dict(zip(feature_names(), f)))
    filas.append(fila)
seg = pd.DataFrame(filas)
seg.to_csv(PROCESSED / "eda_muestra_segmentada.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 3))
seg["n_frutas_c"] = seg["n_frutas"].clip(upper=6)
sns.countplot(data=seg, x="n_frutas_c", hue="calidad", palette={"buena": "#4C9A2A", "mala": "#B23A48"}, ax=ax)
ax.set_xlabel("Frutas detectadas por imagen (6 = 6 o más)"); ax.set_ylabel("Imágenes")
ax.set_title("Número de frutas detectadas (muestra de 40 por subcarpeta)")
savefig("eda_frutas_por_imagen"); plt.show()
print(f"Imágenes con exactamente 1 fruta: {(seg.n_frutas == 1).mean():.1%}")
"""),
        md("## 5. Distribución del tamaño (diámetro equivalente normalizado)"),
        code("""
fig, ax = plt.subplots(figsize=(8, 3.5))
sns.boxplot(data=seg.dropna(subset=["d_norm"]), x="fruta", y="d_norm", hue="calidad",
            palette={"buena": "#4C9A2A", "mala": "#B23A48"}, fliersize=1, ax=ax)
ax.set_ylabel(r"$\\tilde d = 2\\sqrt{A/\\pi}\\,/\\,\\min(H,W)$"); ax.set_xlabel("")
ax.set_title("Diámetro normalizado de la fruta principal")
savefig("eda_tamano"); plt.show()
seg.groupby("fruta")["d_norm"].describe()[["mean", "std", "min", "50%", "max"]]
"""),
        md("""
**Observaciones.** En Kaggle la distancia de la cámara **no está controlada**, así que $\\tilde d$ mezcla el tamaño real con el encuadre. Los terciles calculados sobre Kaggle **no** representan el tamaño físico. Para las fotos propias se fija la distancia (protocolo en `data/README.md`) y se calibran los umbrales con ellas, idealmente con una moneda de referencia para convertir a cm.
"""),
        md("## 6. Primera mirada a la separabilidad (características de color y defectos)"),
        code("""
cols = ["dark_frac", "lowsat_frac", "hsv_v_std", "lab_l_mean", "hsv_s_mean", "gradient_energy"]
long = seg.melt(id_vars=["fruta", "calidad"], value_vars=cols, var_name="caracteristica", value_name="valor")
g = sns.catplot(data=long, x="fruta", y="valor", hue="calidad", col="caracteristica", col_wrap=3,
                kind="box", sharey=False, height=2.6, aspect=1.5, fliersize=1,
                palette={"buena": "#4C9A2A", "mala": "#B23A48"})
g.set_xticklabels(rotation=30); g.set_xlabels("")
plt.savefig(FIGURES / "eda_separabilidad.pdf", bbox_inches="tight"); plt.show()
"""),
        code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = seg[feature_names()].dropna()
y = seg.loc[X.index, "calidad"]
Z = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
fig, ax = plt.subplots(figsize=(5.5, 4))
for c, col in (("buena", "#4C9A2A"), ("mala", "#B23A48")):
    m = (y == c).values
    ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, label=c, color=col)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("PCA de las 62 características manuales")
savefig("eda_pca"); plt.show()
"""),
        md("""
## Resumen de hallazgos (para `docs/fases/02_comprension_datos.md`)

Completar con los números que imprimen las celdas de arriba:
1. **Tamaño:** ~19.5k imágenes JPG de baja resolución (~256 px), 6 frutas, 2 calidades más una carpeta mixta.
2. **Desbalanceo:** `Pomegranate_Good` ≈ 5× las demás clases.
3. **Duplicados:** ver la sección 3. Se debe partir **por grupo** para evitar fuga de información.
4. **Fondos heterogéneos y varias frutas por imagen:** se requiere segmentación robusta y enmascarar el fondo.
5. **Tamaño:** $\\tilde d$ de Kaggle no es comparable entre fotos; se calibra con las fotos propias.
6. **Tercera clase de calidad:** no existe en el dataset y se construye con anotación manual.
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "fruitq", "display_name": "Python (fruitq)", "language": "python"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Ejecuta el notebook y guarda las salidas")
    args = ap.parse_args()

    NB_DIR.mkdir(exist_ok=True)
    out = NB_DIR / "02_eda.ipynb"
    nb = eda_notebook()
    if args.run:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=3600, kernel_name="fruitq",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbf.write(nb, out)
    print(f"Notebook escrito en {out}")


if __name__ == "__main__":
    main()
