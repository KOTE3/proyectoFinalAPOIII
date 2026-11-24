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
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("PCA de las 57 características manuales")
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


def modelos_clasicos_notebook() -> nbf.NotebookNode:
    cells = [
        md(r"""
# Fase 4 — Modelado con ML clásico (características manuales)

**Tarea:** clasificación binaria buena (0) / mala (1) con las 57 características de `fruitq.features`, extraídas de la fruta principal segmentada (`scripts/extract_features.py`). La clase `regular` se agrega después a partir de la probabilidad $P(\text{mala}\mid x)$ (sección 6).

**Modelos**
| Modelo | Por qué se prueba |
|---|---|
| Clase mayoritaria | Piso absoluto: cualquier modelo debe superarlo |
| Regresión logística | **Línea base** lineal e interpretable |
| SVM con kernel RBF | Frontera no lineal y de margen máximo; buena con ~60 dimensiones y ~10⁴ muestras |
| Random Forest | Captura interacciones (p. ej. "oscuro **y** de baja saturación"), es robusto a la escala y da importancias |
| XGBoost | *Boosting*: suele ser el más fuerte con datos tabulares |

**Protocolo**
- Búsqueda de hiperparámetros con `GridSearchCV` y **`StratifiedGroupKFold(k=5)` sobre train**: los grupos de casi duplicados nunca se separan entre pliegues.
- Métrica de selección: **F1-macro**. La validación (`val`) compara modelos; el **test se usa una sola vez**, con el modelo elegido.
- Pesos de muestra $w_i = \dfrac{N}{K\,N_{e(i)}}$, con $e(i)$ el estrato fruta×calidad, para compensar el exceso de granada buena: cada estrato aporta lo mismo a la pérdida.
"""),
        code("""
import sys, time, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (f1_score, accuracy_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from fruitq.config import PROCESSED, FIGURES, MODELS, SEED
from fruitq.features import feature_names

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
MODELS.mkdir(exist_ok=True)
def savefig(name):
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight"); plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)

df = pd.read_parquet(PROCESSED / "features.parquet")
df = df[df.error.fillna("") == ""].reset_index(drop=True)
FEATS = feature_names()
df["y"] = (df.calidad == "mala").astype(int)
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
Xtr, ytr, gtr = tr[FEATS].values, tr.y.values, tr.grupo.values
Xva, yva, Xte, yte = va[FEATS].values, va.y.values, te[FEATS].values, te.y.values

n_est = tr.estrato.value_counts()
w_tr = (len(tr) / (len(n_est) * tr.estrato.map(n_est))).values
print(f"train {len(tr)} | val {len(va)} | test {len(te)}   (mala en train: {ytr.mean():.1%})")
print(f"Segmentación sin fruta detectada (se usó la imagen completa): {(~df.seg_ok).mean():.1%}")
"""),
        md("## 1. Búsqueda de hiperparámetros"),
        code("""
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
pipe = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])

busquedas = {
    "Regresión logística": (pipe(LogisticRegression(max_iter=3000)),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)": (pipe(SVC(kernel="rbf")),
                  {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.003, 0.01, 0.03]}),
    "Random Forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
                      {"clf__max_depth": [None, 20, 10], "clf__min_samples_leaf": [1, 3, 10],
                       "clf__max_features": ["sqrt", 0.3]}),
    "XGBoost": (pipe(XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                                   random_state=SEED, n_jobs=1, eval_metric="logloss")),
                {"clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1]}),
}

resultados, mejores = [], {}
for nombre, (est, grid) in busquedas.items():
    t0 = time.time()
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1, return_train_score=True)
    gs.fit(Xtr, ytr, groups=gtr, clf__sample_weight=w_tr)
    mejores[nombre] = gs
    i = gs.best_index_
    resultados.append({
        "modelo": nombre, "mejores hiperparámetros": gs.best_params_,
        "F1 train (CV)": gs.cv_results_["mean_train_score"][i],
        "F1 val (CV)": gs.best_score_, "± std": gs.cv_results_["std_test_score"][i],
        "minutos": (time.time() - t0) / 60,
    })
    print(f"{nombre:<20} F1-CV = {gs.best_score_:.3f}  {gs.best_params_}  ({(time.time()-t0)/60:.1f} min)")
pd.DataFrame(resultados).round(3)
"""),
        md(r"""
**Lectura del sobreajuste.** La diferencia entre "F1 train (CV)" y "F1 val (CV)" estima la **brecha de generalización**. Random Forest y XGBoost profundos suelen llegar a F1 ≈ 1 en train: memorizan. Lo que importa es el F1 de validación y que su desviación entre pliegues sea pequeña.
"""),
        code("""
# Efecto de los hiperparámetros de la SVM (evidencia del ajuste para el informe)
r = pd.DataFrame(mejores["SVM (RBF)"].cv_results_)
r["gamma"] = r["param_clf__gamma"].astype(str); r["C"] = r["param_clf__C"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, col, titulo in ((axes[0], "mean_train_score", "F1 en entrenamiento"), (axes[1], "mean_test_score", "F1 en validación cruzada")):
    sns.heatmap(r.pivot(index="gamma", columns="C", values=col), annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar=False)
    ax.set_title(titulo)
plt.suptitle("SVM-RBF: F1-macro según C y γ"); plt.tight_layout(); savefig("modelos_svm_grid"); plt.show()
"""),
        md(r"""
En la SVM, $C$ penaliza los errores de entrenamiento en $\min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_i w_i\,\xi_i$, y $\gamma$ controla el alcance del kernel $K(x,x') = \exp(-\gamma\|x-x'\|^2)$. Con $C$ y $\gamma$ grandes, la frontera se ajusta a cada punto: el F1 de train sube y el de validación baja (**sobreajuste**). Con valores pequeños, la frontera es demasiado suave (**subajuste**).
"""),
        md("## 2. Comparación en el conjunto de validación"),
        code("""
def scores(model, X):
    if hasattr(model, "predict_proba"):
        try: return model.predict_proba(X)[:, 1]
        except AttributeError: pass
    return model.decision_function(X)

dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
modelos = {"Clase mayoritaria": dummy, **{k: v.best_estimator_ for k, v in mejores.items()}}
filas = []
for nombre, m in modelos.items():
    t0 = time.perf_counter(); p = m.predict(Xva); ms = 1000 * (time.perf_counter() - t0) / len(Xva)
    s = scores(m, Xva) if nombre != "Clase mayoritaria" else np.zeros(len(yva))
    filas.append({"modelo": nombre, "accuracy": accuracy_score(yva, p), "F1-macro": f1_score(yva, p, average="macro"),
                  "recall(mala)": recall_score(yva, p), "AUC": roc_auc_score(yva, s) if s.std() > 0 else 0.5,
                  "ms/imagen": ms})
comp = pd.DataFrame(filas).set_index("modelo").round(3)
comp.to_csv(FIGURES.parent / "tabla_modelos_clasicos_val.csv")
comp
"""),
        code("""
ax = comp[["accuracy", "F1-macro", "recall(mala)", "AUC"]].plot(kind="bar", figsize=(9, 3.6), edgecolor="black", width=0.8)
ax.set_ylim(0, 1); ax.set_ylabel("Valor (validación)"); ax.set_xlabel(""); plt.xticks(rotation=15)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)); savefig("modelos_comparacion_val"); plt.show()
"""),
        md("## 3. Modelo elegido: evaluación final en test (una sola vez)"),
        code("""
elegido = comp.drop(index="Clase mayoritaria")["F1-macro"].idxmax()
best = modelos[elegido]
print("Modelo elegido por F1-macro en validación:", elegido)
p_te = best.predict(Xte); s_te = scores(best, Xte)
print(classification_report(yte, p_te, target_names=["buena", "mala"], digits=3))
print(f"AUC test: {roc_auc_score(yte, s_te):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, norm, t in ((axes[0], None, "Conteos"), (axes[1], "true", "Normalizada por fila (recall)")):
    ConfusionMatrixDisplay.from_predictions(yte, p_te, display_labels=["buena", "mala"], normalize=norm,
                                            cmap="Blues", ax=ax, colorbar=False, values_format=".2f" if norm else "d")
    ax.set_title(t)
plt.suptitle(f"{elegido} — matriz de confusión en test"); plt.tight_layout(); savefig("modelos_confusion_test"); plt.show()
"""),
        code("""
te["pred"] = p_te; te["acierto"] = te.pred == te.y
por_fruta = te.groupby(["fruta", "calidad"]).acierto.mean().unstack().round(3)
ax = por_fruta.plot(kind="bar", figsize=(8, 3.3), color={"buena": "#4C9A2A", "mala": "#B23A48"}, edgecolor="black")
ax.set_ylim(0, 1); ax.set_ylabel("Exactitud en test"); ax.set_xlabel(""); plt.xticks(rotation=0)
ax.set_title(f"{elegido}: exactitud por fruta y calidad"); savefig("modelos_por_fruta"); plt.show()
por_fruta
"""),
        md("## 4. ¿Qué características usa el modelo?"),
        code("""
pi = permutation_importance(best, Xva, yva, scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False).head(15)
ax = imp[::-1].plot(kind="barh", figsize=(6, 4.5), color="#3B6EA5")
ax.set_xlabel("Caída del F1-macro al permutar la característica"); ax.set_title(f"Importancia por permutación ({elegido})")
savefig("modelos_importancia"); plt.show()
"""),
        md(r"""
La importancia por permutación mide cuánto empeora el F1 en validación si se baraja una característica, lo que rompe su relación con $y$. Es agnóstica al modelo y, a diferencia de la importancia Gini del RF, no favorece a las variables continuas con muchos valores.
"""),
        md("## 5. Galería de errores"),
        code("""
import cv2
from fruitq.io import read_image_max
err = te[~te.acierto].sample(min(12, (~te.acierto).sum()), random_state=SEED)
fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
for ax, r in zip(axes.ravel(), err.itertuples()):
    ax.imshow(cv2.cvtColor(read_image_max(r.path, 256), cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title(f"{r.fruta}\\nreal: {r.calidad}", fontsize=8)
plt.suptitle("Errores en test (etiqueta real)"); plt.tight_layout(); savefig("modelos_errores"); plt.show()
"""),
        md(r"""
## 6. De 2 a 3 clases: umbrales sobre $P(\text{mala}\mid x)$

En vez de entrenar un clasificador de 3 clases sin ejemplos de `regular`, se usa la probabilidad del modelo binario como una **puntuación de deterioro** $s(x) \in [0,1]$:

$$\hat y = \begin{cases}\text{buena} & s(x) < t_1\\ \text{regular} & t_1 \le s(x) < t_2 \\ \text{mala} & s(x) \ge t_2\end{cases}$$

Los umbrales $t_1, t_2$ se ajustan con los recortes etiquetados a mano como `regular`. El histograma muestra cómo se reparte $s(x)$ en test: la zona intermedia es la que ocupará la clase `regular`.
"""),
        code("""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
# La SVM no da probabilidades: se calibra (Platt, sigmoide) sobre validación sin reentrenarla
cal = best if hasattr(best, "predict_proba") else CalibratedClassifierCV(FrozenEstimator(best), method="sigmoid").fit(Xva, yva)
p_mala = cal.predict_proba(Xte)[:, 1]
fig, ax = plt.subplots(figsize=(7, 3.2))
for c, col in (("buena", "#4C9A2A"), ("mala", "#B23A48")):
    ax.hist(p_mala[te.calidad == c], bins=40, alpha=0.6, color=col, label=c, density=True)
ax.axvspan(0.35, 0.65, color="#E8B931", alpha=0.25, label="zona 'regular' (ilustrativa)")
ax.set_xlabel(r"$P(\\mathrm{mala} \\mid x)$"); ax.set_ylabel("Densidad"); ax.legend()
ax.set_title("Distribución de la probabilidad de 'mala' en test"); savefig("modelos_prob_mala"); plt.show()
joblib.dump({"modelo": cal, "nombre": elegido, "features": FEATS}, MODELS / "clasico_mejor.joblib")
print("Modelo guardado en models/clasico_mejor.joblib")
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "fruitq", "display_name": "Python (fruitq)", "language": "python"}
    return nb


def cnn_notebook() -> nbf.NotebookNode:
    cells = [
        md(r"""
# Fase 4b/5 — CNN desde cero y comparación con ML clásico

La CNN se entrena con `scripts/train_cnn.py`; este notebook **lee los resultados guardados** (historias, predicciones y pesos) y los analiza.

- **Entrada:** recorte 128×128 de la fruta principal con el fondo en negro (`scripts/cache_tiles.py`).
- **Arquitectura:** 3 bloques [Conv3×3 → BN → ReLU]×2 → MaxPool, con 16/32/64 filtros, Global Average Pooling, Dropout y capa densa. ~72 mil parámetros (ver `src/fruitq/cnn.py`).
- **Entrenamiento:** AdamW, decaimiento coseno, entropía cruzada $\mathcal L = -\frac1N\sum_i \log \mathrm{softmax}(z_i)_{y_i}$, muestreo balanceado por estrato, aumento con las 8 simetrías del cuadrado y brillo/contraste, parada temprana con el F1 de validación.
"""),
        code("""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib, torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from scipy.stats import binomtest
from fruitq.config import PROCESSED, FIGURES, MODELS, SEED
from fruitq.cnn import SmallCNN, MEAN, STD, TILE

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
CNN = ROOT / "reports" / "cnn"
def savefig(name):
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight"); plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
"""),
        md("## 1. Búsqueda de hiperparámetros"),
        code("""
res = pd.DataFrame([json.loads(p.read_text()) for p in sorted(CNN.glob("*_resumen.json"))])
res["lr"] = res.args.map(lambda a: a["lr"]); res["dropout"] = res.args.map(lambda a: a["dropout"])
tabla = res[["tag", "lr", "dropout", "params", "mejor_epoca", "epocas", "f1_val", "min_total"]].sort_values("f1_val", ascending=False)
tabla.round(4)
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for tag in res.tag:
    h = pd.read_csv(CNN / f"{tag}_historia.csv")
    l, = axes[0].plot(h.epoca, h.loss_train, label=f"{tag} train")
    axes[0].plot(h.epoca, h.loss_val, "--", color=l.get_color(), label=f"{tag} val")
    axes[1].plot(h.epoca, h.acc_train, color=l.get_color()); axes[1].plot(h.epoca, h.acc_val, "--", color=l.get_color())
    axes[2].plot(h.epoca, h.f1_val, color=l.get_color(), label=tag)
axes[0].set_title("Pérdida (— train, -- val)"); axes[1].set_title("Exactitud (— train, -- val)"); axes[2].set_title("F1-macro en validación")
for a in axes: a.set_xlabel("época")
axes[0].legend(fontsize=6); axes[2].legend(fontsize=7)
plt.tight_layout(); savefig("cnn_curvas"); plt.show()
"""),
        md(r"""
**Cómo leer las curvas.** Si la pérdida de validación sube mientras la de entrenamiento sigue bajando, hay **sobreajuste**, y la parada temprana guarda la época anterior a ese punto. Aquí la exactitud de train se mide *con aumento de datos*, una tarea más difícil que validación, así que puede quedar por debajo de val sin que eso sea un error.
"""),
        md("## 2. Mejor CNN en test vs. XGBoost"),
        code("""
best_tag = tabla.iloc[0].tag
pc = pd.read_csv(CNN / f"{best_tag}_predicciones.csv")
pc_te = pc[pc.split == "test"].reset_index(drop=True)

feats = pd.read_parquet(PROCESSED / "features.parquet")
xgb = joblib.load(MODELS / "clasico_mejor.joblib")
te = feats[feats.split == "test"].set_index("path").loc[pc_te.path]
pc_te["p_xgb"] = xgb["modelo"].predict_proba(te[xgb["features"]].values)[:, 1]
pc_te["p_ens"] = (pc_te.p_mala + pc_te.p_xgb) / 2   # ensamble: promedio de probabilidades

y = pc_te.y.values
filas = []
for nombre, col in ((f"CNN ({best_tag})", "p_mala"), (f"XGBoost", "p_xgb"), ("Ensamble CNN+XGB", "p_ens")):
    p = pc_te[col].values; yhat = (p >= 0.5).astype(int)
    filas.append({"modelo": nombre, "accuracy": accuracy_score(y, yhat), "F1-macro": f1_score(y, yhat, average="macro"),
                  "recall(mala)": recall_score(y, yhat), "AUC": roc_auc_score(y, p)})
comp = pd.DataFrame(filas).set_index("modelo").round(4)
comp.to_csv(FIGURES.parent / "tabla_cnn_vs_xgb_test.csv")
comp
"""),
        md(r"""
### ¿La diferencia es significativa? Prueba de McNemar

Ambos modelos se evalúan sobre **las mismas** imágenes, así que se comparan sus desacuerdos: $b$ = imágenes que solo acierta la CNN y $c$ = imágenes que solo acierta XGBoost. Bajo $H_0$ (los dos son igual de buenos), cada desacuerdo cae de un lado o del otro con probabilidad ½, es decir $b \sim \mathrm{Bin}(b+c,\ \tfrac12)$. Un valor $p < 0.05$ indica que la diferencia no se explica por azar.
"""),
        code("""
ok_cnn = (pc_te.p_mala >= 0.5).astype(int) == y
ok_xgb = (pc_te.p_xgb >= 0.5).astype(int) == y
b, c = int((ok_cnn & ~ok_xgb).sum()), int((~ok_cnn & ok_xgb).sum())
p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
print(f"Solo acierta la CNN: b = {b} | solo acierta XGBoost: c = {c} | McNemar exacto p = {p:.4f}")
print(f"Ambos fallan: {int((~ok_cnn & ~ok_xgb).sum())} imágenes (errores 'duros', posible ruido de etiqueta)")
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, col, t in ((axes[0], "p_mala", f"CNN ({best_tag})"), (axes[1], "p_xgb", "XGBoost")):
    ConfusionMatrixDisplay.from_predictions(y, (pc_te[col] >= 0.5).astype(int), display_labels=["buena", "mala"],
                                            normalize="true", values_format=".3f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(t)
plt.tight_layout(); savefig("cnn_vs_xgb_confusion"); plt.show()

pc_te["ok_cnn"], pc_te["ok_xgb"] = ok_cnn, ok_xgb
pc_te.groupby(["fruta", "calidad"])[["ok_cnn", "ok_xgb"]].mean().round(3)
"""),
        md("## 3. Calibración: ¿las probabilidades sirven para la clase `regular`?"),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for col, nombre in (("p_mala", "CNN"), ("p_xgb", "XGBoost")):
    fr, mp = calibration_curve(y, pc_te[col], n_bins=10, strategy="quantile")
    axes[0].plot(mp, fr, "o-", label=nombre)
    axes[1].hist(pc_te[col], bins=40, alpha=0.55, label=nombre, density=True)
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfecta")
axes[0].set_xlabel("probabilidad predicha"); axes[0].set_ylabel("fracción real de 'mala'"); axes[0].set_title("Diagrama de confiabilidad"); axes[0].legend()
axes[1].set_xlabel(r"$P(\\mathrm{mala}\\mid x)$"); axes[1].set_title("Distribución de las probabilidades (test)"); axes[1].legend()
plt.tight_layout(); savefig("cnn_calibracion"); plt.show()
for col, nombre in (("p_mala", "CNN"), ("p_xgb", "XGBoost")):
    print(f"{nombre}: {((pc_te[col] > 0.2) & (pc_te[col] < 0.8)).mean():.1%} de las imágenes con probabilidad intermedia (0.2–0.8)")
"""),
        md("## 4. ¿Dónde mira la red? Grad-CAM"),
        md(r"""
Grad-CAM (Selvaraju et al., 2017) pondera los mapas de activación $A^k$ del último bloque convolucional por la importancia de cada canal, $\alpha_k = \frac{1}{HW}\sum_{i,j} \partial z_{\text{mala}} / \partial A^k_{ij}$, y calcula $\mathrm{ReLU}\left(\sum_k \alpha_k A^k\right)$. Las zonas cálidas son las que más empujan la predicción hacia `mala`. Si la red funciona bien, deberían coincidir con las manchas o la podredumbre.
"""),
        code("""
ck = torch.load(MODELS / f"cnn_{best_tag}.pt")
net = SmallCNN(2, tuple(ck["widths"]), ck["dropout"]); net.load_state_dict(ck["state_dict"]); net.eval()
tiles = np.load(PROCESSED / f"tiles{TILE}.npy", mmap_mode="r")
m = pd.read_csv(PROCESSED / "manifest.csv"); pos = pd.Series(m.index, index=m.path)

acts = {}
h = net.features[-1].register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
def gradcam(tile):
    x = torch.from_numpy(((tile.astype(np.float32) / 255 - MEAN) / STD)).permute(2, 0, 1)[None].requires_grad_(True)
    z = net(x); acts["a"].retain_grad(); z[0, 1].backward()
    a, g = acts["a"][0], acts["a"].grad[0]
    cam = torch.relu((g.mean((1, 2))[:, None, None] * a).sum(0)).detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    import cv2
    return cv2.resize(cam, (TILE, TILE)), torch.softmax(z, 1)[0, 1].item()

sel = pd.concat([pc_te[(pc_te.y == 1) & ok_cnn].sample(6, random_state=SEED), pc_te[(pc_te.y == 0) & ok_cnn].sample(3, random_state=SEED),
                 pc_te[~ok_cnn].sample(3, random_state=SEED)])
fig, axes = plt.subplots(2, 12, figsize=(16, 3.2))
for j, r in enumerate(sel.itertuples()):
    t = np.asarray(tiles[pos[r.path]]); cam, pm = gradcam(t)
    axes[0, j].imshow(t); axes[0, j].set_title(f"{r.fruta}\\n{r.calidad} · p={pm:.2f}", fontsize=7)
    axes[1, j].imshow(t); axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
for a in axes.ravel(): a.axis("off")
plt.suptitle("Grad-CAM hacia 'mala' (6 malas acertadas, 3 buenas acertadas, 3 errores)", y=1.02); savefig("cnn_gradcam"); plt.show()
h.remove()
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "fruitq", "display_name": "Python (fruitq)", "language": "python"}
    return nb


def tamano_notebook() -> nbf.NotebookNode:
    cells = [
        md(r"""
# Fases 4 y 5 — Tamaño en centímetros con las fotos propias

Estimación del tamaño físico de la fruta a partir de una sola foto, con las 45 fotos del conjunto principal (`data/labels/anotaciones_propias.csv`). Especificación: `docs/TRASPASO.md` §4.4.

**Formulación.** La cámara estuvo a distancia fija (30 cm), así que la longitud real es proporcional a la longitud en píxeles:

$$\text{medida}_{cm} \approx k \cdot L_{px}, \qquad \hat k = \operatorname{mediana}_i\left(\frac{\text{medida}_{cm,i}}{L_{px,i}}\right).$$

- $L_{px}$ es el lado mayor del rectángulo de área mínima que contiene la máscara (`cv2.minAreaRect`), la misma magnitud que se midió con regla: diámetro mayor en manzana y guayaba, largo de punta a punta en banano.
- La mediana, a diferencia de la media o de los mínimos cuadrados, es robusta a fotos tomadas a una altura algo distinta.
- $\hat k$ se valida **dejando una fruta fuera** (`id_fruta`): las tres fotos de una fruta nunca se evalúan con un $\hat k$ ajustado con ellas.
- Las clases pequeño / mediano / grande se definen con los terciles de `medida_cm` dentro de cada fruta.

Todas las imágenes se reducen a 1024 px por el lado mayor (`read_image_max`), por lo que $\hat k$ solo es válido a esa escala.
"""),
        code("""
import sys, json, math
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, cv2
from PIL import Image, ExifTags
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from fruitq.config import FIGURES, MODELS, RAW_PROPIAS, LABELS, SIZE_CLASSES
from fruitq.io import project_path, read_image_max
from fruitq.data import read_annotations
from fruitq.segmentation import fruit_on_uniform_background, largest_fruit
from fruitq.size import (characteristic_length, cross_validated_cm, equivalent_diameter,
                         fit_scale, fit_thresholds, size_class)

MAX_SIDE = 1024
FRUTAS = ["banano", "guayaba", "manzana"]
COLOR = {"banano": "#2a78d6", "guayaba": "#eb6834", "manzana": "#1baf7a"}
MARCA = {"banano": "o", "guayaba": "s", "manzana": "^"}

sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
def savefig(name):
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight"); plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
"""),
        md("## 1. Medición en píxeles"),
        md(r"""
Cada foto se segmenta con `fruit_on_uniform_background`, que excluye la sombra; si no detecta fruta se usa `largest_fruit`. Además de $L_{px}$ se calculan, para comparar, el diámetro equivalente $d = 2\sqrt{A/\pi}$ (definición inicial de la especificación para manzana y guayaba) y el diámetro de Feret máximo (mayor distancia entre dos puntos del contorno). El celular y la fecha se leen de los metadatos EXIF.
"""),
        code("""
anot = read_annotations(LABELS / "anotaciones_propias.csv")
anot = anot[anot.conjunto == "principal"].reset_index(drop=True)
TAG = {v: k for k, v in ExifTags.TAGS.items()}

def exif(path):
    ex = Image.open(path).getexif()
    sub = ex.get_ifd(0x8769) if ex else {}
    fecha = sub.get(TAG["DateTimeOriginal"]) or ex.get(TAG["DateTime"]) or ""
    return (ex.get(TAG["Model"]) or "sin metadatos"), fecha[:10].replace(":", "-")

filas, sin_fondo_uniforme = [], []
for r in anot.itertuples():
    ruta = RAW_PROPIAS / r.archivo
    img = read_image_max(ruta, MAX_SIDE)
    reg = fruit_on_uniform_background(img)
    if reg is None:
        sin_fondo_uniforme.append(r.archivo); reg = largest_fruit(img)
    hull = cv2.convexHull(reg.contour).reshape(-1, 2).astype(float)
    feret = np.sqrt(((hull[:, None] - hull[None]) ** 2).sum(-1)).max()
    celular, fecha = exif(ruta)
    filas.append(dict(archivo=r.archivo, id_fruta=r.id_fruta, fruta=r.fruta, calidad=r.calidad, medida_cm=r.medida_cm,
                      celular=celular, fecha=fecha, L_px=characteristic_length(reg.contour),
                      d_eq_px=equivalent_diameter(reg.area), feret_px=feret))
tam = pd.DataFrame(filas)
print(f"{len(tam)} fotos medidas; segmentadas con largest_fruit: {sin_fondo_uniforme or 'ninguna'}")
tam.groupby("fruta")[["medida_cm", "L_px", "d_eq_px", "feret_px"]].agg(["min", "median", "max"]).round(1)
"""),
        md("## 2. Factor de escala y validación cruzada agrupada"),
        code("""
variantes = {
    "Lado mayor del rectángulo mínimo (elegida)": tam.L_px,
    "Diámetro equivalente; banano: lado mayor": np.where(tam.fruta == "banano", tam.L_px, tam.d_eq_px),
    "Feret máximo": tam.feret_px,
}
filas = []
for nombre, L in variantes.items():
    pred = cross_validated_cm(tam.medida_cm, L, tam.id_fruta)
    err = np.abs(pred - tam.medida_cm)
    ratio = tam.medida_cm / L
    fila = {"medida en píxeles": nombre, "k (todas las fotos)": fit_scale(tam.medida_cm, L),
            "CV de medida/L (%)": 100 * ratio.std() / ratio.mean(), "EAM global (cm)": err.mean()}
    fila.update({f"EAM {f} (cm)": err[tam.fruta == f].mean() for f in FRUTAS})
    fila["error relativo medio (%)"] = 100 * (err / tam.medida_cm).mean()
    filas.append(fila)
comparacion = pd.DataFrame(filas).set_index("medida en píxeles")
comparacion.round(4)
"""),
        md(r"""
La variante elegida se fija por la **definición de la medida con regla** (diámetro mayor y largo de punta a punta), no por su error; las otras dos se muestran como comparación. EAM = error absoluto medio de la estimación validada (cada fruta estimada con un $\hat k$ que no la incluye).
"""),
        code("""
K = fit_scale(tam.medida_cm, tam.L_px)                       # k final, con las 45 fotos, para la aplicación
tam["cm_estimado"] = cross_validated_cm(tam.medida_cm, tam.L_px, tam.id_fruta)
tam["error_cm"] = tam.cm_estimado - tam.medida_cm             # > 0: sobreestima
print(f"k = {K:.5f} cm/px a {MAX_SIDE} px por el lado mayor")
resumen = tam.groupby("fruta").apply(lambda g: pd.Series({
    "fotos": len(g), "EAM (cm)": g.error_cm.abs().mean(), "sesgo medio (cm)": g.error_cm.mean(),
    "error máximo (cm)": g.error_cm.abs().max(), "error relativo medio (%)": 100 * (g.error_cm.abs() / g.medida_cm).mean()}),
    include_groups=False)
resumen.loc["todas"] = [len(tam), tam.error_cm.abs().mean(), tam.error_cm.mean(), tam.error_cm.abs().max(),
                        100 * (tam.error_cm.abs() / tam.medida_cm).mean()]
resumen.round(3)
"""),
        code("""
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for f in FRUTAS:
    s = tam[tam.fruta == f]
    ax.scatter(s.L_px, s.medida_cm, s=42, marker=MARCA[f], color=COLOR[f], edgecolor="white", linewidth=0.8, label=f, zorder=3)
x = np.linspace(0, tam.L_px.max() * 1.05, 50)
ax.plot(x, K * x, color="#555555", lw=1.5, ls="--", label=f"medida = {K:.4f} · L", zorder=2)
ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.set_xlabel(f"L (px, imagen a {MAX_SIDE} px)"); ax.set_ylabel("medida con regla (cm)")
ax.set_title("Escala píxel → centímetro (45 fotos propias)")
ax.legend(frameon=False)
plt.tight_layout(); savefig("tamano_escala"); plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, f in zip(axes, FRUTAS):
    s = tam[tam.fruta == f]
    lo, hi = min(s.medida_cm.min(), s.cm_estimado.min()) - 0.5, max(s.medida_cm.max(), s.cm_estimado.max()) + 0.5
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#e8e8e8", zorder=1, label="±1 cm")
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=1.2, zorder=2, label="estimado = real")
    ax.scatter(s.medida_cm, s.cm_estimado, s=42, marker=MARCA[f], color=COLOR[f], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(f"{f} · EAM = {s.error_cm.abs().mean():.2f} cm"); ax.set_xlabel("medida con regla (cm)")
axes[0].set_ylabel("estimado, validación cruzada (cm)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0))
plt.tight_layout(rect=(0, 0.07, 1, 1)); savefig("tamano_estimado_vs_real"); plt.show()
"""),
        md("### Fuentes del error: sesión de captura y celular"),
        md(r"""
Cada combinación fruta × calidad se fotografió en una sesión distinta. Un sesgo medio grande y del mismo signo en toda una sesión indica que la cámara no quedó exactamente a 30 cm: si quedó más cerca, la fruta ocupa más píxeles y su tamaño se **sobreestima**. Celular y sesión están confundidos (ver `docs/TRASPASO.md` §4.1), así que esta tabla describe el error, pero no separa sus causas.
"""),
        code("""
sesion = tam.groupby(["fruta", "calidad", "celular", "fecha"]).agg(
    fotos=("archivo", "size"), L_px_mediana=("L_px", "median"), medida_cm_media=("medida_cm", "mean"),
    sesgo_cm=("error_cm", "mean"), EAM_cm=("error_cm", lambda e: e.abs().mean())).round(3)
sesion
"""),
        md("## 3. Clases de tamaño por terciles"),
        md(r"""
Para cada fruta, los terciles $t_1, t_2$ de `medida_cm` definen la clase real (pequeño si $x < t_1$, mediano si $t_1 \le x < t_2$, grande si $x \ge t_2$). La clase estimada aplica los mismos umbrales a la estimación validada en cm. La exactitud se compara con la de asignar siempre la clase más frecuente.
"""),
        code("""
terciles = {f: fit_thresholds(tam.loc[tam.fruta == f, "medida_cm"]) for f in FRUTAS}
tam["clase_real"] = [size_class(v, terciles[f]) for v, f in zip(tam.medida_cm, tam.fruta)]
tam["clase_estimada"] = [size_class(v, terciles[f]) for v, f in zip(tam.cm_estimado, tam.fruta)]
orden = {c: i for i, c in enumerate(SIZE_CLASSES)}
tam["salto"] = (tam.clase_estimada.map(orden) - tam.clase_real.map(orden)).abs()

clases = tam.groupby("fruta").apply(lambda g: pd.Series({
    "t1 (cm)": terciles[g.name][0], "t2 (cm)": terciles[g.name][1], "rango medido (cm)": g.medida_cm.max() - g.medida_cm.min(),
    "exactitud": (g.clase_real == g.clase_estimada).mean(), "error de dos clases": (g.salto == 2).mean(),
    "clase más frecuente": g.clase_real.value_counts(normalize=True).iloc[0]}), include_groups=False)
clases.loc["todas"] = [np.nan, np.nan, np.nan, (tam.clase_real == tam.clase_estimada).mean(), (tam.salto == 2).mean(),
                       np.nan]
clases.round(3)
"""),
        code("""
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ConfusionMatrixDisplay(confusion_matrix(tam.clase_real, tam.clase_estimada, labels=SIZE_CLASSES),
                       display_labels=["pequeño", "mediano", "grande"]).plot(cmap="Blues", colorbar=False, ax=ax)
ax.grid(False); ax.set_xlabel("clase estimada"); ax.set_ylabel("clase real (regla)"); ax.set_title("Clase de tamaño (45 fotos)")
plt.tight_layout(); savefig("tamano_clases"); plt.show()
"""),
        md("## 4. Parámetros para la aplicación"),
        md(r"""
Se guardan en `models/umbrales.json`, sección `tamano`. La sección `calidad` (umbrales $t_1, t_2$ sobre $P(\text{mala} \mid x)$, tarea §4.3) se agrega en el mismo archivo sin borrar esta. La tabla por foto queda en `reports/tabla_tamano_fotos_propias.csv`.
"""),
        code("""
ruta = MODELS / "umbrales.json"
params = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
params["tamano"] = {
    "descripcion": "medida_cm = k * L_px; L_px = lado mayor de cv2.minAreaRect sobre la máscara de fruit_on_uniform_background",
    "max_side_px": MAX_SIDE,
    "k_cm_por_px": round(K, 6),
    "terciles_cm": {f: [round(t, 3) for t in terciles[f]] for f in FRUTAS},
    "clases": SIZE_CLASSES,
    "validacion": {"esquema": "dejar una fruta fuera (id_fruta)", "fotos": int(len(tam)),
                   "eam_cm": {f: round(float(tam.loc[tam.fruta == f, "error_cm"].abs().mean()), 3) for f in FRUTAS},
                   "exactitud_clase": round(float((tam.clase_real == tam.clase_estimada).mean()), 3)},
}
ruta.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
cols = ["archivo", "id_fruta", "fruta", "calidad", "celular", "fecha", "medida_cm", "L_px", "d_eq_px", "feret_px",
        "cm_estimado", "error_cm", "clase_real", "clase_estimada"]
tam[cols].round(3).to_csv(ROOT / "reports" / "tabla_tamano_fotos_propias.csv", index=False)
print(json.dumps(params["tamano"], indent=2, ensure_ascii=False))
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "fruitq", "display_name": "Python (fruitq)", "language": "python"}
    return nb


NOTEBOOKS = {
    "eda": ("02_eda.ipynb", eda_notebook),
    "clasicos": ("04_modelos_clasicos.ipynb", modelos_clasicos_notebook),
    "cnn": ("05_cnn_y_comparacion.ipynb", cnn_notebook),
    "tamano": ("06_tamano.ipynb", tamano_notebook),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nb", nargs="*", choices=list(NOTEBOOKS), default=list(NOTEBOOKS),
                    help="Notebooks a generar (por defecto, todos)")
    ap.add_argument("--run", action="store_true", help="Ejecuta el notebook y guarda las salidas")
    args = ap.parse_args()

    NB_DIR.mkdir(exist_ok=True)
    for key in args.nb:
        fname, builder = NOTEBOOKS[key]
        nb = builder()
        if args.run:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=7200, kernel_name="fruitq",
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbf.write(nb, NB_DIR / fname)
        print(f"Notebook escrito en {NB_DIR / fname}")


if __name__ == "__main__":
    main()
