"""Limpieza del dataset, agrupación de casi duplicados y partición por grupos.

¿Por qué partir por grupos?
---------------------------
El EDA mostró que ~30 % del dataset son casi duplicados (ráfagas de fotos y
copias en gris). Si x_i va a train y su copia x_j a test, el modelo "ve" el
test durante el entrenamiento y la métrica estimada E_test deja de ser un
estimador insesgado del error real (fuga de información / data leakage).
Por eso todas las imágenes de un mismo grupo van al mismo subconjunto.

Detección de casi duplicados
----------------------------
Se construye un grafo cuyos nodos son las imágenes y se unen dos imágenes si:

(a) tienen el mismo average-hash (aHash: imagen reducida a 8x8 en gris,
    cada píxel comparado con la media -> 64 bits), o
(b) son parte de la misma **ráfaga**: misma carpeta y hora de captura (leída
    del nombre del archivo, p. ej. IMG_20190910_175712) a <= 1 s de la anterior.

En (a) solo se usa el hash dentro de la misma carpeta y si lo comparten como
máximo 3 imágenes: un hash de 8x8 repetido muchas veces corresponde a una
composición genérica (fruta centrada sobre fondo claro), no a una copia.

Los grupos son las componentes conexas del grafo.

Se descartó usar distancia de Hamming d_H(a,b) = popcount(a XOR b) <= tau
con union-find: por transitividad (A~B, B~C, ...) se forman cadenas que con
tau = 1 ya unían más de 600 fotos distintas en un solo grupo, y con tau = 4
más de 8000. Ver docs/fases/03_preparacion_datos.md.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import SEED


# --------------------------------------------------------------------------
# Hash y detección de imágenes en gris
# --------------------------------------------------------------------------

def image_signature(path: str) -> dict:
    """aHash de 64 bits, MD5 del archivo y bandera de escala de grises."""
    raw = np.fromfile(path, dtype=np.uint8)
    # IMREAD_REDUCED_COLOR_4 decodifica a 1/4 de resolución: mucho más rápido
    # con las fotos de 12-48 MP y suficiente para un hash de 8x8.
    img = cv2.imdecode(raw, cv2.IMREAD_REDUCED_COLOR_4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).ravel()
    ahash = int("".join("1" if b else "0" for b in bits), 2)
    i = img.astype(np.int16)
    is_gray = bool(np.abs(i[:, :, 0] - i[:, :, 1]).mean() < 2 and np.abs(i[:, :, 1] - i[:, :, 2]).mean() < 2)
    return {"md5": hashlib.md5(raw.tobytes()).hexdigest(), "ahash": ahash, "gris": is_gray}


# --------------------------------------------------------------------------
# Agrupación de casi duplicados
# --------------------------------------------------------------------------

_TS = re.compile(r"(\d{8})_?(\d{6})")
_SEQ = re.compile(r"^IMG_(\d{3,5})$", re.IGNORECASE)


def burst_ids(paths: pd.Series, max_gap_s: float = 1.0, max_seq_gap: int | None = None) -> np.ndarray:
    """Id de ráfaga por imagen a partir del nombre del archivo (-1 = sin información).

    ``max_seq_gap`` activa además la regla de números de secuencia (IMG_8134,
    IMG_8135). Está desactivada por defecto: en este dataset encadena sesiones
    completas (un solo grupo de 521 imágenes de banano bueno).
    """
    p = pd.Series(paths).reset_index(drop=True)
    folder = p.map(lambda s: str(Path(s).parent))
    stem = p.map(lambda s: Path(s).stem)
    out = np.full(len(p), -1)
    next_id = 0

    ts_parts = stem.str.extract(_TS)
    ts = pd.to_datetime(ts_parts[0] + ts_parts[1], format="%Y%m%d%H%M%S", errors="coerce")
    seq = pd.to_numeric(stem.str.extract(_SEQ)[0], errors="coerce")

    reglas = [(ts, max_gap_s)] + ([(seq, max_seq_gap)] if max_seq_gap is not None else [])
    for key, gap in reglas:
        d = pd.DataFrame({"folder": folder, "k": key}).dropna().sort_values(["folder", "k"])
        if d.empty:
            continue
        diff = d["k"].diff()
        if np.issubdtype(diff.dtype, np.timedelta64):
            diff = diff.dt.total_seconds()
        new = (d["folder"] != d["folder"].shift()) | (diff > gap)
        ids = new.cumsum().to_numpy() - 1 + next_id
        out[d.index.to_numpy()] = ids
        next_id = int(ids.max()) + 1
    return out


def connected_groups(*keys: np.ndarray) -> np.ndarray:
    """Componentes conexas del grafo en que dos imágenes se unen si comparten
    algún valor en cualquiera de las claves (valores negativos = sin clave)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(keys[0])
    rows, cols, offset = [], [], n
    for k in keys:
        k = np.asarray(k)
        _, inv = np.unique(k, return_inverse=True)
        valid = np.array([not (isinstance(v, (int, np.integer)) and v < 0) for v in k])
        rows.append(np.nonzero(valid)[0])
        cols.append(inv[valid] + offset)  # nodo auxiliar por cada valor de la clave
        offset += inv.max() + 1
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(offset, offset))
    _, labels = connected_components(g, directed=False)
    _, group = np.unique(labels[:n], return_inverse=True)
    return group


def group_near_duplicates(hashes: np.ndarray, max_hamming: int = 0) -> np.ndarray:
    """Grupos por distancia de Hamming entre aHash (union-find).

    Se deja para análisis; con max_hamming > 0 produce cadenas gigantes en este
    dataset (ver el docstring del módulo). La partición usa ``connected_groups``.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n - 1):
        d = np.bitwise_count(h[i + 1:] ^ h[i])
        for j in np.nonzero(d <= max_hamming)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


# --------------------------------------------------------------------------
# Partición train / val / test por grupos
# --------------------------------------------------------------------------

def group_split(
    df: pd.DataFrame,
    strat_col: str,
    group_col: str = "grupo",
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> pd.Series:
    """Asigna 'train' / 'val' / 'test' respetando grupos y estratificando.

    Usa StratifiedGroupKFold: con k = round(1 / test_frac) pliegues, un
    pliegue es test; del resto se separa val de la misma forma. Así se
    mantiene aproximadamente la proporción de clases en los tres conjuntos.
    """
    split = pd.Series("train", index=df.index)

    k_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df[strat_col], df[group_col]))
    split.iloc[test_idx] = "test"

    rest = df[split == "train"]
    k_val = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    _, val_idx = next(sgkf.split(rest, rest[strat_col], rest[group_col]))
    split.loc[rest.index[val_idx]] = "val"
    return split


def check_no_leakage(df: pd.DataFrame, group_col: str = "grupo", split_col: str = "split") -> None:
    """Lanza un error si algún grupo aparece en más de un subconjunto."""
    n_splits = df.groupby(group_col)[split_col].nunique()
    bad = n_splits[n_splits > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} grupos aparecen en más de un subconjunto")
