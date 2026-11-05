"""Fase 3: limpieza del dataset de Kaggle y partición train/val/test por grupos.

Pasos (justificación en docs/fases/03_preparacion_datos.md):
  1. Indexa data/raw/kaggle (buena / mala; las imágenes 'mixed' se usan aparte
     a través de los recortes segmentados y revisados).
  2. Calcula aHash, MD5 y bandera de gris de cada imagen (con caché).
  3. Elimina las imágenes en escala de grises.
  4. Agrupa casi duplicados: mismo aHash o misma ráfaga de fotos (hora en el nombre).
  5. Elimina los grupos con etiquetas contradictorias (misma foto marcada buena y mala).
  6. Partición estratificada por (fruta, calidad) y agrupada: 70 / 15 / 15.

Salida: data/processed/manifest.csv  (una fila por imagen utilizable)

Uso:
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitq.config import PROCESSED, RAW_KAGGLE  # noqa: E402
from fruitq.data import burst_ids, check_no_leakage, connected_groups, group_split, image_signature  # noqa: E402
from fruitq.io import index_folder  # noqa: E402

CALIDAD = {"Good Quality_Fruits": "buena", "Bad Quality_Fruits": "mala"}
FRUTA = {"Apple": "manzana", "Banana": "banano", "Guava": "guayaba",
         "Lime": "limon", "Orange": "naranja", "Pomegranate": "granada"}


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = index_folder(RAW_KAGGLE).rename(columns={"level_0": "carpeta", "level_1": "subcarpeta"})
    df = df[df.carpeta.isin(CALIDAD)].copy()
    df["calidad"] = df.carpeta.map(CALIDAD)
    df["fruta"] = df.subcarpeta.str.replace(r"_(Good|Bad)$", "", regex=True).map(FRUTA)
    df["fuente"] = "kaggle"
    log = [("Imágenes buena/mala en Kaggle", len(df))]

    cache = PROCESSED / "firmas_kaggle.csv"
    if cache.exists():
        sig = pd.read_csv(cache, dtype={"ahash": str})
        sig["ahash"] = sig["ahash"].apply(int)
    else:
        sig = pd.DataFrame([image_signature(p) for p in tqdm(df.path, desc="Firmas")])
        sig.insert(0, "path", df.path.values)
        sig["ahash"] = sig["ahash"].astype(str)  # 64 bits sin signo no caben en int64
        sig.to_csv(cache, index=False)
        sig["ahash"] = sig["ahash"].apply(int)
    df = df.merge(sig, on="path")

    df = df[~df.gris].copy()
    log.append(("Tras eliminar imágenes en gris", len(df)))

    df = df.drop_duplicates("md5")
    log.append(("Tras eliminar duplicados exactos (MD5)", len(df)))

    df = df.reset_index(drop=True)
    df["rafaga"] = burst_ids(df["path"])
    # Hash dentro de la misma carpeta; se ignoran hashes "genéricos" (> 3 imágenes)
    clave = df["subcarpeta"] + "|" + df["ahash"].astype(str)
    clave_id = np.where(clave.map(clave.value_counts()) <= 3, pd.factorize(clave)[0], -1)
    df["grupo"] = connected_groups(clave_id, df["rafaga"].values)
    conflictos = df.groupby("grupo")["calidad"].nunique()
    malos = conflictos[conflictos > 1].index
    df = df[~df.grupo.isin(malos)].copy()
    log.append((f"Tras eliminar {len(malos)} grupos con etiqueta contradictoria", len(df)))

    df["estrato"] = df.fruta + "_" + df.calidad
    df["split"] = group_split(df, strat_col="estrato")
    check_no_leakage(df)

    cols = ["path", "fuente", "fruta", "calidad", "estrato", "grupo", "split"]
    df[cols].to_csv(PROCESSED / "manifest.csv", index=False)

    print("\n== Limpieza ==")
    for desc, n in log:
        print(f"  {desc:<55} {n:>6}")
    tam = df.groupby("grupo").size()
    print(f"\n  Grupos: {tam.size}  | imágenes por grupo: media {tam.mean():.2f}, máx {tam.max()}")
    print(f"  Grupos con >1 imagen: {(tam > 1).sum()} ({(tam > 1).mean():.1%})")

    print("\n== Partición (imágenes) ==")
    tabla = pd.crosstab(df.estrato, df.split)[["train", "val", "test"]]
    tabla.loc["TOTAL"] = tabla.sum()
    tabla["% test"] = (100 * tabla["test"] / tabla.sum(axis=1)).round(1)
    print(tabla.to_string())
    print("\nSin fuga de información entre subconjuntos: OK")
    print(f"Manifest: {PROCESSED / 'manifest.csv'}")


if __name__ == "__main__":
    main()
