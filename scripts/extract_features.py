"""Extrae las características manuales de todas las imágenes del manifest.

Para cada imagen:
  1. Se reduce a 256 px de lado mayor (misma escala para todas).
  2. Se segmenta y se toma la fruta de mayor área. Si no se detecta ninguna,
     se usa la imagen completa y se marca seg_ok = False.
  3. Se calcula el vector de 57 características (fruitq.features) y el
     diámetro normalizado.

Se ejecuta en paralelo (joblib) y guarda data/processed/features.parquet.

Uso:
    python scripts/extract_features.py            # todos los núcleos
    python scripts/extract_features.py --n-jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitq.config import PROCESSED  # noqa: E402
from fruitq.features import extract_features, feature_names  # noqa: E402
from fruitq.io import read_image_max  # noqa: E402
from fruitq.segmentation import find_fruits  # noqa: E402
from fruitq.size import normalized_diameter  # noqa: E402

NAMES = feature_names()


def procesar(path: str) -> dict:
    try:
        img = read_image_max(path, 256)
        regs = find_fruits(img)
        if regs:
            r = regs[0]
            crop, mask, area, ok = r.crop, r.mask, r.area, True
        else:
            crop = img
            mask = np.full(img.shape[:2], 255, np.uint8)
            area, ok = float(mask.size), False
        f = extract_features(crop, mask)
        out = dict(zip(NAMES, f.tolist()))
        out.update(path=path, seg_ok=ok, n_frutas=len(regs),
                   d_norm=normalized_diameter(area, img.shape), error="")
    except Exception as e:  # una imagen corrupta no debe tumbar todo el proceso
        out = {n: np.nan for n in NAMES}
        out.update(path=path, seg_ok=False, n_frutas=0, d_norm=np.nan, error=repr(e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    t0 = time.time()
    filas = Parallel(n_jobs=args.n_jobs, batch_size=32, verbose=5)(
        delayed(procesar)(p) for p in manifest.path
    )
    feats = pd.DataFrame(filas)
    df = manifest.merge(feats, on="path")
    out = PROCESSED / "features.parquet"
    df.to_parquet(out, index=False)

    dt = time.time() - t0
    print(f"\n{len(df)} imágenes en {dt / 60:.1f} min ({1000 * dt / len(df):.0f} ms/imagen con {args.n_jobs} jobs)")
    print(f"Segmentación sin fruta detectada: {(~df.seg_ok).sum()} ({(~df.seg_ok).mean():.1%})")
    print(f"Errores: {(df.error != '').sum()}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
