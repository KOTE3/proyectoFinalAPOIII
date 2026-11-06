"""Precalcula los recortes 128x128 enmascarados de todas las imágenes del manifest.

Segmentar en cada época de entrenamiento sería muy lento (~0.3 s/imagen), así
que se hace una sola vez y se guarda:
    data/processed/tiles128.npy        (N, 128, 128, 3) uint8, mismo orden que el manifest
    data/processed/tiles128_index.csv  path, seg_ok

Uso:
    python scripts/cache_tiles.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitq.cnn import TILE, fruit_tile  # noqa: E402
from fruitq.config import PROCESSED  # noqa: E402
from fruitq.io import read_image_max  # noqa: E402


def procesar(path: str):
    try:
        tile, ok = fruit_tile(read_image_max(path, 256), TILE)
        return tile, ok
    except Exception:
        return np.zeros((TILE, TILE, 3), np.uint8), False


def main() -> None:
    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    t0 = time.time()
    res = Parallel(n_jobs=-1, batch_size=32, verbose=5)(delayed(procesar)(p) for p in manifest.path)
    tiles = np.stack([r[0] for r in res])
    np.save(PROCESSED / f"tiles{TILE}.npy", tiles)
    pd.DataFrame({"path": manifest.path, "seg_ok": [r[1] for r in res]}).to_csv(
        PROCESSED / f"tiles{TILE}_index.csv", index=False)
    print(f"{len(tiles)} recortes en {(time.time() - t0) / 60:.1f} min -> {tiles.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
