"""Descarga el dataset Fruit Quality Classification de Kaggle en data/raw/kaggle.

Requisitos:
    1. pip install kaggle  (versión >= 1.8, que acepta los tokens nuevos KGAT_...)
    2. Token de API (Kaggle -> Settings -> API -> Create New Token) guardado en
       ~/.kaggle/access_token, o en la variable de entorno KAGGLE_API_TOKEN.
       También sirve el formato antiguo ~/.kaggle/kaggle.json.
       NUNCA subir el token a Git.

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitq.config import KAGGLE_DATASET, RAW_KAGGLE  # noqa: E402
from fruitq.io import index_folder  # noqa: E402


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except (OSError, SystemExit) as e:
        sys.exit(
            f"No se pudo autenticar con Kaggle: {e}\n"
            "Crea un token en https://www.kaggle.com/settings (API -> Create New Token) "
            f"y guárdalo en {Path.home() / '.kaggle' / 'access_token'}."
        )
    RAW_KAGGLE.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {KAGGLE_DATASET} en {RAW_KAGGLE} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_KAGGLE), unzip=True, quiet=False)

    df = index_folder(RAW_KAGGLE)
    print(f"\n{len(df)} imágenes descargadas.")
    level_cols = [c for c in df.columns if c.startswith("level_")]
    if level_cols:
        print("\nConteo por carpeta (primeros dos niveles):")
        print(df.groupby(level_cols[:2]).size().to_string())


if __name__ == "__main__":
    main()
