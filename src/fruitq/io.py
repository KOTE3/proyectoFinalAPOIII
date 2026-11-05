"""Utilidades de lectura y escritura de imágenes e indexación del dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_EXTS


def read_image(path: str | Path) -> np.ndarray:
    """Lee una imagen en BGR. Soporta rutas con tildes o espacios en Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"No se pudo leer la imagen: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"No se pudo codificar la imagen: {path}")
    buf.tofile(str(path))


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def index_folder(root: str | Path) -> pd.DataFrame:
    """Tabla con una fila por imagen y cada carpeta de su ruta como columna.

    Útil para el EDA cuando las etiquetas vienen de la estructura de carpetas
    (p. ej. ``Good Quality_Fruits/Apple/xxx.jpg`` -> level_0, level_1).
    """
    root = Path(root)
    rows = []
    for p in list_images(root):
        parts = p.relative_to(root).parts[:-1]
        row = {"path": str(p), "file": p.name}
        row.update({f"level_{i}": part for i, part in enumerate(parts)})
        rows.append(row)
    return pd.DataFrame(rows)
