"""Utilidades de lectura y escritura de imágenes e indexación del dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_EXTS, ROOT


def project_path(path: str | Path) -> str:
    """Ruta relativa a la raíz del proyecto, con ``/`` como separador.

    Los CSV y parquet guardan las rutas así para que funcionen en cualquier
    computador y sistema operativo, sin depender de dónde se clonó el repositorio.
    """
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(ROOT.resolve())
        except ValueError:
            return p.as_posix()  # fuera del proyecto: se conserva absoluta
    return p.as_posix()


def resolve_path(path: str | Path) -> Path:
    """Ruta absoluta a partir de una ruta guardada con ``project_path``."""
    p = Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else ROOT / p


def read_image(path: str | Path) -> np.ndarray:
    """Lee una imagen en BGR. Soporta rutas con tildes o espacios en Windows."""
    data = np.fromfile(str(resolve_path(path)), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"No se pudo leer la imagen: {path}")
    return img


def read_image_max(path: str | Path, max_side: int = 256) -> np.ndarray:
    """Lee una imagen y la reduce para que su lado mayor sea ``max_side``.

    El dataset mezcla fotos de 256 px con fotos de 12-48 MP. Trabajar todas a la
    misma escala hace comparables las características (p. ej. la varianza del
    Laplaciano depende de la resolución) y acelera la segmentación.
    """
    return resize_max(read_image(path), max_side)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Reduce una imagen ya cargada para que su lado mayor sea ``max_side`` (nunca la amplía)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def decode_image(data: bytes) -> np.ndarray:
    """Decodifica una imagen recibida como bytes (p. ej. desde la aplicación) en BGR."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("No se pudo decodificar la imagen.")
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
        row = {"path": project_path(p), "file": p.name}
        row.update({f"level_{i}": part for i, part in enumerate(parts)})
        rows.append(row)
    return pd.DataFrame(rows)
