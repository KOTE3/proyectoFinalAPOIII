"""Estimación del tamaño relativo de la fruta.

Diámetro equivalente: el de un círculo con la misma área que la máscara,

    d = 2 * sqrt(A / pi).

Para que no dependa de la resolución se normaliza por el lado menor de la
imagen: d_norm = d / min(H, W). Con la cámara a distancia fija, d_norm es
proporcional al tamaño real. Las categorías se obtienen con umbrales (t1, t2)
calculados como terciles de d_norm **por tipo de fruta**, porque un limón
grande sigue siendo más pequeño que una manzana pequeña.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .config import SIZE_CLASSES


def equivalent_diameter(area_px: float) -> float:
    return 2.0 * math.sqrt(area_px / math.pi)


def normalized_diameter(area_px: float, image_shape: tuple[int, ...]) -> float:
    h, w = image_shape[:2]
    return equivalent_diameter(area_px) / min(h, w)


def fit_thresholds(d_norm: Iterable[float]) -> tuple[float, float]:
    """Terciles (percentiles 33.3 y 66.7) de los diámetros normalizados."""
    arr = np.asarray(list(d_norm), dtype=float)
    if arr.size < 3:
        raise ValueError("Se necesitan al menos 3 muestras para calcular terciles.")
    t1, t2 = np.percentile(arr, [100 / 3, 200 / 3])
    return float(t1), float(t2)


def size_class(d_norm: float, thresholds: tuple[float, float]) -> str:
    t1, t2 = thresholds
    if d_norm < t1:
        return SIZE_CLASSES[0]
    if d_norm < t2:
        return SIZE_CLASSES[1]
    return SIZE_CLASSES[2]


def pixels_to_cm(d_px: float, ref_diameter_px: float, ref_diameter_cm: float) -> float:
    """Convierte a cm usando un objeto de referencia (p. ej. una moneda) en la misma foto."""
    return d_px * ref_diameter_cm / ref_diameter_px
