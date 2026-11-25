"""Clase de calidad en tres niveles y preparación de imágenes nuevas.

Los modelos se entrenaron con dos clases (buena / mala). La clase ``regular``
se obtiene con dos umbrales sobre un puntaje ``p`` que crece con el deterioro
(por ejemplo, P(mala | x) de XGBoost, de la CNN o su promedio):

    buena    si p < t1
    regular  si t1 <= p < t2
    mala     si p >= t2

Los umbrales se eligen por búsqueda en rejilla, maximizando el F1-macro de
tres clases sobre los recortes etiquetados por el grupo (``fit_three_thresholds``).

Cualquier transformación monótona del puntaje (por ejemplo, una calibración de
Platt) conserva el orden de las imágenes y, por tanto, las particiones posibles:
calibrar cambia la lectura de la probabilidad, no el F1 alcanzable con dos umbrales.

``kaggle_view`` lleva una foto nueva a la escala de las imágenes de entrenamiento:
un cuadrado centrado en la fruta, con la fruta ocupando la misma fracción del
lado que la mediana de Kaggle, reducido a 256 px.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .config import QUALITY_CLASSES
from .size import equivalent_diameter

KAGGLE_SIDE = 256       # lado mayor de las imágenes con que se extrajeron las características
KAGGLE_D_NORM = 0.436   # mediana de d / min(H, W) en el train de Kaggle


def three_class(p: Iterable[float], t1: float, t2: float) -> np.ndarray:
    """Clase de calidad (``buena``, ``regular``, ``mala``) a partir del puntaje ``p``."""
    if not t1 <= t2:
        raise ValueError("Se requiere t1 <= t2.")
    p = np.asarray(p, dtype=float)
    idx = (p >= t1).astype(int) + (p >= t2).astype(int)
    return np.asarray(QUALITY_CLASSES, dtype=object)[idx]


def fit_three_thresholds(
    p: Iterable[float], y: Iterable[str], n_grid: int = 101
) -> tuple[float, float, float]:
    """Umbrales (t1, t2) que maximizan el F1-macro de tres clases.

    Los candidatos son los cuantiles del propio puntaje, así la rejilla tiene
    resolución donde hay datos, aunque el modelo concentre las probabilidades
    cerca de 0 y de 1. Devuelve (t1, t2, F1-macro alcanzado). En caso de empate
    se conserva el primer par encontrado (el de umbrales más bajos).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=object)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("p e y deben tener la misma longitud y no estar vacíos.")
    pos = {c: i for i, c in enumerate(QUALITY_CLASSES)}
    try:
        y_idx = np.array([pos[v] for v in y], dtype=int)
    except KeyError as e:
        raise ValueError(f"Etiqueta desconocida {e}; se esperaba una de {QUALITY_CLASSES}.") from None
    grid = np.unique(np.quantile(p, np.linspace(0, 1, n_grid)))
    grid = np.append(grid, np.nextafter(p.max(), np.inf))  # permite clases vacías arriba
    best = (float(grid[0]), float(grid[-1]), -1.0)
    for i, t1 in enumerate(grid):
        for t2 in grid[i:]:
            pred = (p >= t1).astype(int) + (p >= t2).astype(int)
            f1 = _macro_f1(y_idx, pred)
            if f1 > best[2] + 1e-12:
                best = (float(t1), float(t2), float(f1))
    return best


def _macro_f1(y_idx: np.ndarray, pred_idx: np.ndarray, k: int = 3) -> float:
    """F1-macro con clases 0..k-1 (equivale a sklearn con ``zero_division=0``)."""
    cm = np.bincount(k * y_idx + pred_idx, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def logit(p: Iterable[float], eps: float = 1e-6) -> np.ndarray:
    """log(p / (1 - p)), recortando p a [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def kaggle_view(
    img_bgr: np.ndarray,
    contour: np.ndarray,
    d_norm: float = KAGGLE_D_NORM,
    side: int = KAGGLE_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorte de la fruta a la escala de entrenamiento.

    Toma un cuadrado centrado en la fruta de lado ``d / d_norm``, donde ``d`` es
    el diámetro equivalente del contorno; si el cuadrado sale de la imagen, se
    completa replicando el borde. Luego lo reduce a ``side`` px. Devuelve el
    recorte BGR de la fruta y su máscara (0/255), del mismo tamaño, listos para
    ``extract_features`` o ``masked_tile``.
    """
    h, w = img_bgr.shape[:2]
    full = np.zeros((h, w), np.uint8)
    cv2.drawContours(full, [contour], -1, 255, thickness=cv2.FILLED)
    d = equivalent_diameter(float((full > 0).sum()))
    half = int(round(d / d_norm / 2))
    x, y, bw, bh = cv2.boundingRect(contour)
    cx, cy = x + bw // 2, y + bh // 2
    pad = max(0, half - cx, half - cy, cx + half - w, cy + half - h)
    img_p = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    msk_p = cv2.copyMakeBorder(full, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y0, x0 = cy + pad - half, cx + pad - half
    sq_img = img_p[y0:y0 + 2 * half, x0:x0 + 2 * half]
    sq_msk = msk_p[y0:y0 + 2 * half, x0:x0 + 2 * half]
    interp = cv2.INTER_AREA if 2 * half > side else cv2.INTER_CUBIC
    sq_img = cv2.resize(sq_img, (side, side), interpolation=interp)
    sq_msk = cv2.resize(sq_msk, (side, side), interpolation=cv2.INTER_NEAREST)
    xs, ys, ws, hs = cv2.boundingRect(sq_msk)
    return sq_img[ys:ys + hs, xs:xs + ws].copy(), sq_msk[ys:ys + hs, xs:xs + ws].copy()


def masked_tile(crop_bgr: np.ndarray, mask: np.ndarray, size: int = 128) -> np.ndarray:
    """Entrada de la CNN: fruta con el fondo en negro, centrada en un cuadrado de ``size`` px (RGB)."""
    crop = crop_bgr.copy()
    crop[mask == 0] = 0
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    tile = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
