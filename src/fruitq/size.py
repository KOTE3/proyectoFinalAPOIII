"""Estimación del tamaño de la fruta.

Dos medidas de longitud en píxeles a partir de la máscara:

- **Diámetro equivalente**, el de un círculo con la misma área que la máscara,

      d = 2 * sqrt(A / pi).

  Normalizado por el lado menor de la imagen, d_norm = d / min(H, W), se usa
  en el dataset de Kaggle, donde la distancia de la cámara no está controlada.

- **Longitud característica**, el lado mayor del rectángulo de área mínima que
  contiene el contorno (``cv2.minAreaRect``). Es la magnitud que se midió con
  regla en las fotos propias: el diámetro mayor en manzana y guayaba y el largo
  de punta a punta en banano.

En las fotos propias la cámara estuvo a distancia fija (30 cm), así que la
longitud en centímetros es proporcional a la longitud en píxeles,
medida_cm ≈ k · L_px. El factor de escala se estima como la mediana de los
cocientes, que es robusta a fotos tomadas a una altura algo distinta:

    k = mediana(medida_cm_i / L_px_i).

Las clases pequeño / mediano / grande se definen con terciles calculados
**por tipo de fruta**, porque un banano pequeño sigue siendo más largo que
una manzana grande.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from .config import SIZE_CLASSES


def equivalent_diameter(area_px: float) -> float:
    return 2.0 * math.sqrt(area_px / math.pi)


def normalized_diameter(area_px: float, image_shape: tuple[int, ...]) -> float:
    h, w = image_shape[:2]
    return equivalent_diameter(area_px) / min(h, w)


def characteristic_length(contour: np.ndarray) -> float:
    """Lado mayor (en píxeles) del rectángulo de área mínima que contiene el contorno."""
    _, (w, h), _ = cv2.minAreaRect(contour)
    return float(max(w, h))


def fit_scale(length_cm: Iterable[float], length_px: Iterable[float]) -> float:
    """Factor de escala k (cm/px): mediana de los cocientes medida_cm / L_px."""
    cm = np.asarray(list(length_cm), dtype=float)
    px = np.asarray(list(length_px), dtype=float)
    if cm.shape != px.shape or cm.size == 0:
        raise ValueError("Se necesitan listas no vacías y de igual longitud.")
    if np.any(px <= 0):
        raise ValueError("Las longitudes en píxeles deben ser positivas.")
    return float(np.median(cm / px))


def to_cm(length_px: float, k: float) -> float:
    """Convierte una longitud en píxeles a centímetros con el factor k (cm/px)."""
    return length_px * k


def cross_validated_cm(
    length_cm: Iterable[float], length_px: Iterable[float], groups: Iterable
) -> np.ndarray:
    """Estimación en cm de cada foto con k ajustado **sin** la fruta de esa foto.

    Validación cruzada dejando un grupo fuera: para cada grupo (``id_fruta``),
    k se calcula con las fotos de los demás grupos y se aplica a las del grupo
    excluido. Así ninguna foto se evalúa con un k que la usó para ajustarse.
    """
    cm = np.asarray(list(length_cm), dtype=float)
    px = np.asarray(list(length_px), dtype=float)
    g = np.asarray(list(groups))
    if not (cm.shape == px.shape == g.shape):
        raise ValueError("length_cm, length_px y groups deben tener la misma longitud.")
    if np.unique(g).size < 2:
        raise ValueError("Se necesitan al menos dos grupos.")
    pred = np.empty_like(cm)
    for grupo in np.unique(g):
        fuera = g == grupo
        pred[fuera] = px[fuera] * fit_scale(cm[~fuera], px[~fuera])
    return pred


def fit_thresholds(values: Iterable[float]) -> tuple[float, float]:
    """Terciles (percentiles 33.3 y 66.7) de una medida de tamaño."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 3:
        raise ValueError("Se necesitan al menos 3 muestras para calcular terciles.")
    t1, t2 = np.percentile(arr, [100 / 3, 200 / 3])
    return float(t1), float(t2)


def size_class(value: float, thresholds: tuple[float, float]) -> str:
    t1, t2 = thresholds
    if value < t1:
        return SIZE_CLASSES[0]
    if value < t2:
        return SIZE_CLASSES[1]
    return SIZE_CLASSES[2]
