"""Extracción manual de características para los modelos clásicos (SVM, RF, ...).

Cada fruta segmentada se resume en un vector de dimensión fija:

- Color: histograma HSV normalizado (H: 16 bins, S: 8, V: 8) solo sobre la fruta.
- Momentos de color: media, desviación y asimetría por canal en HSV y Lab.
- Forma: circularidad 4*pi*A/P^2, solidez A/A_convexa, relación de aspecto.
- Textura/nitidez: varianza del Laplaciano y energía media del gradiente (Sobel).
- Defectos: fracción de píxeles oscuros (V bajo) y de baja saturación dentro
  de la fruta, como indicadores de manchas o podredumbre.
"""

from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 16, 8, 8


def feature_names() -> list[str]:
    names = [f"h_hist_{i}" for i in range(H_BINS)]
    names += [f"s_hist_{i}" for i in range(S_BINS)]
    names += [f"v_hist_{i}" for i in range(V_BINS)]
    for space in ("hsv", "lab"):
        for c in space:
            names += [f"{space}_{c}_mean", f"{space}_{c}_std", f"{space}_{c}_skew"]
    names += ["circularity", "solidity", "aspect_ratio"]
    names += ["laplacian_var", "gradient_energy"]
    names += ["dark_frac", "lowsat_frac"]
    return names


def extract_features(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vector de características de una fruta (recorte BGR + máscara 0/255)."""
    m = mask > 0
    if m.sum() == 0:
        raise ValueError("La máscara está vacía.")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask_u8 = m.astype(np.uint8) * 255

    feats: list[float] = []

    # Histogramas de color, normalizados a suma 1 (invariantes al tamaño de la fruta)
    for ch, bins, rng in ((0, H_BINS, 180), (1, S_BINS, 256), (2, V_BINS, 256)):
        hist = cv2.calcHist([hsv], [ch], mask_u8, [bins], [0, rng]).ravel()
        feats.extend(hist / (hist.sum() + 1e-9))

    # Momentos de color: media, desviación y asimetría (skewness)
    for img in (hsv, lab):
        for c in range(3):
            vals = img[:, :, c][m].astype(float)
            mu, sd = vals.mean(), vals.std()
            skew = ((vals - mu) ** 3).mean() / (sd**3 + 1e-9)
            feats.extend([mu, sd, skew])

    # Forma
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    _, (w, h), _ = cv2.minAreaRect(c)
    feats.append(4 * np.pi * area / (perim**2 + 1e-9))
    feats.append(area / (hull_area + 1e-9))
    feats.append(max(w, h) / (min(w, h) + 1e-9))

    # Textura / nitidez (se erosiona la máscara para no medir el borde fruta-fondo)
    inner = cv2.erode(mask_u8, np.ones((5, 5), np.uint8)) > 0
    if inner.sum() == 0:
        inner = m
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    feats.append(lap[inner].var())
    feats.append(np.sqrt(gx[inner] ** 2 + gy[inner] ** 2).mean())

    # Indicadores de defectos
    v = hsv[:, :, 2][m]
    s = hsv[:, :, 1][m]
    feats.append(float((v < 60).mean()))
    feats.append(float((s < 40).mean()))

    return np.asarray(feats, dtype=np.float32)
