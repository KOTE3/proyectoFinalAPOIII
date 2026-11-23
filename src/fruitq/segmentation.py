"""Segmentación fruta/fondo con OpenCV.

El dataset de Kaggle NO cumple siempre el supuesto de "fondo uniforme": hay
pasto, baldosas, cartulinas de colores y varias frutas que se tocan. Por eso
la segmentación se hace en tres etapas:

1. **Modelo de color del fondo a partir del borde.** Supuesto: la franja del
   borde de la imagen es casi toda fondo. Se agrupan los píxeles del borde en
   el espacio CIELab con k-medias (k = 3) -> centros c_1..c_k. Para cada píxel
   x se calcula su distancia al fondo

       D(x) = min_j || Lab(x) - c_j ||_2 .

   Lab es aproximadamente uniforme perceptualmente, así que la distancia
   euclidiana aproxima la diferencia de color que percibe una persona. El
   umbral sobre D se elige con Otsu, que maximiza la varianza entre clases

       sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2 .

2. **Refinamiento con GrabCut** (Rother et al., 2004): modela fruta y fondo
   con mezclas de gaussianas (GMM) y minimiza una energía con corte de grafo,
   lo que corrige los bordes irregulares que deja el umbral (p. ej. en pasto).

3. **Separación de frutas que se tocan con watershed** sobre la transformada
   de distancia: cada máximo local de la distancia al borde de la máscara se
   toma como el "centro" de una fruta.

Las fotos propias (cartulina blanca, una fruta por foto) usan
``fruit_on_uniform_background``, que decide por croma respecto al fondo y así
deja fuera la sombra de la fruta y la moneda de referencia.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FruitRegion:
    """Una fruta detectada en la imagen."""

    mask: np.ndarray                 # máscara binaria (uint8, 0/255) del tamaño del recorte
    crop: np.ndarray                 # recorte BGR de la fruta
    bbox: tuple[int, int, int, int]  # x, y, w, h en la imagen original
    area: float                      # área en píxeles
    contour: np.ndarray              # contorno en coordenadas de la imagen original


# --------------------------------------------------------------------------
# Etapa 1: distancia al color del fondo
# --------------------------------------------------------------------------

def background_distance(img_bgr: np.ndarray, border: float = 0.06, k: int = 3) -> np.ndarray:
    """Mapa D(x) de distancia (en Lab) de cada píxel al color del fondo más cercano."""
    smooth = cv2.medianBlur(img_bgr, 7)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(border * min(h, w)))

    ring = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(ring, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    flat = lab.reshape(-1, 1, 3)
    dist = np.linalg.norm(flat - centers[None, :, :], axis=2).min(axis=1)
    return dist.reshape(h, w)


def coarse_mask(img_bgr: np.ndarray, morph: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Máscara inicial: Otsu sobre D(x) + limpieza morfológica.

    Devuelve (máscara, mapa D normalizado a 0..255) para reutilizar D después.
    """
    d = background_distance(img_bgr)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, d8


# --------------------------------------------------------------------------
# Etapa 2: refinamiento con GrabCut
# --------------------------------------------------------------------------

def refine_grabcut(img_bgr: np.ndarray, mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """Refina una máscara 0/255 con GrabCut.

    La zona interior (máscara erosionada) se marca como fruta probable/segura,
    el exterior lejano como fondo seguro, y la banda intermedia como dudosa
    para que GrabCut decida.
    """
    if mask.max() == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sure_fg = cv2.erode(mask, kernel, iterations=2)
    maybe = cv2.dilate(mask, kernel, iterations=2)

    gc = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask  # p. ej. si no hay píxeles de alguna clase
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # GrabCut puede "pegar" frutas vecinas rellenando el fondo entre ellas; solo
    # se le permite crecer en una banda estrecha alrededor de la máscara inicial.
    band = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return cv2.bitwise_and(out, band)


def fruit_mask(img_bgr: np.ndarray, use_grabcut: bool = True) -> np.ndarray:
    """Máscara binaria (0/255) de todas las frutas de la imagen."""
    mask, d8 = coarse_mask(img_bgr)
    if use_grabcut:
        mask = refine_grabcut(img_bgr, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return _fill_holes(mask, d8)


# --------------------------------------------------------------------------
# Etapa 3: separar frutas que se tocan
# --------------------------------------------------------------------------

def split_touching(mask: np.ndarray, min_dist_ratio: float = 0.6, peak_ratio: float = 0.45) -> np.ndarray:
    """Watershed sobre la transformada de distancia.

    Devuelve una imagen de etiquetas (0 = fondo, 1..n = frutas).

    La transformada de distancia dist(x) da, para cada píxel de la máscara, la
    distancia al fondo más cercano; en una fruta aproximadamente circular su
    máximo está en el centro y vale el radio r. Dos frutas pegadas producen
    dos máximos locales. Se toman como semillas los máximos locales separados
    al menos ``min_dist_ratio * r_max`` y con dist > ``peak_ratio * r_max``,
    y se inunda -dist desde ellas (watershed), de modo que las fronteras caen
    en los "cuellos" donde dist es mínima.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    n, comp = cv2.connectedComponents(mask)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    markers = np.zeros(mask.shape, np.int32)
    next_label = 1
    for i in range(1, n):
        region = comp == i
        r_max = float(dist[region].max())
        if r_max < 2:
            continue
        peaks = peak_local_max(
            np.where(region, dist, 0),
            min_distance=max(3, int(min_dist_ratio * r_max)),
            threshold_abs=peak_ratio * r_max,
            exclude_border=False,
        )
        if len(peaks) == 0:
            peaks = np.argwhere(dist == r_max)[:1]
        for y, x in peaks:
            markers[y, x] = next_label
            next_label += 1
    return watershed(-dist, markers, mask=mask > 0)


def find_fruits(
    img_bgr: np.ndarray,
    min_area_frac: float = 0.004,
    min_solidity: float = 0.6,
    max_border_frac: float = 0.25,
    pad: int = 6,
    split: bool = True,
    use_grabcut: bool = True,
) -> list[FruitRegion]:
    """Detecta cada fruta de la imagen, de mayor a menor área.

    Se descartan regiones muy pequeñas (``min_area_frac`` del total) y poco
    convexas (solidez = A / A_envolvente_convexa < ``min_solidity``), que
    suelen ser parches de pasto o sombras. Las bananas tienen solidez menor,
    por eso el umbral no es más estricto.
    """
    mask = fruit_mask(img_bgr, use_grabcut=use_grabcut)
    h, w = mask.shape
    # Antes de separar, quitamos objetos que recorren el borde (mesa, pared):
    # después del watershed quedarían partidos en trozos que no se detectan.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if _border_fraction(c, h, w) > max_border_frac:
            cv2.drawContours(mask, [c], -1, 0, thickness=cv2.FILLED)
    if split:
        labels = split_touching(mask)
    else:
        _, labels = cv2.connectedComponents(mask)

    regions: list[FruitRegion] = []
    for i in range(1, int(labels.max()) + 1):
        single = np.where(labels == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area_frac * h * w:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0 or area / hull_area < min_solidity:
            continue
        if _border_fraction(c, h, w) > max_border_frac:
            continue  # franjas de mesa, pared, etc. pegadas al borde

        x, y, bw, bh = cv2.boundingRect(c)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
        single = np.zeros_like(mask)
        cv2.drawContours(single, [c], -1, 255, thickness=cv2.FILLED)
        regions.append(
            FruitRegion(
                mask=single[y0:y1, x0:x1],
                crop=img_bgr[y0:y1, x0:x1].copy(),
                bbox=(x0, y0, x1 - x0, y1 - y0),
                area=float(area),
                contour=c,
            )
        )
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def largest_fruit(img_bgr: np.ndarray) -> FruitRegion | None:
    """La fruta de mayor área (caso de uso normal: una fruta por foto)."""
    regions = find_fruits(img_bgr, split=False)
    return regions[0] if regions else None


def fruit_on_uniform_background(
    img_bgr: np.ndarray,
    border: float = 0.05,
    min_chroma: float = 8.0,
    chroma_margin: float = 3.0,
    dark_ratio: float = 0.35,
    pad: int = 6,
    min_area_frac: float = 0.002,
    max_border_frac: float = 0.25,
) -> FruitRegion | None:
    """Fruta principal en una foto del protocolo propio (fondo blanco o gris uniforme).

    En una cartulina blanca o gris, las sombras cambian la luminosidad L pero
    casi no el color. Por eso aquí la decisión se basa en la **croma respecto
    al fondo**:

        C(x) = sqrt((a(x) - a_0)^2 + (b(x) - b_0)^2),

    donde (a_0, b_0) es la mediana del borde de la imagen en Lab. Un píxel es
    fruta si C(x) supera un umbral, o si es mucho más oscuro que el fondo
    (L < ``dark_ratio`` · L_0), para conservar zonas negras de podredumbre.
    La sombra (gris claro) y la moneda de referencia (gris metálico) quedan
    fuera.

    Después se conserva la componente conexa más grande y se rellenan todos
    sus huecos: en una foto con una sola fruta, cualquier hueco interno
    (manchas, cavidad del pedúnculo) pertenece a la fruta.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.medianBlur(img_bgr, 5), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    k = max(2, int(border * min(h, w)))
    ring = np.zeros((h, w), bool)
    ring[:k], ring[-k:], ring[:, :k], ring[:, -k:] = True, True, True, True
    a0, b0, L0 = np.median(a[ring]), np.median(b[ring]), np.median(L[ring])
    chroma = np.hypot(a - a0, b - b0)
    # umbral: croma máxima del fondo (percentil 99 del borde) más un margen
    t = max(min_chroma, float(np.percentile(chroma[ring], 99)) + chroma_margin)
    mask = (((chroma > t) | (L < dark_ratio * L0)) * 255).astype(np.uint8)

    ks = max(3, int(0.01 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours
                if cv2.contourArea(c) >= min_area_frac * h * w and _border_fraction(c, h, w) <= max_border_frac]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    single = np.zeros((h, w), np.uint8)
    cv2.drawContours(single, [c], -1, 255, thickness=cv2.FILLED)  # relleno completo: sin huecos

    x, y, bw, bh = cv2.boundingRect(c)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    return FruitRegion(
        mask=single[y0:y1, x0:x1],
        crop=img_bgr[y0:y1, x0:x1].copy(),
        bbox=(x0, y0, x1 - x0, y1 - y0),
        area=float(cv2.contourArea(c)),
        contour=c,
    )


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray, bg: int = 0) -> np.ndarray:
    """Pinta el fondo de un color constante para que el modelo no aprenda del fondo."""
    out = np.full_like(img_bgr, bg)
    out[mask > 0] = img_bgr[mask > 0]
    return out


def _border_fraction(contour: np.ndarray, h: int, w: int, margin: int = 2) -> float:
    """Fracción de puntos del contorno que están sobre el borde de la imagen."""
    pts = contour.reshape(-1, 2)
    on_border = (pts[:, 0] <= margin) | (pts[:, 0] >= w - 1 - margin) | \
                (pts[:, 1] <= margin) | (pts[:, 1] >= h - 1 - margin)
    return float(on_border.mean())


def _fill_holes(mask: np.ndarray, d8: np.ndarray | None = None, small_frac: float = 0.05) -> np.ndarray:
    """Rellena huecos internos de la máscara, pero solo los que son parte de la fruta.

    Un hueco se rellena si es pequeño (< ``small_frac`` del área de su objeto:
    brillos, semillas, manchas) o si su color está lejos del fondo (mediana de
    D alta). Un hueco grande con color de fondo -p. ej. el pasto en el centro
    de un anillo de frutas- se conserva como fondo.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    out = mask.copy()
    hierarchy = hierarchy[0]
    for i, c in enumerate(contours):
        parent = hierarchy[i][3]
        if parent < 0:
            continue  # contorno externo
        hole_area = cv2.contourArea(c)
        parent_area = cv2.contourArea(contours[parent])
        fill = hole_area < small_frac * parent_area
        if not fill and d8 is not None:
            hole = np.zeros_like(mask)
            cv2.drawContours(hole, [c], -1, 255, thickness=cv2.FILLED)
            fill = np.median(d8[hole > 0]) > 64
        if fill:
            cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out
