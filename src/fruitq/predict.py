"""Análisis de una foto nueva: segmentación, calidad en tres clases y tamaño.

Reúne en una sola función los pasos que usa la aplicación (``app/app.py``), con
los mismos parámetros que los notebooks de evaluación (``docs/TRASPASO.md`` §4.3
a §4.5):

1. **Escala.** La imagen se reduce a 1024 px por el lado mayor, como las fotos
   propias en ``notebooks/06_tamano.ipynb`` y ``notebooks/07_tres_clases.ipynb``.
2. **Segmentación.** ``fruit_on_uniform_background``; si no detecta fruta, se
   usa ``largest_fruit``.
3. **Calidad.** ``kaggle_view`` lleva la fruta a la escala de entrenamiento, se
   extraen las 57 características y XGBoost da p = P(mala | x). La clase sale de
   los umbrales t1 y t2 de ``models/umbrales.json``. La probabilidad que se
   muestra es la calibrada con Platt,

       P = sigma(a · logit(p) + b),

   que conserva el orden de p y, por tanto, la clase.
4. **Tamaño.** Lado mayor de ``cv2.minAreaRect`` sobre el contorno, expresado a
   la escala de 1024 px, multiplicado por k (cm/px). La clase de tamaño usa los
   terciles de la fruta indicada por el usuario, porque el modelo no reconoce el
   tipo de fruta.

Los parámetros se leen de ``models/umbrales.json``; no se escriben en el código.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import MODELS
from .features import extract_features
from .io import decode_image, resize_max
from .quality import kaggle_view, logit, three_class
from .segmentation import fruit_on_uniform_background, largest_fruit
from .size import characteristic_length, size_class

RUTA_UMBRALES = MODELS / "umbrales.json"
RUTA_MODELO = MODELS / "clasico_mejor.joblib"


@dataclass
class Resultado:
    """Salida del análisis de una foto."""

    imagen: np.ndarray            # imagen BGR a la escala de trabajo (lado mayor <= 1024 px)
    contorno: np.ndarray          # contorno de la fruta en coordenadas de ``imagen``
    segmentacion: str             # "fondo uniforme" o "respaldo (largest_fruit)"
    p_mala: float                 # P(mala | x) de XGBoost, sin calibrar
    p_mala_calibrada: float       # misma probabilidad calibrada con Platt
    calidad: str                  # buena, regular o mala
    longitud_px: float            # lado mayor del rectángulo mínimo, a la escala de 1024 px
    medida_cm: float              # longitud estimada en centímetros
    clase_tamano: str | None      # pequeno, mediano o grande; None si la fruta no tiene terciles
    segundos: float               # tiempo de procesamiento (sin contar la carga del modelo)


def cargar_parametros(ruta: str | Path = RUTA_UMBRALES) -> dict:
    """Umbrales de calidad, factor de escala y terciles de tamaño (``models/umbrales.json``)."""
    params = json.loads(Path(ruta).read_text(encoding="utf-8"))
    columna = params["calidad"]["columna"]
    if columna != "p_xgb":
        raise ValueError(f"El puntaje '{columna}' no está soportado; la aplicación usa XGBoost (p_xgb).")
    return params


def cargar_modelo(ruta: str | Path = RUTA_MODELO) -> dict:
    """Modelo clásico elegido (diccionario con ``modelo``, ``nombre`` y ``features``)."""
    return joblib.load(ruta)


def probabilidad_calibrada(p: float, a: float, b: float) -> float:
    """Calibración de Platt: sigma(a · logit(p) + b)."""
    z = a * float(logit([p])[0]) + b
    return float(1.0 / (1.0 + np.exp(-z)))


def analizar(img_bgr: np.ndarray, fruta: str, modelo: dict, params: dict) -> Resultado | None:
    """Calidad y tamaño de la fruta principal de la foto; ``None`` si no se detecta fruta."""
    t0 = time.perf_counter()
    lado = int(params["tamano"]["max_side_px"])
    img = resize_max(img_bgr, lado)

    reg = fruit_on_uniform_background(img)
    segmentacion = "fondo uniforme"
    if reg is None:
        reg = largest_fruit(img)
        segmentacion = "respaldo (largest_fruit)"
    if reg is None:
        return None

    crop, mask = kaggle_view(img, reg.contour)
    x = np.asarray([extract_features(crop, mask)])   # mismo orden de columnas que modelo["features"]
    p = float(modelo["modelo"].predict_proba(x)[0, 1])
    cal = params["calidad"]
    calidad = str(three_class([p], cal["t1"], cal["t2"])[0])
    p_cal = probabilidad_calibrada(p, cal["platt"]["a"], cal["platt"]["b"])

    tam = params["tamano"]
    longitud = characteristic_length(reg.contour) * lado / max(img.shape[:2])
    medida = longitud * tam["k_cm_por_px"]
    terciles = tam["terciles_cm"].get(fruta)
    clase = size_class(medida, tuple(terciles)) if terciles else None

    return Resultado(
        imagen=img, contorno=reg.contour, segmentacion=segmentacion,
        p_mala=p, p_mala_calibrada=p_cal, calidad=calidad,
        longitud_px=float(longitud), medida_cm=float(medida), clase_tamano=clase,
        segundos=time.perf_counter() - t0,
    )


def analizar_bytes(data: bytes, fruta: str, modelo: dict, params: dict) -> Resultado | None:
    """Igual que ``analizar``, pero a partir de la imagen codificada (JPG o PNG); incluye la decodificación en el tiempo."""
    t0 = time.perf_counter()
    res = analizar(decode_image(data), fruta, modelo, params)
    if res is not None:
        res.segundos = time.perf_counter() - t0
    return res


def dibujar_contorno(res: Resultado, grosor: int | None = None) -> np.ndarray:
    """Imagen RGB con el contorno de la fruta, para mostrarla en la aplicación."""
    img = res.imagen.copy()
    grosor = grosor or max(2, round(max(img.shape[:2]) / 300))
    cv2.drawContours(img, [res.contorno], -1, (255, 0, 255), grosor)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
