"""Pruebas del análisis de una foto nueva que usa la aplicación (``fruitq.predict``)."""

import cv2
import numpy as np
import pytest

from fruitq.config import QUALITY_CLASSES, SIZE_CLASSES
from fruitq.predict import analizar, analizar_bytes, cargar_modelo, cargar_parametros, probabilidad_calibrada


@pytest.fixture(scope="module")
def recursos():
    return cargar_modelo(), cargar_parametros()


def foto_sintetica(lado_mayor: int = 1600, eje_px: int = 500) -> np.ndarray:
    """Fruta naranja (elipse) sobre fondo blanco, con el lado mayor de la imagen en ``lado_mayor`` px."""
    h, w = int(lado_mayor * 0.75), lado_mayor
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (eje_px // 2, int(eje_px * 0.4)), 0, 0, 360, (30, 120, 230), -1)
    return img


def test_analizar_devuelve_clases_validas(recursos):
    modelo, params = recursos
    res = analizar(foto_sintetica(), "manzana", modelo, params)
    assert res is not None
    assert res.calidad in QUALITY_CLASSES
    assert res.clase_tamano in SIZE_CLASSES
    assert 0.0 <= res.p_mala <= 1.0 and 0.0 <= res.p_mala_calibrada <= 1.0
    assert max(res.imagen.shape[:2]) == params["tamano"]["max_side_px"]
    assert res.segmentacion == "fondo uniforme"
    assert res.segundos > 0


def test_tamano_no_depende_de_la_resolucion(recursos):
    """La longitud se expresa a la escala de 1024 px, aunque la foto llegue con otra resolución."""
    modelo, params = recursos
    grande = analizar(foto_sintetica(2048, 1000), "manzana", modelo, params)
    pequena = analizar(foto_sintetica(640, 312), "manzana", modelo, params)
    esperado = 1000 * 1024 / 2048 * params["tamano"]["k_cm_por_px"]
    assert grande.medida_cm == pytest.approx(esperado, rel=0.03)
    assert pequena.medida_cm == pytest.approx(esperado, rel=0.03)


def test_fruta_sin_terciles_no_tiene_clase_de_tamano(recursos):
    modelo, params = recursos
    res = analizar(foto_sintetica(), "naranja", modelo, params)
    assert res.clase_tamano is None and res.medida_cm > 0


def test_imagen_sin_fruta(recursos):
    modelo, params = recursos
    assert analizar(np.full((600, 800, 3), 235, np.uint8), "banano", modelo, params) is None


def test_analizar_bytes_decodifica_jpg(recursos):
    modelo, params = recursos
    ok, buf = cv2.imencode(".jpg", foto_sintetica())
    assert ok
    res = analizar_bytes(buf.tobytes(), "guayaba", modelo, params)
    assert res is not None and res.calidad in QUALITY_CLASSES
    with pytest.raises(IOError):
        analizar_bytes(b"no es una imagen", "guayaba", modelo, params)


def test_calibracion_conserva_el_orden():
    a, b = 0.36, -1.83
    p = [0.01, 0.2, 0.5, 0.9, 0.999]
    cal = [probabilidad_calibrada(x, a, b) for x in p]
    assert cal == sorted(cal)
    assert all(0 < c < 1 for c in cal)
