"""Pruebas de la estimación de tamaño en centímetros (fotos propias)."""

import cv2
import numpy as np
import pytest

from fruitq.size import characteristic_length, cross_validated_cm, fit_scale, to_cm


def contour_of(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def test_characteristic_length_circle_is_its_diameter():
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 100, 255, -1)
    assert characteristic_length(contour_of(mask)) == pytest.approx(200, rel=0.02)


def test_characteristic_length_rotated_ellipse_is_major_axis():
    """Un banano inclinado: la longitud no depende de la orientación."""
    mask = np.zeros((600, 600), np.uint8)
    cv2.ellipse(mask, (300, 300), (200, 40), 35, 0, 360, 255, -1)
    assert characteristic_length(contour_of(mask)) == pytest.approx(400, rel=0.02)


def test_fit_scale_is_median_ratio():
    # Cocientes 0.04, 0.05 y 0.10 (atípico): la mediana ignora el atípico.
    assert fit_scale([4, 5, 10], [100, 100, 100]) == pytest.approx(0.05)
    assert to_cm(200, 0.05) == pytest.approx(10)


def test_fit_scale_rejects_invalid_input():
    with pytest.raises(ValueError):
        fit_scale([], [])
    with pytest.raises(ValueError):
        fit_scale([5], [0])


def test_cross_validated_cm_excludes_own_group():
    """La foto de un grupo atípico no influye en su propia estimación."""
    cm = [4.0, 4.0, 5.0, 5.0, 30.0]
    px = [100, 100, 100, 100, 100]
    groups = ["a", "a", "b", "b", "c"]
    pred = cross_validated_cm(cm, px, groups)
    # Para "c", k sale de a y b (cocientes 0.04 y 0.05): mediana 0.045 -> 4.5 cm.
    assert pred[4] == pytest.approx(4.5)
    # Para "a", k sale de b y c (0.05, 0.05, 0.30): mediana 0.05 -> 5 cm.
    assert pred[0] == pytest.approx(5.0)
