"""Pruebas con imágenes sintéticas: círculos de color sobre fondo uniforme."""

import math

import cv2
import numpy as np
import pytest

from fruitq.features import extract_features, feature_names
from fruitq.segmentation import find_fruits, largest_fruit
from fruitq.size import equivalent_diameter, fit_thresholds, normalized_diameter, size_class


def synthetic(bg=(240, 240, 240), circles=((200, 200, 80, (30, 30, 200)),), shape=(400, 600)):
    img = np.full((*shape, 3), bg, dtype=np.uint8)
    for cx, cy, r, color in circles:
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


@pytest.mark.parametrize("bg", [(240, 240, 240), (20, 20, 20)])
def test_single_fruit_area(bg):
    img = synthetic(bg=bg)
    fruit = largest_fruit(img)
    assert fruit is not None
    expected = math.pi * 80**2
    assert abs(fruit.area - expected) / expected < 0.05


def test_multiple_fruits():
    img = synthetic(circles=((120, 200, 60, (30, 30, 200)), (300, 200, 70, (30, 180, 60)), (480, 200, 50, (0, 140, 255))))
    assert len(find_fruits(img)) == 3


def test_size():
    assert equivalent_diameter(math.pi * 50**2) == pytest.approx(100)
    assert normalized_diameter(math.pi * 50**2, (200, 400, 3)) == pytest.approx(0.5)
    t = fit_thresholds([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert size_class(0.05, t) == "pequeno"
    assert size_class(0.35, t) == "mediano"
    assert size_class(0.9, t) == "grande"


def test_features_shape():
    fruit = largest_fruit(synthetic())
    f = extract_features(fruit.crop, fruit.mask)
    assert f.shape == (len(feature_names()),)
    assert np.isfinite(f).all()


def test_uniform_background_ignores_shadow_and_coin():
    """En el protocolo propio: la sombra gris y la moneda no forman parte de la fruta."""
    from fruitq.segmentation import fruit_on_uniform_background

    img = synthetic(bg=(235, 235, 235), circles=())
    cv2.ellipse(img, (210, 230), (85, 40), 0, 0, 360, (150, 150, 150), -1)  # sombra gris
    cv2.circle(img, (200, 200), 80, (40, 160, 220), -1)                      # fruta amarilla
    cv2.circle(img, (450, 200), 25, (120, 125, 125), -1)                     # moneda gris
    fruit = fruit_on_uniform_background(img)
    assert fruit is not None
    expected = math.pi * 80**2
    assert abs(fruit.area - expected) / expected < 0.05
    x, y, w, h = fruit.bbox
    assert x + w < 400  # la moneda queda fuera
