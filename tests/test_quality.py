"""Pruebas de la clase de calidad en tres niveles y de la preparación de fotos nuevas."""

import cv2
import numpy as np
import pytest

from fruitq.cnn import fruit_tile
from fruitq.quality import KAGGLE_D_NORM, fit_three_thresholds, kaggle_view, masked_tile, three_class
from fruitq.size import equivalent_diameter


def test_three_class_boundaries():
    p = [0.0, 0.29, 0.3, 0.69, 0.7, 1.0]
    assert list(three_class(p, 0.3, 0.7)) == ["buena", "buena", "regular", "regular", "mala", "mala"]
    with pytest.raises(ValueError):
        three_class(p, 0.8, 0.2)


def test_fit_three_thresholds_separable():
    rng = np.random.default_rng(42)
    p = np.concatenate([rng.uniform(0.00, 0.01, 30), rng.uniform(0.40, 0.50, 20), rng.uniform(0.98, 1.0, 10)])
    y = np.array(["buena"] * 30 + ["regular"] * 20 + ["mala"] * 10, dtype=object)
    t1, t2, f1 = fit_three_thresholds(p, y)
    assert f1 == pytest.approx(1.0)
    assert 0.01 < t1 <= 0.40 and 0.50 < t2 <= 0.98


def test_fit_three_thresholds_is_invariant_to_monotone_transform():
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 120)
    y = np.where(p + rng.normal(0, 0.2, 120) < 0.4, "buena", np.where(p < 0.8, "regular", "mala")).astype(object)
    f1_raw = fit_three_thresholds(p, y)[2]
    f1_logit = fit_three_thresholds(np.log(p / (1 - p)), y)[2]
    assert f1_raw == pytest.approx(f1_logit)


def test_fit_three_thresholds_rejects_unknown_label():
    with pytest.raises(ValueError):
        fit_three_thresholds([0.1, 0.9], ["buena", "podrida"])


def test_kaggle_view_scale_and_border():
    img = np.full((600, 800, 3), 230, np.uint8)
    cv2.circle(img, (60, 300), 50, (0, 0, 200), -1)  # fruta pegada al borde: el cuadrado sale de la imagen
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (60, 300), 50, 255, -1)
    contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    crop, m = kaggle_view(img, contour)
    assert crop.shape[:2] == m.shape
    d = equivalent_diameter(float((m > 0).sum()))
    assert d / 256 == pytest.approx(KAGGLE_D_NORM, rel=0.03)
    assert crop[m > 0].mean(axis=0)[2] > 150  # dentro de la máscara está la fruta roja


def test_masked_tile_matches_fruit_tile_background():
    crop = np.full((40, 80, 3), 200, np.uint8)
    mask = np.zeros((40, 80), np.uint8)
    mask[10:30, 20:60] = 255
    tile = masked_tile(crop, mask, 128)
    assert tile.shape == (128, 128, 3)
    assert tile[0, 0].sum() == 0 and tile[64, 64].sum() > 0
    _, ok = fruit_tile(np.zeros((64, 64, 3), np.uint8))
    assert not ok
