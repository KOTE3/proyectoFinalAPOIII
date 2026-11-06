"""CNN pequeña entrenada desde cero y utilidades de preprocesamiento.

Entrada: recorte de la fruta principal con el fondo en negro, centrado en un
lienzo cuadrado y redimensionado a 128x128 (``fruit_tile``). Enmascarar el
fondo evita que la red aprenda atajos del entorno (pasto, manos, mesa).

Arquitectura (``SmallCNN``): 3 bloques
    [Conv 3x3 -> BatchNorm -> ReLU] x 2 -> MaxPool 2x2
con 32, 64 y 128 filtros, seguidos de Global Average Pooling, Dropout y
una capa densa de salida.

- La convolución 3x3 comparte pesos en toda la imagen (equivarianza a
  traslaciones): el mismo detector de "mancha" funciona en cualquier lugar.
- Dos convoluciones 3x3 seguidas ven un campo receptivo de 5x5 con menos
  parámetros (2·9·C² frente a 25·C²) y una no linealidad extra.
- El MaxPool reduce la resolución a la mitad: los bloques profundos ven
  regiones más grandes (campo receptivo final ≈ 36 px de 128).
- BatchNorm normaliza las activaciones de cada canal en el lote, lo que
  estabiliza y acelera el entrenamiento.
- El Global Average Pooling (promedio por canal) reemplaza a un Flatten
  grande: ~0.29 M parámetros en total, lo que reduce el sobreajuste.

Pérdida: entropía cruzada ponderada
    L = - (1 / sum_i w_i) * sum_i w_i * log softmax(z_i)[y_i]
con w_i el peso del estrato fruta×calidad de la muestra i.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .segmentation import find_fruits

TILE = 128
# Media y desviación por canal (RGB) de ImageNet; solo normalizan la escala.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fruit_tile(img_bgr: np.ndarray, size: int = TILE) -> tuple[np.ndarray, bool]:
    """Recorte enmascarado de la fruta principal, cuadrado y de ``size`` px (RGB uint8).

    Devuelve (tile, seg_ok). Si no se detecta fruta, usa la imagen completa.
    """
    regs = find_fruits(img_bgr)
    if regs:
        r = regs[0]
        crop = r.crop.copy()
        crop[r.mask == 0] = 0
        ok = True
    else:
        crop, ok = img_bgr, False
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    tile = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), ok


def to_tensor(tiles_rgb: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float normalizado."""
    x = tiles_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 2, widths: tuple[int, ...] = (32, 64, 128), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(_block(cin, w))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
