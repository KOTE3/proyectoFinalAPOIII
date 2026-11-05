"""Rutas y constantes compartidas por todo el proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW_KAGGLE = DATA / "raw" / "kaggle"
RAW_PROPIAS = DATA / "raw" / "propias"
SEGMENTADAS = DATA / "interim" / "segmentadas"
PROCESSED = DATA / "processed"
LABELS = DATA / "labels"

MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

KAGGLE_DATASET = "ryandpark/fruit-quality-classification"

# Clases de calidad (ajustar según la definición acordada en data/README.md)
QUALITY_CLASSES = ["buena", "regular", "mala"]
SIZE_CLASSES = ["pequeno", "mediano", "grande"]

IMG_SIZE = 128  # lado de la imagen de entrada a la CNN
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
