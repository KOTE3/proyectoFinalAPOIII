"""Recorta cada fruta individual de las imágenes con varias frutas (carpeta mixed_quality).

Para cada imagen de entrada guarda:
    <output>/<ruta_relativa>/<nombre>_f<k>.jpg   recorte de la fruta k
    <output>/<ruta_relativa>/<nombre>_f<k>_mask.png   su máscara
y un resumen <output>/segmentacion.csv con bbox, área y diámetro normalizado,
listo para revisar y etiquetar a mano.

Uso:
    python scripts/segment_mixed.py --input data/raw/kaggle --output data/interim/segmentadas
    python scripts/segment_mixed.py --input data/raw/kaggle --filter mixed --preview 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitq.io import list_images, read_image, write_image  # noqa: E402
from fruitq.segmentation import find_fruits  # noqa: E402
from fruitq.size import normalized_diameter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("data/interim/segmentadas"))
    ap.add_argument("--filter", default="mixed", help="Solo procesa rutas que contengan este texto (insensible a mayúsculas). Vacío = todas.")
    ap.add_argument("--min-area", type=float, default=0.005, help="Área mínima de una fruta como fracción de la imagen.")
    ap.add_argument("--preview", type=int, default=0, help="Guarda N imágenes con los contornos dibujados para revisar la segmentación.")
    args = ap.parse_args()

    images = [p for p in list_images(args.input) if args.filter.lower() in str(p).lower()]
    if not images:
        sys.exit(f"No se encontraron imágenes en {args.input} que contengan '{args.filter}'.")

    rows = []
    for i, path in enumerate(tqdm(images, desc="Segmentando")):
        img = read_image(path)
        regions = find_fruits(img, min_area_frac=args.min_area)
        rel = path.relative_to(args.input).parent

        for k, r in enumerate(regions):
            out = args.output / rel / f"{path.stem}_f{k}.jpg"
            write_image(out, r.crop)
            write_image(out.with_name(f"{path.stem}_f{k}_mask.png"), r.mask)
            rows.append({
                "origen": str(path),
                "recorte": str(out),
                "fruta_idx": k,
                "x": r.bbox[0], "y": r.bbox[1], "w": r.bbox[2], "h": r.bbox[3],
                "area_px": r.area,
                "d_norm": normalized_diameter(r.area, img.shape),
                # Columnas para la revisión manual: valido = 1 (fruta bien
                # recortada) / 0 (sombra, fondo, fruta partida o pegada).
                "valido": "",
                "calidad": "",
            })

        if i < args.preview:
            vis = img.copy()
            cv2.drawContours(vis, [r.contour for r in regions], -1, (0, 255, 0), 3)
            write_image(args.output / "_preview" / f"{path.stem}_preview.jpg", vis)

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "segmentacion.csv", index=False)
    print(f"{len(images)} imágenes -> {len(df)} frutas recortadas. Resumen en {args.output / 'segmentacion.csv'}")


if __name__ == "__main__":
    main()
