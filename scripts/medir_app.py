"""Mide la latencia de la aplicación y comprueba que reproduce los notebooks.

Para cada foto propia del conjunto ``principal`` (``data/raw/propias/``):
  1. Lee el archivo como bytes, igual que lo recibe la aplicación, y ejecuta
     ``fruitq.predict.analizar_bytes`` con la fruta de la anotación.
  2. Registra el tiempo de procesamiento (decodificación, segmentación,
     características, predicción y tamaño; sin la carga del modelo).
  3. Compara P(mala), la clase de calidad y la longitud en píxeles con las
     tablas de los notebooks (``reports/tabla_calidad_fotos_propias.csv`` y
     ``reports/tabla_tamano_fotos_propias.csv``).

Cada foto se procesa ``--repeticiones`` veces y se conserva la mediana, para
que la medida no dependa de la primera ejecución (cachés del sistema).
Guarda ``reports/tabla_latencia_app.csv``.

Uso:
    python scripts/medir_app.py
    python scripts/medir_app.py --repeticiones 5
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitq.config import LABELS, RAW_PROPIAS  # noqa: E402
from fruitq.data import read_annotations  # noqa: E402
from fruitq.predict import analizar_bytes, cargar_modelo, cargar_parametros  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeticiones", type=int, default=3)
    args = ap.parse_args()

    modelo, params = cargar_modelo(), cargar_parametros()
    anot = read_annotations(LABELS / "anotaciones_propias.csv")
    anot = anot[anot.conjunto == "principal"].reset_index(drop=True)
    faltan = [a for a in anot.archivo if not (RAW_PROPIAS / a).exists()]
    if faltan:
        sys.exit(f"Faltan {len(faltan)} fotos en {RAW_PROPIAS} (p. ej. {faltan[0]}). Cópialas desde el Drive del grupo.")

    filas = []
    for r in anot.itertuples():
        data = (RAW_PROPIAS / r.archivo).read_bytes()
        tiempos, res = [], None
        for _ in range(args.repeticiones):
            res = analizar_bytes(data, r.fruta, modelo, params)
            tiempos.append(res.segundos)
        filas.append(dict(archivo=r.archivo, fruta=r.fruta, segmentacion=res.segmentacion,
                          p_xgb=res.p_mala, calidad_app=res.calidad, L_px=res.longitud_px,
                          medida_cm=res.medida_cm, clase_tamano=res.clase_tamano,
                          latencia_ms=1000 * float(np.median(tiempos))))
    app = pd.DataFrame(filas)

    cal = pd.read_csv(ROOT / "reports" / "tabla_calidad_fotos_propias.csv")[["archivo", "p_xgb", "prediccion"]]
    tam = pd.read_csv(ROOT / "reports" / "tabla_tamano_fotos_propias.csv")[["archivo", "L_px"]]
    comp = app.merge(cal, on="archivo", suffixes=("", "_notebook")).merge(tam, on="archivo", suffixes=("", "_notebook"))
    dif_p = (comp.p_xgb - comp.p_xgb_notebook).abs()
    dif_L = (comp.L_px - comp.L_px_notebook).abs()
    iguales = (comp.calidad_app == comp.prediccion).mean()

    lat = app.latencia_ms
    print(f"Equipo: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print(f"Fotos: {len(app)} | repeticiones por foto: {args.repeticiones}")
    print(f"Latencia (ms): mediana {lat.median():.0f} | p90 {lat.quantile(0.9):.0f} | máximo {lat.max():.0f}")
    print(app.groupby("segmentacion").latencia_ms.agg(["size", "median"]).round(0).to_string())
    print(f"Diferencia con los notebooks: |Δp| máx {dif_p.max():.2e} | |ΔL_px| máx {dif_L.max():.2e} | "
          f"misma clase de calidad en {100 * iguales:.0f} % de las fotos")

    salida = ROOT / "reports" / "tabla_latencia_app.csv"
    app.round(4).to_csv(salida, index=False)
    print(f"Tabla guardada en {salida.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
