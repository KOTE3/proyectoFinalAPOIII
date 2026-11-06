"""Herramienta de etiquetado de los recortes de mixed_quality.

Muestra cada fruta recortada y permite asignarle una calidad con un clic.
Cada integrante guarda sus etiquetas en su propio archivo
(data/labels/recortes_<nombre>.csv) para evitar conflictos en Git.

Uso (desde la raíz del proyecto):
    streamlit run app/etiquetar.py

Requisitos: data/interim/segmentadas/ con segmentacion.csv y los recortes
(generados con scripts/segment_mixed.py o descomprimidos del .zip compartido).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitq.config import LABELS, SEGMENTADAS  # noqa: E402
from fruitq.io import read_image  # noqa: E402

CSV_SEGMENTACION = SEGMENTADAS / "segmentacion.csv"
OPCIONES = ["buena", "regular", "mala", "descartar"]
FRUTA = {"Apple": "manzana", "Banana": "banano", "Guava": "guayaba",
         "Lemon": "limon", "Orange": "naranja", "Pomegranate": "granada"}
CRITERIOS = """
**buena**: sin defectos visibles, color uniforme, apta para la venta.

**regular**: defectos leves (manchas pequeñas, golpe superficial, madurez irregular). Se vendería con descuento.

**mala**: podredumbre, moho, golpes profundos o deformación severa. No se vendería.

**descartar**: el recorte no sirve (es sombra o fondo, sale media fruta, hay dos frutas pegadas o está muy borroso).
"""


@st.cache_data
def cargar_recortes() -> pd.DataFrame:
    df = pd.read_csv(CSV_SEGMENTACION)
    # Rutas relativas a la raíz del proyecto y con "/" para que sirvan en cualquier SO
    df["recorte"] = df["recorte"].str.replace("\\", "/", regex=False)
    df["fruta"] = df["recorte"].map(lambda p: FRUTA.get(Path(p).parent.name, Path(p).parent.name))
    # Orden aleatorio pero fijo: todos ven la misma secuencia y la "parte" asignada
    # a cada integrante no se solapa con la de los demás.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["orden"] = np.arange(len(df))
    return df


def archivo_etiquetas(nombre: str) -> Path:
    limpio = "".join(c for c in nombre.lower().strip() if c.isalnum() or c in "-_") or "anonimo"
    return LABELS / f"recortes_{limpio}.csv"


def cargar_etiquetas(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["recorte", "fruta", "calidad", "etiquetador", "fecha"])


def guardar(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def imagen_rgb(path: Path, lado: int = 320) -> np.ndarray:
    img = read_image(path)
    mask_path = path.with_name(path.stem + "_mask.png")
    if mask_path.exists():
        mask = read_image(mask_path)[:, :, 0]
        contornos, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contornos, -1, (255, 0, 255), 1)
    h, w = img.shape[:2]
    escala = lado / max(h, w)
    img = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Etiquetado de frutas", page_icon="🍎", layout="centered")
    st.title("🍎 Etiquetado de recortes")

    if not CSV_SEGMENTACION.exists():
        st.error(f"No se encontró `{CSV_SEGMENTACION.relative_to(ROOT)}`. "
                 "Genera los recortes con `python scripts/segment_mixed.py` "
                 "o descomprime el .zip compartido en `data/interim/segmentadas/`.")
        st.stop()

    recortes = cargar_recortes()

    with st.sidebar:
        st.header("Configuración")
        nombre = st.text_input("Tu nombre", key="nombre", placeholder="p. ej. sebastian")
        partes = st.number_input("Integrantes que etiquetan", 1, 6, 3)
        parte = st.selectbox("Tu parte", list(range(1, partes + 1)),
                             help="Cada integrante elige una parte distinta para no repetir recortes.")
        frutas = st.multiselect("Frutas", sorted(recortes.fruta.unique()), default=sorted(recortes.fruta.unique()))
        st.divider()
        st.markdown("### Criterios")
        st.markdown(CRITERIOS)

    if not nombre:
        st.info("👈 Escribe tu nombre en la barra lateral para empezar.")
        st.stop()

    ruta = archivo_etiquetas(nombre)
    etiquetas = cargar_etiquetas(ruta)
    hechos = set(etiquetas.recorte)

    mios = recortes[(recortes.orden % partes == parte - 1) & recortes.fruta.isin(frutas)]
    pendientes = mios[~mios.recorte.isin(hechos)]

    total, n_hechos = len(mios), len(mios) - len(pendientes)
    st.progress(n_hechos / max(total, 1), text=f"{n_hechos} / {total} recortes de tu parte")
    if len(etiquetas):
        conteo = etiquetas.calidad.value_counts().reindex(OPCIONES, fill_value=0)
        st.caption(" · ".join(f"{k}: **{v}**" for k, v in conteo.items()))

    if pendientes.empty:
        st.success("🎉 ¡Terminaste tu parte! Sube tu archivo de etiquetas al repositorio.")
        st.code(f"git add {ruta.relative_to(ROOT).as_posix()}\ngit commit -m \"Etiquetas de {nombre}\"\ngit push")
        st.stop()

    fila = pendientes.iloc[0]
    path = ROOT / fila.recorte
    st.subheader(f"{fila.fruta.capitalize()}")
    if path.exists():
        st.image(imagen_rgb(path), caption=Path(fila.recorte).name)
    else:
        st.warning(f"No existe la imagen {fila.recorte}")

    def etiquetar(calidad: str) -> None:
        nueva = pd.DataFrame([{
            "recorte": fila.recorte, "fruta": fila.fruta, "calidad": calidad,
            "etiquetador": nombre, "fecha": datetime.now().isoformat(timespec="seconds"),
        }])
        guardar(ruta, pd.concat([etiquetas, nueva], ignore_index=True))

    cols = st.columns(4)
    colores = {"buena": "🟢", "regular": "🟡", "mala": "🔴", "descartar": "⚪"}
    for col, op in zip(cols, OPCIONES):
        col.button(f"{colores[op]} {op}", key=f"b_{op}", on_click=etiquetar, args=(op,), use_container_width=True)

    if len(etiquetas):
        if st.button("↩️ Deshacer la última"):
            guardar(ruta, etiquetas.iloc[:-1])
            st.rerun()

    st.caption(f"Tus etiquetas se guardan en `{ruta.relative_to(ROOT).as_posix()}`")


main()
