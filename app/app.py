"""Aplicación de clasificación de calidad y tamaño de una fruta.

Recibe una foto (cámara o archivo), segmenta la fruta principal y muestra:
- la imagen con el contorno de la fruta;
- la clase de calidad (buena, regular o mala) con su probabilidad calibrada;
- el tamaño en centímetros y su clase (pequeño, mediano o grande);
- el tiempo de procesamiento.

El análisis está en ``fruitq.predict``; los umbrales y la escala se leen de
``models/umbrales.json``. Especificación: ``docs/TRASPASO.md`` §4.5.
Documentación y latencia medida: ``docs/fases/06_despliegue.md``.

Uso (desde la raíz del proyecto):
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitq.predict import analizar_bytes, cargar_modelo, cargar_parametros, dibujar_contorno  # noqa: E402

FRUTAS = {"Banano": "banano", "Guayaba": "guayaba", "Manzana": "manzana",
          "Limón": "limon", "Naranja": "naranja", "Granada": "granada"}
COLOR_CALIDAD = {"buena": "green", "regular": "orange", "mala": "red"}
NOMBRE_TAMANO = {"pequeno": "pequeño", "mediano": "mediano", "grande": "grande"}


@st.cache_resource
def recursos() -> tuple[dict, dict]:
    return cargar_modelo(), cargar_parametros()


def advertencia(params: dict) -> str:
    cal, tam = params["calidad"], params["tamano"]
    eam = sum(tam["validacion"]["eam_cm"].values()) / len(tam["validacion"]["eam_cm"])
    return (
        "**El modelo no es confiable con fotos nuevas.** "
        f"La calidad se validó en recortes de Kaggle (F1-macro {cal['seleccion']['F1-macro']:.2f}), "
        f"pero en las fotos propias del grupo baja a {cal['fotos_propias']['F1-macro']:.2f}: "
        "el modelo depende de la nitidez y del brillo de las fotos de entrenamiento "
        "(`docs/fases/05_evaluacion.md`). "
        "El tamaño solo es válido con el protocolo de las fotos propias "
        f"(cámara a 30 cm, fondo blanco); en esas fotos el error medio fue de {eam:.1f} cm. "
        "Use los resultados como demostración, no para decidir sobre fruta real."
    )


def main() -> None:
    st.set_page_config(page_title="Calidad y tamaño de frutas", page_icon="🍎", layout="wide")
    st.title("🍎 Calidad y tamaño de frutas")

    modelo, params = recursos()
    cal, tam = params["calidad"], params["tamano"]
    st.warning(advertencia(params), icon="⚠️")

    with st.sidebar:
        st.header("Configuración")
        nombre_fruta = st.selectbox(
            "Fruta", list(FRUTAS),
            help="El modelo no reconoce el tipo de fruta. Se usa para asignar la clase de tamaño.",
        )
        fruta = FRUTAS[nombre_fruta]
        if fruta not in tam["terciles_cm"]:
            st.caption("Para esta fruta solo se estima la medida en cm: las clases de tamaño "
                       f"se calibraron con {', '.join(tam['terciles_cm'])}.")
        st.divider()
        st.markdown("### Cómo se decide")
        st.markdown(
            f"- **Calidad:** XGBoost estima p = P(mala). Es **buena** si p < {cal['t1']:.3f}, "
            f"**regular** si {cal['t1']:.3f} ≤ p < {cal['t2']:.3f} y **mala** si p ≥ {cal['t2']:.3f}.\n"
            "- **Probabilidad mostrada:** p calibrada con Platt; no cambia la clase.\n"
            f"- **Tamaño:** lado mayor del rectángulo mínimo × {tam['k_cm_por_px']:.4f} cm/px, "
            f"con la imagen a {tam['max_side_px']} px por el lado mayor."
        )
        st.caption("Para mejores resultados: una sola fruta, centrada, sobre fondo blanco o gris uniforme.")

    entrada = st.radio("Entrada", ["📷 Cámara", "📁 Subir imagen"], horizontal=True, label_visibility="collapsed")
    if entrada == "📷 Cámara":
        archivo = st.camera_input("Toma una foto de la fruta sobre un fondo claro y uniforme")
    else:
        archivo = st.file_uploader("Imagen de la fruta", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if archivo is None:
        st.info("Toma una foto o sube una imagen para analizarla.")
        st.stop()

    with st.spinner("Analizando la fruta…"):
        try:
            res = analizar_bytes(archivo.getvalue(), fruta, modelo, params)
        except IOError:
            st.error("No se pudo leer la imagen. Usa un archivo JPG o PNG.")
            st.stop()
    if res is None:
        st.error("No se detectó ninguna fruta. Toma la foto con la fruta centrada sobre un fondo claro y uniforme.")
        st.stop()

    col_img, col_res = st.columns([3, 2], gap="large")
    with col_img:
        st.image(dibujar_contorno(res), caption=f"Contorno detectado (segmentación: {res.segmentacion})",
                 width="stretch")
    with col_res:
        st.subheader(f"Calidad: :{COLOR_CALIDAD[res.calidad]}[{res.calidad}]")
        st.metric("Probabilidad de que la fruta esté mala (calibrada)", f"{100 * res.p_mala_calibrada:.1f} %")
        clase = NOMBRE_TAMANO[res.clase_tamano] if res.clase_tamano else "sin clase para esta fruta"
        st.subheader(f"Tamaño: {res.medida_cm:.1f} cm ({clase})")
        st.metric("Tiempo de procesamiento", f"{1000 * res.segundos:.0f} ms",
                  help="Desde la decodificación de la imagen hasta el resultado; no incluye la carga inicial del modelo.")
        with st.expander("Detalle"):
            st.markdown(
                f"- p = P(mala) de XGBoost, sin calibrar: **{res.p_mala:.4f}**\n"
                f"- Umbrales: t1 = {cal['t1']:.4f}, t2 = {cal['t2']:.4f}\n"
                f"- Longitud en píxeles (escala de {tam['max_side_px']} px): {res.longitud_px:.1f}\n"
                f"- Imagen procesada: {res.imagen.shape[1]} × {res.imagen.shape[0]} px"
            )
            if fruta in tam["terciles_cm"]:
                t1, t2 = tam["terciles_cm"][fruta]
                st.markdown(f"- Terciles de {nombre_fruta.lower()}: pequeño < {t1:.1f} cm ≤ mediano < {t2:.1f} cm ≤ grande")


if __name__ == "__main__":
    main()
