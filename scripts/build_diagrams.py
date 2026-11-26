"""Genera los diagramas del informe: flujo CRISP-DM del proyecto y arquitectura del sistema.

Salidas (vectoriales en PDF, con copia PNG para revisión rápida):
    reports/figures/diagrama_crispdm.pdf
    reports/figures/diagrama_arquitectura.pdf

Los diagramas se dibujan con matplotlib para que sean reproducibles y se
actualicen junto con el código. Las cifras que aparecen en ellos provienen de
``docs/fases/`` y de ``models/umbrales.json``; si cambian, se actualizan aquí.
El tamaño de cada figura corresponde al ancho de página de la plantilla IEEE a
dos columnas (7.16 in), y el texto no baja de 6.5 pt.

Uso:
    python scripts/build_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fruitq.config import FIGURES  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _fuente() -> str:
    """Arial (Windows, macOS) o Liberation Sans (Linux), que tiene las mismas medidas; si no, DejaVu Sans."""
    for nombre in ("Arial", "Liberation Sans"):
        try:
            font_manager.findfont(nombre, fallback_to_default=False)
            return nombre
        except ValueError:
            continue
    return "DejaVu Sans"


plt.rcParams.update({
    "font.family": [_fuente(), "DejaVu Sans"],   # DejaVu Sans cubre los subíndices que falten
    "pdf.fonttype": 42,          # fuentes TrueType incrustadas (requisito habitual de IEEE)
    "ps.fonttype": 42,
})

# Paleta: tinta neutra para el flujo principal y un solo color de acento para las iteraciones.
TINTA = "#1f2933"
TINTA_SUAVE = "#52606d"
BORDE = "#9aa5b1"
FONDO_BANDA = "#f2f4f7"
FASE = "#1d4e89"
ACENTO = "#c05621"
ARTEFACTO = "#e8f0fb"

TAM_TITULO = 7.0
TAM_TEXTO = 6.5
MARGEN = 0.06                    # margen interno de las cajas (in)


class Lienzo:
    """Figura con coordenadas en pulgadas y origen arriba a la izquierda."""

    def __init__(self, ancho: float, alto: float):
        self.ancho, self.alto = ancho, alto
        self.fig = plt.figure(figsize=(ancho, alto))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, ancho)
        self.ax.set_ylim(alto, 0)
        self.ax.axis("off")
        self.textos: list[tuple[plt.Text, tuple[float, float, float, float]]] = []

    # --- primitivas -------------------------------------------------------------------------
    def rect(self, x, y, w, h, fc="white", ec=BORDE, lw=0.6, radio=0.04, ls="-", z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radio}",
                                         fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def texto(self, x, y, s, caja=None, **kw):
        kw.setdefault("fontsize", TAM_TEXTO)
        kw.setdefault("color", TINTA)
        t = self.ax.text(x, y, s, zorder=5, **kw)
        if caja is not None:
            self.textos.append((t, caja))
        return t

    def caja(self, x, y, w, h, titulo, cuerpo="", fc="white", ec=BORDE, ls="-"):
        """Caja con título en negrita y cuerpo; devuelve sus puntos de anclaje."""
        self.rect(x, y, w, h, fc=fc, ec=ec, ls=ls, z=2)
        lim = (x, y, w, h)
        self.texto(x + MARGEN, y + MARGEN, titulo, caja=lim, ha="left", va="top",
                   fontsize=TAM_TITULO, fontweight="bold")
        if cuerpo:
            self.texto(x + MARGEN, y + MARGEN + 0.135, cuerpo, caja=lim, ha="left", va="top", linespacing=1.25)
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h),
                "x": x, "y": y, "w": w, "h": h}

    def flecha(self, puntos, color=TINTA_SUAVE, lw=0.8, ls="-", cabeza=True):
        """Polilínea ortogonal; la punta de flecha va en el último segmento."""
        for (x0, y0), (x1, y1) in zip(puntos[:-2], puntos[1:-1]):
            self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")
        self.ax.annotate("", xy=puntos[-1], xytext=puntos[-2], zorder=4,
                         arrowprops=dict(arrowstyle="-|>" if cabeza else "-", color=color, lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0, mutation_scale=7))

    # --- verificación y guardado ----------------------------------------------------------------
    def verificar(self, nombre: str) -> None:
        """Falla si algún texto se sale de su caja (evita cortes al cambiar las cifras)."""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        malos = []
        for t, (x, y, w, h) in self.textos:
            bb = t.get_window_extent(r)
            (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            if x0 < x + 0.02 or x1 > x + w - 0.02 or y0 < y + 0.015 or y1 > y + h - 0.015:
                malos.append(t.get_text().split("\n")[0])
        if malos:
            raise SystemExit(f"{nombre}: texto fuera de su caja: {malos}")

    def guardar(self, nombre: str) -> None:
        self.verificar(nombre)
        FIGURES.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            self.fig.savefig(FIGURES / f"{nombre}.{ext}", dpi=200, facecolor="white")
        plt.close(self.fig)
        print(f"reports/figures/{nombre}.pdf")


# =============================================================================================
# Diagrama de flujo CRISP-DM del proyecto
# =============================================================================================
FASES = [
    ("1", "Comprensión\ndel negocio", [
        ("Problema", "Clasificación manual lenta y subjetiva;\nen Colombia se pierde o desperdicia\nel 34 % de los alimentos (DNP, 2016)."),
        ("Salidas definidas", "Calidad: buena / regular / mala.\nTamaño: cm y pequeño / mediano /\ngrande."),
        ("Métricas", "F1-macro; error crítico: mala → buena;\nerror absoluto medio del tamaño (cm);\ntiempo por imagen."),
    ]),
    ("2", "Comprensión\nde los datos", [
        ("Kaggle: 6 frutas, buena / mala", "18 452 imágenes: 1 188 en gris, 30 %\ncasi duplicadas (ráfagas), granada\nbuena 5.5 veces más que las demás."),
        ("Variabilidad", "Fondos heterogéneos; solo el 36 %\ncon una fruta; distancia de cámara\nno controlada (tamaño no físico)."),
        ("Fotos propias", "45 fotos: 3 frutas × 3 estados × 5,\ndos celulares a 30 cm, fondo blanco\ny medida con regla."),
    ]),
    ("3", "Preparación\nde los datos", [
        ("Limpieza y partición", "Sin imágenes en gris; casi duplicados\nagrupados (aHash o ráfaga ≤ 1 s);\npartición agrupada 70/15/15 sin fuga."),
        ("Segmentación y representación", "Segmentación: k-medias en Lab, Otsu,\nGrabCut y watershed. Entradas: 57\ncaracterísticas y recorte de 128 px."),
        ("Etiquetado de mixed_quality", "4 716 recortes segmentados; tres\npersonas: 569 válidos con la clase\nregular; acuerdo κ = 0.66–0.78."),
    ]),
    ("4", "Modelado", [
        ("Modelos clásicos", "RL, SVM, Random Forest y XGBoost;\nrejilla con validación cruzada agrupada;\npesos por estrato fruta × calidad."),
        ("CNN desde cero", "3 bloques convolucionales,\n72 434 parámetros; aumento de datos\nsin alterar el tono."),
        ("Tres clases y tamaño", "Umbrales t₁, t₂ sobre P(mala),\nelegidos con los recortes; tamaño:\nL_cm = k·L_px (k = 0.0404 cm/px)."),
    ]),
    ("5", "Evaluación", [
        ("Test de Kaggle (2 540)", "XGBoost F1 0.964 > CNN 0.940\n(McNemar p < 10⁻⁴); P(mala) casi\nsiempre cerca de 0 o de 1."),
        ("Recortes con regular", "F1-macro de tres clases 0.62\n(validación cruzada agrupada\npor imagen de origen)."),
        ("Fotos propias (45)", "Calidad: F1 0.25, no generaliza\n(atajo de nitidez: AUC 0.55 → 0.91).\nTamaño: error medio 0.98 cm."),
    ]),
    ("6", "Despliegue", [
        ("Aplicación Streamlit", "Cámara o archivo → contorno, calidad,\nprobabilidad, tamaño y tiempo\n(≈ 0.1 s por foto)."),
        ("Uso responsable", "Advertencia permanente de no\ngeneralización; tabla ética (PI1)\ny matriz de impactos (PI2)."),
        ("Verificación", "La aplicación reproduce los notebooks\nen las 45 fotos; 27 pruebas\nautomáticas."),
    ]),
]


def diagrama_crispdm() -> None:
    W = 7.16
    x_fase, w_fase = 0.08, 0.98               # columna de fases (eje del ciclo CRISP-DM)
    x0, gap_x = 1.16, 0.13                    # cajas de actividades
    w_caja = 1.72
    h_caja, h_banda = 0.62, 0.74
    gutters = [0.16, 0.16, 0.16, 0.34, 0.16]  # el hueco 4–5 lleva la iteración de umbrales
    y = 0.08
    tops = []
    for i in range(len(FASES)):
        tops.append(y)
        y += h_banda + (gutters[i] if i < len(gutters) else 0)
    H = y + 0.36                              # espacio para la leyenda
    c = Lienzo(W, H)
    x_spine = x0 + 3 * w_caja + 2 * gap_x + 0.2

    anclas = []
    for i, ((num, nombre, cajas), top) in enumerate(zip(FASES, tops)):
        c.rect(x_fase, top, x_spine + 0.12 - x_fase, h_banda, fc=FONDO_BANDA, ec="none", radio=0.05, z=0)
        c.rect(x_fase, top, w_fase, h_banda, fc=FASE, ec="none", radio=0.05, z=1)
        c.texto(x_fase + 0.07, top + h_banda / 2, num, ha="left", va="center", color="white",
                fontsize=13, fontweight="bold")
        c.texto(x_fase + 0.25, top + h_banda / 2, nombre, ha="left", va="center", color="white",
                fontsize=TAM_TITULO, fontweight="bold", linespacing=1.15)
        fila = []
        for j, (titulo, cuerpo) in enumerate(cajas):
            fila.append(c.caja(x0 + j * (w_caja + gap_x), top + (h_banda - h_caja) / 2, w_caja, h_caja, titulo, cuerpo))
        for a, b in zip(fila[:-1], fila[1:]):
            c.flecha([a["r"], b["l"]])
        anclas.append(fila)
        if i:  # secuencia de fases por la columna izquierda
            xm = x_fase + w_fase / 2
            c.flecha([(xm, tops[i - 1] + h_banda), (xm, top)], color=FASE, lw=1.0)

    # Iteración 1 (realizada): la sobreconfianza vista en test lleva a umbrales sobre cuantiles.
    src, dst = anclas[4][0], anclas[3][2]
    y_g = tops[3] + h_banda + gutters[3] / 2
    c.flecha([src["t"], (src["t"][0], y_g), (dst["b"][0], y_g), dst["b"]], color=ACENTO, lw=1.0)
    c.texto((src["t"][0] + dst["b"][0]) / 2, y_g - 0.015,
            "Probabilidades extremas: la clase regular sale de umbrales sobre cuantiles, no de calibrar",
            ha="center", va="bottom", color=ACENTO, fontsize=6.3, style="italic")

    # Iteración 2 (trabajo futuro): la falta de generalización vuelve a la preparación de datos.
    src, dst = anclas[4][2], anclas[2][2]
    c.flecha([src["r"], (x_spine, src["r"][1]), (x_spine, dst["r"][1]), dst["r"]], color=ACENTO, lw=1.0, ls=(0, (3, 2)))
    c.texto(x_spine + 0.06, (src["r"][1] + dst["r"][1]) / 2,
            "Trabajo futuro: reentrenar sin nitidez,\ncon brillo normalizado y más fotos propias",
            ha="left", va="center", rotation=90, color=ACENTO, fontsize=6.3, style="italic", linespacing=1.2)

    # Leyenda
    yl = H - 0.18
    xl = x_fase + 0.04
    for etiqueta, color, ls in [("Flujo entre fases", FASE, "-"), ("Flujo dentro de la fase", TINTA_SUAVE, "-"),
                                ("Iteración realizada", ACENTO, "-"), ("Iteración propuesta (trabajo futuro)", ACENTO, (0, (3, 2)))]:
        c.flecha([(xl, yl), (xl + 0.28, yl)], color=color, lw=1.0, ls=ls)
        t = c.texto(xl + 0.34, yl, etiqueta, ha="left", va="center", color=TINTA_SUAVE)
        c.fig.canvas.draw()
        bb = t.get_window_extent(c.fig.canvas.get_renderer())
        xl += 0.34 + bb.width / c.fig.dpi + 0.28
    if xl - 0.28 > W - 0.05:
        raise SystemExit("diagrama_crispdm: la leyenda no cabe en el ancho")
    c.guardar("diagrama_crispdm")


# =============================================================================================
# Arquitectura del sistema
# =============================================================================================
def diagrama_arquitectura() -> None:
    W = 7.16
    xl, wl = 0.08, 0.3                         # etiquetas de carril (texto vertical)
    x0, wc, gx = xl + wl + 0.14, 1.16, 0.16    # rejilla de cinco columnas
    col = [x0 + k * (wc + gx) for k in range(5)]
    DISC = (0, (3, 2))

    # Alturas: carril de entrenamiento (4 filas) y carril de inferencia (2 filas + interfaz)
    h1 = 0.42
    fe = [0.18, 0.68, 1.18, 1.68]
    y_ent, h_ent = 0.08, 2.08
    y_inf = y_ent + h_ent + 0.12
    h2 = 0.52
    fq, fs, fi = y_inf + 0.22, y_inf + 0.92, y_inf + 1.62
    h_inf = fi + 0.44 + 0.1 - y_inf
    c = Lienzo(W, y_inf + h_inf + 0.36)

    def carril(y, h, nombre):
        c.rect(xl, y, W - 2 * xl, h, fc=FONDO_BANDA, ec="none", radio=0.05, z=0)
        c.rect(xl, y, wl, h, fc=FASE, ec="none", radio=0.05, z=1)
        c.texto(xl + wl / 2, y + h / 2, nombre, ha="center", va="center", rotation=90, color="white",
                fontsize=TAM_TITULO, fontweight="bold", linespacing=1.1)

    carril(y_ent, h_ent, "Entrenamiento (fuera de línea)")
    carril(y_inf, h_inf, "Inferencia (aplicación)")

    # --- Entrenamiento: cada cadena termina en un artefacto versionado en models/ ---
    k1 = c.caja(col[0], fe[0], wc, h1, "Kaggle", "17 264 imágenes")
    p1 = c.caja(col[1], fe[0], wc, h1, "prepare_dataset.py", "limpieza y partición")
    p2 = c.caja(col[2], fe[0], wc, h1, "extract_features.py", "segmentación, 57 caract.")
    n4 = c.caja(col[3], fe[0], wc, h1, "Notebook 04", "rejilla → XGBoost")
    a1 = c.caja(col[4], fe[0], wc, h1, "clasico_mejor.joblib", "escalado + XGBoost", fc=ARTEFACTO)
    for a, b in [(k1, p1), (p1, p2), (p2, n4), (n4, a1)]:
        c.flecha([a["r"], b["l"]])

    cn = c.caja(col[3], fe[1], wc, h1, "train_cnn.py", "CNN desde cero", ls=DISC)
    a2 = c.caja(col[4], fe[1], wc, h1, "cnn_*.pt", "solo comparación", fc=ARTEFACTO, ls=DISC)
    c.flecha([p1["b"], (p1["b"][0], cn["l"][1]), cn["l"]], ls=DISC)
    c.texto((p1["b"][0] + cn["l"][0]) / 2, cn["l"][1] - 0.03, "recortes de 128 px (cache_tiles.py)",
            ha="center", va="bottom", color=TINTA_SUAVE, style="italic")
    c.flecha([cn["r"], a2["l"]], ls=DISC)

    rc = c.caja(col[0], fe[2], wc, h1, "Recortes etiquetados", "569 de mixed_quality")
    n7 = c.caja(col[3], fe[2], wc, h1, "Notebook 07", "umbrales t₁, t₂ y Platt")
    fp = c.caja(col[0], fe[3], wc, h1, "Fotos propias", "45, medidas con regla")
    n6 = c.caja(col[3], fe[3], wc, h1, "Notebook 06", "escala k y terciles")
    au = c.caja(col[4], fe[2], wc, fe[3] + h1 - fe[2], "umbrales.json",
                "calidad: t₁, t₂, Platt\ntamaño: k, terciles", fc=ARTEFACTO)
    c.flecha([rc["r"], n7["l"]])
    c.texto((rc["r"][0] + n7["l"][0]) / 2, rc["r"][1] - 0.03, "puntajes de XGBoost; validación cruzada agrupada",
            ha="center", va="bottom", color=TINTA_SUAVE, style="italic")
    c.flecha([fp["r"], n6["l"]])
    c.texto((fp["r"][0] + n6["l"][0]) / 2, fp["r"][1] - 0.03, "lado mayor del rectángulo mínimo; una fruta fuera",
            ha="center", va="bottom", color=TINTA_SUAVE, style="italic")
    c.flecha([n7["r"], (au["x"], n7["r"][1])])
    c.flecha([n6["r"], (au["x"], n6["r"][1])])

    # --- Inferencia: fruitq.predict.analizar_bytes, llamada desde app/app.py ---
    e1 = c.caja(col[0], fq, wc, h2, "Entrada", "st.camera_input o\nst.file_uploader")
    e2 = c.caja(col[1], fq, wc, h2, "Lectura", "decode_image; lado\nmayor a 1024 px")
    e3 = c.caja(col[2], fq, wc, h2, "Segmentación", "fondo uniforme (respaldo:\nlargest_fruit)")
    e4 = c.caja(col[3], fq, wc, h2, "Características", "kaggle_view (256 px)\ny 57 características")
    e5 = c.caja(col[4], fq, wc, h2, "Calidad", "XGBoost → p; t₁, t₂ →\nbuena / regular / mala")
    s1 = c.caja(col[3], fs, wc, h2, "Longitud", "lado mayor de\nminAreaRect → L_px")
    s2 = c.caja(col[4], fs, wc, h2, "Tamaño", "L_cm = k·L_px; terciles\nde la fruta elegida")
    for a, b in [(e1, e2), (e2, e3), (e3, e4), (e4, e5), (s1, s2)]:
        c.flecha([a["r"], b["l"]])
    xs = e3["r"][0] + gx / 2
    c.flecha([(xs, e3["r"][1]), (xs, s1["l"][1]), s1["l"]])

    ui = c.caja(col[0], fi, col[4] + wc - col[0], 0.44, "Interfaz (app/app.py)",
                "Contorno · calidad y probabilidad calibrada · tamaño en cm y clase · "
                "tiempo de procesamiento · advertencia permanente de no generalización")
    c.flecha([e3["b"], (e3["b"][0], ui["y"])])
    c.texto(e3["b"][0] + 0.05, (e3["b"][1] + ui["y"]) / 2, "contorno", ha="left", va="center",
            color=TINTA_SUAVE, style="italic")
    c.flecha([s2["b"], (s2["b"][0], ui["y"])])
    x_der = col[4] + wc + 0.06
    c.flecha([e5["r"], (x_der, e5["r"][1]), (x_der, ui["r"][1]), (ui["r"][0], ui["r"][1])])
    c.texto(col[0] + 0.02, fs + h2 / 2, "fruitq.predict.analizar_bytes:\nla misma preparación que\nlos notebooks de evaluación",
            ha="left", va="center", color=TINTA_SUAVE, style="italic", linespacing=1.25)

    # Artefactos que lee la inferencia: etiquetas con el mismo color que en el entrenamiento
    def etiqueta(caja, s):
        t = c.texto(caja["x"] + caja["w"] - 0.06, caja["y"], s, ha="right", va="center", fontsize=6.0,
                    color=FASE, bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08", fc=ARTEFACTO, ec=FASE, lw=0.5))
        return t
    etiqueta(e5, "joblib + umbrales.json")
    etiqueta(s2, "umbrales.json")

    # Leyenda
    yl = c.alto - 0.18
    c.rect(x0, yl - 0.07, 0.3, 0.14, fc=ARTEFACTO, ec=BORDE, radio=0.03, z=2)
    c.texto(x0 + 0.38, yl, "Artefacto versionado en models/", ha="left", va="center", color=TINTA_SUAVE)
    c.rect(x0 + 2.4, yl - 0.07, 0.3, 0.14, fc="white", ec=BORDE, radio=0.03, ls=DISC, z=2)
    c.texto(x0 + 2.78, yl, "Modelo comparado, no usado por la aplicación", ha="left", va="center", color=TINTA_SUAVE)
    c.guardar("diagrama_arquitectura")


if __name__ == "__main__":
    diagrama_crispdm()
    diagrama_arquitectura()
