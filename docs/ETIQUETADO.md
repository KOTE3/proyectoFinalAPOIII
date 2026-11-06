# Guía de etiquetado de recortes

**Objetivo:** asignar una calidad (`buena` / `regular` / `mala`) a las frutas recortadas de la carpeta `mixed_quality` de Kaggle. Es la única fuente de ejemplos de la clase **regular** fuera de las fotos propias.

**Meta:** unos **600 recortes válidos** entre todo el grupo, cerca de 200 por persona y unos 20–30 minutos cada uno.

## 1. Preparar (una sola vez)

1. Clonar el repositorio y entrar a la carpeta:
   ```bash
   git clone <URL del repositorio>
   cd <carpeta del repositorio>
   ```
2. Descargar `recortes_segmentados.zip` (≈ 17 MB, enlace en el grupo) y **descomprimirlo en la raíz del repositorio**. Debe quedar la carpeta `data/interim/segmentadas/` con un `segmentacion.csv` adentro.
3. Crear el entorno e instalar lo mínimo:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
   pip install streamlit opencv-python pandas
   ```

## 2. Etiquetar

```bash
streamlit run app/etiquetar.py
```

Se abre en el navegador. En la barra lateral:
- **Tu nombre**: tus etiquetas se guardan en `data/labels/recortes_<nombre>.csv`.
- **Integrantes que etiquetan**: 3. **Tu parte**: cada uno elige un número distinto (1, 2 o 3), así nadie repite recortes.

Después haces clic en un botón por cada fruta. Si te equivocas, usa **Deshacer la última**. Puedes cerrar y seguir otro día: retoma donde quedaste.

## 3. Criterios (iguales para todos)

| Etiqueta | Cuándo usarla |
|---|---|
| 🟢 **buena** | Sin defectos visibles, color uniforme, apta para la venta |
| 🟡 **regular** | Defectos leves: manchas pequeñas, golpe superficial, madurez irregular. Se vendería con descuento |
| 🔴 **mala** | Podredumbre, moho, golpes profundos o deformación severa. No se vendería |
| ⚪ **descartar** | El recorte no sirve: es sombra o fondo, sale media fruta, hay dos frutas pegadas o está muy borroso |

Si dudas entre dos, elige la **peor**: en este problema es más grave dejar pasar una fruta mala que descartar una buena.

## 4. Entregar

```bash
git add data/labels/recortes_<tu-nombre>.csv
git commit -m "Etiquetas de <tu-nombre>"
git push
```

## 5. Consistencia entre etiquetadores (para el informe)

Al final, cada integrante etiqueta también **50 recortes de la parte de otro**. Con esos recortes se calcula el acuerdo entre personas (κ de Cohen), lo que muestra si los criterios son claros y da un límite de cuánto puede acertar un modelo: si las personas solo coinciden en el 80 %, no se puede esperar un 99 %.
