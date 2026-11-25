# Fase 1 — Comprensión del negocio

## Problema
La clasificación manual de productos frescos según tamaño, madurez o defectos es lenta, subjetiva y propensa a errores. Esto genera pérdidas económicas, desperdicio de alimentos y falta de estandarización.

## Preguntas a responder (con fuentes)
- [ ] ¿Cuánto alimento se pierde en la etapa de poscosecha y comercialización? (FAO, DANE, MinAgricultura)
- [ ] ¿Cuánto cuesta la clasificación manual (tiempo y personal) frente a la automatizada?
- [ ] ¿Quiénes son los usuarios? (pequeños productores, plazas de mercado, agroindustria)
- [ ] ¿Qué normas de calidad existen? (p. ej. NTC del ICONTEC para frutas frescas)

## Objetivos
| Tipo | Objetivo | Criterio de éxito |
|---|---|---|
| Negocio | Reducir el error y el tiempo de clasificación | _definir_ |
| Analítica | Clasificar la calidad en ≥ 3 clases | F1-macro ≥ _X_ en test |
| Analítica | Minimizar que pase fruta mala como buena | Recall(`mala`) ≥ _X_ |
| Analítica | Estimar el tamaño | Exactitud ≥ _X_ frente a la etiqueta manual |

## Costo de los errores
Clasificar una fruta `mala` como `buena` (llega al cliente) es más costoso que el error inverso (se desperdicia una fruta buena). Esto justifica priorizar el **recall de `mala`** y analizar la matriz de confusión, no solo la exactitud.

## Formulación matemática (borrador, PI3)
Dada una imagen $x \in \mathbb{R}^{H\times W\times 3}$, se busca
$$f_\theta : x \mapsto (\hat y_q, \hat y_s),\quad \hat y_q \in \{\text{buena},\text{regular},\text{mala}\},\ \hat y_s \in \{\text{peq.},\text{med.},\text{gr.}\}$$
- Calidad: $\hat y_q = \arg\max_k p_\theta(k\mid x)$, con $\theta^* = \arg\min_\theta -\frac{1}{N}\sum_i \sum_k w_k\, y_{ik}\log p_\theta(k\mid x_i)$ (entropía cruzada ponderada por clase).
- Tamaño: máscara $M = \text{seg}(x)$; longitud $L_{px}$ = lado mayor del rectángulo de área mínima que contiene $M$; longitud real $\hat\ell = \hat k \cdot L_{px}$, con $\hat k = \operatorname{mediana}_i(\ell_i / L_{px,i})$ ajustado con las fotos propias a distancia fija; clase por umbrales $t_1,t_2$ (terciles de $\ell$ por fruta). Ver `docs/fases/04_modelado.md`.
