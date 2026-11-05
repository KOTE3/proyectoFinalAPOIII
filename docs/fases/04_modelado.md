# Fase 4 — Modelado

## Modelos
| Tipo | Modelo | Entrada | Hiperparámetros a buscar |
|---|---|---|---|
| Línea base | Clase mayoritaria / Regresión logística | Características | `C` |
| ML clásico | SVM (RBF) | Características | `C`, `gamma` |
| ML clásico | Random Forest | Características | `n_estimators`, `max_depth`, `min_samples_leaf` |
| ML clásico (opcional) | XGBoost / KNN | Características | `learning_rate`, `max_depth` / `k` |
| Deep learning | CNN desde cero (3 bloques conv + pooling + densas) | Imagen 128×128 | lr, batch, dropout, filtros |

Búsqueda: `GridSearchCV` con `StratifiedKFold(k=5)` y métrica `f1_macro`.

## Registro de experimentos
| Fecha | Modelo | Mejores hiperparámetros | F1-macro CV | Notas |
|---|---|---|---|---|
