# Ética e impactos (competencias PI1 y PI2)

## Marco de referencia

- **Código de Ética y Conducta Profesional de la ACM (2018).** Sus principios se citan por número.
- **Ley 842 de 2003, Título IV:** código de ética para el ejercicio de la ingeniería y sus profesiones afines en Colombia.
- **Ley 1581 de 2012:** régimen general de protección de datos personales en Colombia.

| Principio | Enunciado |
|---|---|
| ACM 1.1 | Contribuir a la sociedad y al bienestar humano |
| ACM 1.2 | Evitar el daño |
| ACM 1.3 | Ser honesto y digno de confianza |
| ACM 1.4 | Ser justo y actuar para no discriminar |
| ACM 1.5 | Respetar el trabajo necesario para producir nuevas ideas y obras |
| ACM 1.6 | Respetar la privacidad |
| ACM 2.5 | Evaluar los sistemas y sus impactos de forma completa, incluidos los riesgos |
| ACM 3.1 | Hacer del bien público la preocupación central |

## Tabla de decisiones éticas (PI1)

La columna **Estado** indica si la medida ya está aplicada en el proyecto o si sigue pendiente.

| # | Tema | Situación o decisión | Riesgo | Principio | Medida adoptada | Estado |
|---|---|---|---|---|---|---|
| E1 | Propiedad intelectual | Uso del dataset *Fruit Quality Classification* de Kaggle. Se publica como CC0, pero es una copia de FruitNet (Meshram y Patil, 2022), cuyo artículo se publicó con licencia CC BY 4.0. | Atribuir la autoría a quien solo redistribuyó los datos; omitir la fuente original, que el enunciado del curso considera fraude. | ACM 1.5, ACM 1.3 | Se citan la página de Kaggle y el artículo original. El repositorio no redistribuye las imágenes: se descargan con `scripts/download_data.py`. | Parcial: citado en el `README.md`; falta la referencia en el informe |
| E2 | Sesgo del dataset | Las imágenes de Kaggle se tomaron en India, con las variedades, fondos y cámaras de ese contexto. La aplicación se usaría con fruta colombiana. | Menor desempeño con variedades locales de guayaba o banano, que perjudica al productor cuyo producto no se parece al del dataset. | ACM 1.4, ACM 2.5 | Se fotografiaron 15 frutas compradas en Colombia (45 fotos) para medir la generalización a fruta local. La evaluación mostró que el modelo **no generaliza** (F1-macro de 3 clases 0.25), y el resultado se reporta sin ocultarlo. | Evaluada; pendiente reentrenar con fruta local |
| E3 | Sesgo del dataset | El error en test no es igual entre frutas: 10.4 % en manzana buena y 8.1 % en naranja buena, frente a 0 % en banano y limón buenos. | Rechazo injusto de fruta sana, con pérdidas concentradas en los productores de ciertas frutas. | ACM 1.4 | Se reporta la exactitud por fruta y calidad (fase 5) y se usan pesos de muestra por estrato fruta × calidad. La aplicación debe informar el desempeño de la fruta analizada. | Aplicada en la evaluación; pendiente en la aplicación |
| E4 | Sesgo del dataset | Las características de nitidez son las más importantes del modelo, pero su relación con la calidad cambia de dirección según la fruta. | El modelo decide por la cámara o la sesión de captura y no por el defecto; con otra cámara la decisión cambia sin que el usuario lo note. | ACM 2.5, ACM 1.3 | Ablación documentada en la fase 4 (F1 0.957 con todas las características, 0.943 sin nitidez). En las fotos propias se confirmó: al neutralizar la nitidez, la AUC buena vs mala sube de 0.55 a 0.91. | Confirmado; pendiente reentrenar sin nitidez |
| E5 | Etiquetado | La clase `regular` no existe en Kaggle y depende del criterio de cada etiquetador. El acuerdo se mide en los recortes comunes, sobre los que ambos etiquetadores consideraron válidos. Sebastián y Giuseppe: κ de Cohen = 0.78 (ponderado: 0.83) en 20 de 50 recortes, con intervalo de confianza del 95 % de [0.52, 1.00] por *bootstrap*; los desacuerdos solo ocurren entre clases vecinas. Luis y Giuseppe: κ = 0.66 (ponderado: 0.66) en 38 de 59 recortes, con intervalo de [0.45, 0.84]; 2 de los 8 desacuerdos son entre `buena` y `mala`. Ambos valores son un acuerdo sustancial según la escala de Landis y Koch. En la decisión de descartar un recorte, κ = 0.52 y 0.80, respectivamente. | El modelo aprende el criterio de una sola persona, o se reporta un desempeño mayor del que permite el acuerdo humano. | ACM 1.3, ACM 2.5 | Criterios escritos comunes (`docs/ETIQUETADO.md` §3), 50 recortes comunes y κ reportado. Las etiquetas no se editan después de calcular el acuerdo: las discrepancias se resuelven con una adjudicación documentada. | Parcial: falta el par Sebastián–Luis |
| E6 | Privacidad | Toma de fotos de fruta en plazas, supermercados o viviendas. | Captar personas, rostros o datos de vendedores sin consentimiento. | ACM 1.6, Ley 1581 de 2012 | Solo se encuadra la fruta, se pide permiso en lugares públicos y se descartan las fotos con personas. Las 45 fotos propias muestran solo la fruta sobre fondo uniforme. | Aplicada |
| E7 | Privacidad | 20 de las 45 fotos propias (todas las del iPhone 13) conservan las coordenadas GPS en sus metadatos EXIF. | Revelar la ubicación de la vivienda de un integrante al compartir las fotos con otros grupos, como pide el enunciado. | ACM 1.6, Ley 1581 de 2012 | Las imágenes no se versionan en Git (`.gitignore`). Antes de compartirlas fuera del grupo se eliminan los metadatos de ubicación. | Aplicada en Git; pendiente antes de compartir |
| E8 | Errores del modelo | Fruta mala aprobada como buena. En test, 52 de 1 048 frutas malas (5.0 %) se clasifican como buenas, y en 34 de esos 52 casos el modelo asigna P(mala) < 0.2. | Fruta en mal estado llega al consumidor. Una revisión humana limitada a los casos dudosos no detectaría la mayoría de estos errores, porque el modelo se equivoca con alta confianza. | ACM 1.2, ACM 3.1, Ley 842 de 2003 | El recall de `mala` es una métrica principal (fase 1). El sistema se presenta como apoyo a la inspección, no como certificación de inocuidad. La aplicación debe mostrar la probabilidad junto con esta advertencia. | Aplicada en la evaluación; pendiente en la aplicación |
| E9 | Errores del modelo | La clase `regular` se obtiene con umbrales sobre P(mala \| x), pero XGBoost está sobreconfiado: solo el 4.4 % de las probabilidades de test cae entre 0.2 y 0.8. | Presentar como `regular` una decisión sin respaldo en los datos. | ACM 1.3, ACM 2.5 | Los umbrales se eligen con validación cruzada sobre 569 recortes etiquetados (F1-macro 0.62) y se publican con sus métricas en `models/umbrales.json`. La aplicación muestra una probabilidad calibrada con Platt. | Aplicada |
| E10 | Errores del modelo | Estimación del tamaño: error absoluto medio de 0.98 cm y 55.6 % de exactitud en la clase de tamaño. Una sesión con la cámara más cerca tiene un sesgo de +3.37 cm. | Fijar precios o rechazar lotes con una medida que no es confiable a nivel de clase. | ACM 1.3, ACM 2.5 | Las limitaciones están documentadas en la fase 5. La aplicación debe mostrar la longitud en cm con su error típico y presentar la clase de tamaño como orientativa. | Aplicada en la documentación; pendiente en la aplicación |
| E11 | Honestidad en los resultados | El 30 % de las imágenes de Kaggle son casi duplicados, y se comparan varios modelos y configuraciones. | Resultados inflados por fuga de información o por elegir el modelo mirando el test. | ACM 1.3, ACM 2.5 | Partición por grupos de casi duplicados, selección con validación cruzada y un solo uso del test (fase 5). Semilla fija 42. También se reportan los resultados desfavorables, como la exactitud de la clase de tamaño. | Aplicada |
| E12 | Empleo | Automatizar la clasificación que hoy hacen personas en plazas, centrales de abasto y empacadoras. | Pérdida de empleo de clasificadores manuales, muchas veces trabajadores con pocas alternativas laborales. | ACM 1.1, ACM 3.1 | El sistema se diseña como herramienta de apoyo: la decisión final queda en la persona, que puede revisar más fruta con un criterio uniforme. El análisis económico está en la matriz de impactos. | Aplicada en el diseño |

### Responsabilidad ante los errores

- El sistema **recomienda** una clase; quien vende o compra la fruta toma la decisión final. Esta distinción debe quedar visible en la aplicación y en el informe.
- Las cifras de error (E3, E8, E10) se miden en datos que el modelo no vio. Aun así, solo garantizan el desempeño en condiciones parecidas a las del dataset, como establece E2.
- El grupo responde por lo que afirma del sistema: ninguna cifra se presenta sin el conjunto de datos y el procedimiento que la producen.

### Acciones pendientes que salen de la tabla

- [ ] E1: citar FruitNet (Meshram y Patil, 2022, *Data in Brief* 40, 107686, doi:10.1016/j.dib.2021.107686) en las referencias del informe.
- [x] E2 y E4: evaluar el clasificador en las fotos propias y comparar los resultados por celular.
- [ ] E2 y E4: reentrenar sin nitidez, normalizar el brillo e incluir fruta local, validando con fotos nuevas.
- [ ] E3, E8 y E10: en la aplicación, mostrar la probabilidad, el desempeño por fruta, el error típico del tamaño y la advertencia de uso como apoyo.
- [ ] E5: medir el acuerdo del par Sebastián–Luis, con 50 recortes de la parte 2 etiquetados por Sebastián (`docs/PLAN_DE_TRABAJO.md` §2).
- [ ] E7: eliminar los metadatos de ubicación antes de compartir las fotos fuera del grupo.
- [x] E9: elegir y evaluar los umbrales de la clase `regular`.

### Origen de las cifras

- Errores del clasificador (E3, E8, E9): `models/clasico_mejor.joblib` sobre la partición de test de `data/processed/features.parquet`, con umbral 0.5.
- Acuerdo entre etiquetadores (E5): recortes comunes de `data/labels/recortes_giuseppe.csv` y `data/labels/recortes_sebastian.csv`, sin la categoría `descartar`; intervalo con 5 000 remuestreos y semilla 42.
- Metadatos (E7): lectura del bloque EXIF de `data/raw/propias/*.jpg`.
- Tamaño (E10): `notebooks/06_tamano.ipynb`. Casi duplicados (E11): fase 2.

## Matriz de impactos (PI2)

### Contexto

- En Colombia se pierden o desperdician **9.76 millones de toneladas de alimentos al año**, el 34 % de los alimentos disponibles (DNP, 2016).
  - Por etapa: producción agrícola 40.5 %, distribución 20.6 %, poscosecha y almacenamiento 19.8 %, consumo 15.6 % y procesamiento industrial 3.5 %.
  - La clasificación por calidad y tamaño ocurre en la poscosecha y la distribución, que suman el 40.4 % de esas pérdidas.
- En el mundo, cerca del 14 % de los alimentos se pierde entre la cosecha y la venta al por menor, y las pérdidas de frutas y hortalizas son mayores que las de cereales y leguminosas en casi todas las etapas (FAO, 2019).
- **Partes interesadas:** pequeños productores, comerciantes de plazas y centrales de abasto, clasificadores manuales, consumidores, agroindustria de pulpas y jugos, y el ambiente.

### Matriz

Las mitigaciones marcadas como *(pendiente)* dependen de tareas que aún no están terminadas. La columna **Ética** remite a la fila correspondiente de la tabla PI1.

| Dimensión | Impacto positivo | Impacto negativo o riesgo | Mitigación | Indicador de seguimiento | Ética |
|---|---|---|---|---|---|
| **Social:** productores, comerciantes y consumidores | Un criterio de calidad uniforme y reproducible, que reduce las discusiones de precio entre quien vende y quien compra. El consumidor recibe menos fruta dañada. | El 5.0 % de las frutas malas del test se aprueba como buena, casi siempre con alta confianza. Además, el sistema exige un celular y, en su versión actual, un computador que ejecute la aplicación, lo que excluye a quien no los tiene. | Presentar el sistema como apoyo a la inspección y no como certificación. Interfaz en español que solo pide una foto por fruta *(pendiente)*. | Recall de `mala` en fruta local; proporción de usuarios que completan una clasificación sin ayuda. | E8 |
| **Social:** clasificadores manuales | Menos trabajo repetitivo. Un criterio de referencia para capacitar a personas nuevas en la clasificación. | Sustitución de puestos de trabajo, muchas veces de personas con pocas alternativas laborales. | La decisión final queda en la persona. El sistema aumenta la fruta que cada clasificador revisa, en lugar de reemplazarlo, y el personal puede pasar a tareas de control de calidad y supervisión. | Fruta revisada por persona y por hora, sin reducción de personal. | E12 |
| **Económica:** costo de operación | Clasificación más rápida: unas 3 imágenes por segundo en un procesador común. Costo de entrada bajo: código abierto (licencia MIT), sin GPU y con un celular que el usuario ya tiene. La fruta `regular` podría venderse con descuento en lugar de desecharse. | Los errores tienen costo: el 2.4 % de las frutas buenas del test se rechaza (ingreso perdido) y el 5.0 % de las malas se aprueba (reclamos y devoluciones). El error no es igual entre frutas: 10.4 % en manzana buena (E3). | Ajustar el umbral de decisión según el costo de cada tipo de error, que es mayor para la fruta mala aprobada (fase 1). Reportar el desempeño por fruta y validar con fruta local antes de cualquier uso comercial *(pendiente)*. | Costo esperado por lote, $c_{FN} \cdot FN + c_{FP} \cdot FP$, medido por fruta. | E3, E8 |
| **Económica:** poder de negociación | El productor puede verificar con la misma herramienta la clasificación que le asigna el comprador. | El comprador podría usar la herramienta para justificar castigos de precio. La clase de tamaño acierta solo el 55.6 % de las veces, y la medida en cm tiene un error medio de 0.98 cm. | Mostrar siempre la incertidumbre de cada resultado *(pendiente)*. No usar la clase de tamaño para fijar precios hasta definirla con una norma de calibre (NTC del ICONTEC). El código abierto permite que ambas partes usen la misma herramienta. | Diferencia entre la clase asignada por el sistema y la de un inspector, por fruta. | E10 |
| **Ambiental:** pérdida de alimentos | Separar a tiempo la fruta mala evita que contamine el resto del lote. La clase `regular` permite enviar fruta con defectos leves a la agroindustria (pulpas, jugos) en lugar de desecharla. | Cada fruta buena rechazada por error es alimento desperdiciado. Si el umbral prioriza demasiado la detección de fruta mala, el desperdicio aumenta. | Umbrales de la clase `regular` elegidos con recortes etiquetados, para que la fruta dudosa vaya a procesamiento y no a la basura. Con los umbrales actuales, la mayor parte de la fruta mala mal clasificada cae en `regular` y no en `buena`. | Kilogramos enviados a procesamiento frente a kilogramos desechados. | E9 |
| **Ambiental:** huella computacional | Modelos pequeños: XGBoost ocupa 1.2 MB y la CNN 0.3 MB (72 434 parámetros). Todo el entrenamiento se hizo en procesador, sin GPU: unos 11 minutos de búsqueda de hiperparámetros para los modelos clásicos y unos 104 minutos para las tres configuraciones de la CNN. | El entrenamiento y la descarga del dataset (3.3 GB) consumen energía, y cada reentrenamiento repite ese consumo. | Los modelos finales se versionan en el repositorio para usarlos sin reentrenar. La inferencia funciona en equipos existentes, sin comprar hardware nuevo. | Tiempo de procesador por entrenamiento y por imagen. | — |
| **Global:** Objetivos de Desarrollo Sostenible | Contribuye a la meta 12.3 de los ODS (reducir las pérdidas de alimentos en las cadenas de producción y suministro, incluidas las posteriores a la cosecha) y al ODS 2 (hambre cero). | Un solo proyecto académico no mueve indicadores nacionales. Atribuirle ese efecto sin medirlo sería exagerar su alcance. | Presentar la contribución como una herramienta de apoyo cuyo efecto debe medirse en campo, con los indicadores de esta matriz. | Pérdida en poscosecha antes y después del uso, en un piloto. | E11 |
| **Global:** transferencia a otros contextos | Código abierto y documentado por fases CRISP-DM, que otros grupos, regiones o frutas pueden reutilizar. | El dataset proviene de India. Usar el modelo en otros países o variedades sin validarlo traslada ese sesgo. Además, el sistema supone una fruta sobre fondo uniforme y solo conoce seis frutas. | Compartir fotos y anotaciones entre grupos, sin metadatos de ubicación (E7). Evaluar por origen de los datos: hecho, con una caída de F1-macro de 0.62 en recortes de Kaggle a 0.25 en fotos propias. Documentar las condiciones de uso: una fruta, fondo uniforme y una de las seis frutas del dataset. | F1-macro por origen de los datos (Kaggle frente a fotos propias). | E2, E4 |

### Balance

- El impacto neto es positivo **si el sistema se usa como apoyo** a la clasificación: reduce la subjetividad, cuesta poco y puede disminuir la pérdida de fruta en poscosecha.
- Los riesgos principales son dos:
  - los errores con alta confianza (E8);
  - la falta de validación con fruta local (E2).
- La evaluación en las fotos propias confirmó el segundo riesgo: el modelo no generaliza a fruta local fotografiada con otros celulares. Hasta reentrenarlo con fruta local y validarlo con fotos nuevas, el sistema solo debe usarse como demostración académica, no para decisiones comerciales.

### Fuentes de la matriz

- Departamento Nacional de Planeación (DNP), *Pérdida y desperdicio de alimentos en Colombia: estudio de la Dirección de Seguimiento y Evaluación de Políticas Públicas*, Bogotá, 2016.
- FAO, *El estado mundial de la agricultura y la alimentación 2019: progresos en la lucha contra la pérdida y el desperdicio de alimentos*, Roma, 2019.
- Naciones Unidas, *Agenda 2030 para el Desarrollo Sostenible*, meta 12.3 y ODS 2, 2015.
- Cifras propias: errores y desempeño por fruta en `docs/fases/05_evaluacion.md` (test, 2 540 imágenes); tiempos de entrenamiento en `docs/fases/04_modelado.md`; tiempos de inferencia en `docs/arquitectura.md` §6; tamaño en `notebooks/06_tamano.ipynb`.
