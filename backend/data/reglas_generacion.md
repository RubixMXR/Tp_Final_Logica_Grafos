# Reglas de generación del dataset sintético

## Variables
El dataset incluye estrictamente las columnas solicitadas en el enunciado, más dos variables extra (`nivel` y `dificultad`) que aportan contexto sobre la materia:

- `estudiante_id`: identificador del estudiante.
- `materia_id`: identificador de la materia.
- `promedio_previo`: promedio general del estudiante hasta ese momento (se asigna al inicio de su cursada).
- `cant_materias_aprobadas_previas`: cantidad de materias que el estudiante ya aprobó hasta el momento.
- `cant_materias_cursando_simultaneo`: cantidad de materias que está cursando en ese mismo período (~ Uniforme(2, 6)).
- `nota_promedio_correlativas`: promedio de las notas obtenidas en las materias correlativas.
- `es_recursada`: booleano (0 o 1) que indica si el estudiante ya había cursado y desaprobado la materia (Bernoulli(0.15)).
- `resultado`: variable a predecir (aprobó / desaprobó / abandonó).
- `nivel`: nivel de la materia según `plan_estudios.json` (1 a 5). *(Extra)*
- `dificultad`: propiedad fija de cada materia (1–10), definida en `plan_estudios.json`. *(Extra)*

## Generación del historial y notas
Para que la columna `nota_promedio_correlativas` tenga sentido real (y coincida con lo que el usuario carga en el frontend), el script simula el avance del estudiante guardando las notas de las materias que aprueba:

1. Cada estudiante arranca con un `promedio_previo` base ~ Normal(6.5, 1.2), truncado a [2, 10].
2. Cuando un estudiante aprueba una materia, se le asigna una `nota_final` ~ Normal(`promedio_previo` + 0.5, 1.0), truncada al rango [6, 10].
3. Cuando el estudiante avanza a una nueva materia, `nota_promedio_correlativas` se calcula como el promedio exacto de las `nota_final` de sus correlativas.
4. Si la materia no tiene correlativas (ej. materias de 1er nivel), se usa el `promedio_previo` del estudiante como valor por defecto.

## Resultado (ground truth)
La probabilidad latente de aprobar se calcula con la siguiente función logística, que combina el perfil del estudiante y las características de la materia:

```
logit =  0.6
       + 0.7 * (promedio_previo - 6)
       + 0.4 * (nota_promedio_correlativas - 6)
       - 0.1 * (cant_materias_cursando_simultaneo - 4)
       - 0.8 * es_recursada
       + 0.05 * cant_materias_aprobadas_previas
       - 0.15 * (dificultad - 5)

P(aprobar) = sigmoid(logit)
```

Luego, el resultado simulado se define así:
- `aprobó` con probabilidad `P(aprobar)`.
- `abandonó` con probabilidad `0.10 * (1 - P(aprobar))`.
- `desaprobó` con el complemento.

## Notas de diseño
- El intercept base `0.6` fija el punto de partida en ~65% para un estudiante promedio (6.5) cursando una materia de dificultad media (5) con 4 materias simultáneas, evitando que el modelo subestime las probabilidades al inicio.
- El modelo debería recuperar estas relaciones. Se espera que la regresión logística asigne coeficientes positivos a `promedio_previo` y `nota_promedio_correlativas`, y negativos a `dificultad` y `es_recursada`.
- `materia_id` no se usa como feature de entrenamiento (para evitar que el modelo memorice materias específicas en lugar de aprender patrones generales).
- **Plan 2023 UTN FRRo**: se modelaron 31 materias obligatorias + 3 electivas agrupadas por nivel (`ELEC3`, `ELEC4`, `ELEC5`).
- **Simplificación del régimen de correlatividades**: se usa el criterio "para cursar regularizadas". El input del usuario se trata como "aprobadas" (aquellas con nota ≥ 6).