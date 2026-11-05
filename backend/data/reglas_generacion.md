# Reglas de generación del dataset sintético

## Variables por estudiante
- `promedio_previo` ~ Normal(6.5, 1.2), truncado a [2, 10]. Fijo por estudiante.

## Variables por cursada (estudiante, materia)
- `nivel`: nivel de la materia según `plan_estudios.json`.
- `cant_materias_aprobadas_previas`: ~ Uniforme(0, 4 * (nivel - 1)).
- `cant_materias_cursando_simultaneo`: ~ Uniforme(2, 6).
- `nota_promedio_correlativas`:
    - si la materia tiene correlativas: Normal(promedio_previo - 0.5, 0.8), truncado a [2, 10];
    - si no tiene correlativas: igual a `promedio_previo`.
- `es_recursada`: Bernoulli(0.15).

## Resultado (ground truth)