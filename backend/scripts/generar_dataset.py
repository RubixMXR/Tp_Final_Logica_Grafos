import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_ESTUDIANTES = 500
P_CURSAR = 0.85

BASE = Path(__file__).parent.parent
PLAN = BASE / "data" / "plan_estudios.json"
OUT  = BASE / "data" / "cursadas.csv"

rng = np.random.default_rng(SEED)
plan = json.load(open(PLAN, encoding="utf-8"))
materias = plan["materias"]
dificultad = {m["id"]: m["dificultad"] for m in materias}
por_nivel  = {n: [m for m in materias if m["nivel"] == n] for n in range(1, 6)}

filas = []
for i in range(N_ESTUDIANTES):
    eid = f"E{i:04d}"
    promedio_previo = float(np.clip(rng.normal(6.5, 1.2), 2, 10))
    
    # Ahora guardamos las notas de las materias aprobadas, no solo el set
    notas_aprobadas = {} 

    for nivel in range(1, 6):
        for m in por_nivel[nivel]:
            # Verificamos que tenga todas las correlativas aprobadas
            if not all(c in notas_aprobadas for c in m["correlativas"]):
                continue
            if rng.random() > P_CURSAR:
                continue

            cant_simult = int(rng.integers(2, 7))
            recursada = int(rng.random() < 0.15)
            aprobadas_prev = len(notas_aprobadas)

            # Calculamos la nota promedio de las correlativas
            if m["correlativas"]:
                notas_corr = [notas_aprobadas[c] for c in m["correlativas"]]
                nota_corr = float(np.mean(notas_corr))
            else:
                nota_corr = promedio_previo

            # Fórmula del ground truth
            logit = ( 0.6
                    + 0.7 * (promedio_previo - 6)
                    + 0.4 * (nota_corr - 6)
                    - 0.10 * (cant_simult - 4)
                    - 0.80 * recursada
                    + 0.05 * aprobadas_prev
                    - 0.15 * (dificultad[m["id"]] - 5) )
            p = 1.0 / (1.0 + np.exp(-logit))

            r = rng.random()
            if r < p:
                resultado = "aprobó"
                # Si aprueba, le asignamos una nota final (entre 6 y 10)
                # Un alumno con promedio 7 tenderá a sacarse 7.5
                nota_final = float(np.clip(rng.normal(promedio_previo + 0.5, 1.0), 6, 10))
                notas_aprobadas[m["id"]] = nota_final
            elif r < p + 0.10 * (1 - p):
                resultado = "abandonó"
            else:
                resultado = "desaprobó"

            # Columnas exactas que pide el enunciado + nivel y dificultad
            filas.append({
                "estudiante_id": eid,
                "materia_id": m["id"],
                "promedio_previo": round(promedio_previo, 2),
                "cant_materias_aprobadas_previas": aprobadas_prev,
                "cant_materias_cursando_simultaneo": cant_simult,
                "nota_promedio_correlativas": round(nota_corr, 2),
                "es_recursada": recursada,
                "resultado": resultado,
                "nivel": nivel,                 # Extra
                "dificultad": m["dificultad"],  # Extra
            })

df = pd.DataFrame(filas)

# Forzamos el orden de las columnas como pide la tabla (más las extras)
columnas_ordenadas = [
    "estudiante_id", "materia_id", "promedio_previo", 
    "cant_materias_aprobadas_previas", "cant_materias_cursando_simultaneo", 
    "nota_promedio_correlativas", "es_recursada", "resultado",
    "nivel", "dificultad"
]
df = df[columnas_ordenadas]

df.to_csv(OUT, index=False, encoding="utf-8")

print(f"Filas generadas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(f"Distribución de resultado:\n{df['resultado'].value_counts(normalize=True).round(3)}")