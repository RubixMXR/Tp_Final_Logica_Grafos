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
    aprobadas = set()

    for nivel in range(1, 6):
        for m in por_nivel[nivel]:
            if not all(c in aprobadas for c in m["correlativas"]):
                continue
            if rng.random() > P_CURSAR:
                continue

            cant_simult = int(rng.integers(2, 7))
            recursada = int(rng.random() < 0.15)
            aprobadas_prev = len(aprobadas)
            
            if m["correlativas"]:
                nota_corr = float(np.clip(rng.normal(promedio_previo - 0.5, 0.8), 2, 10))
            else:
                nota_corr = promedio_previo

            logit = ( 0.9 * (promedio_previo - 6)
                    + 0.7 * (nota_corr - 6)
                    - 0.4 * (cant_simult - 4)
                    - 0.8 * recursada
                    + 0.05 * aprobadas_prev
                    - 0.3 * (dificultad[m["id"]] - 5) )
            p = 1.0 / (1.0 + np.exp(-logit))

            r = rng.random()
            if r < p:
                resultado = "aprobó"
                aprobadas.add(m["id"])
            elif r < p + 0.10 * (1 - p):
                resultado = "abandonó"
            else:
                resultado = "desaprobó"

            filas.append({
                "estudiante_id": eid,
                "materia_id": m["id"],
                "nivel": nivel,
                "promedio_previo": round(promedio_previo, 2),
                "cant_materias_aprobadas_previas": aprobadas_prev,
                "cant_materias_cursando_simultaneo": cant_simult,
                "nota_promedio_correlativas": round(nota_corr, 2),
                "es_recursada": recursada,
                "resultado": resultado,
            })

df = pd.DataFrame(filas)
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"Filas generadas: {len(df)}")