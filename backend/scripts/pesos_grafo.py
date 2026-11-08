import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE  = Path(__file__).parent.parent
PLAN  = BASE / "data" / "plan_estudios.json"
MODEL = BASE / "models" / "modelo.pkl"

NUM_FEATS = [
    "promedio_previo",
    "cant_materias_aprobadas_previas",
    "cant_materias_cursando_simultaneo",
    "nota_promedio_correlativas",
    "nivel",
]
CAT_FEATS = ["es_recursada"]
EPS = 1e-6

def cargar_plan():
    return json.load(open(PLAN, encoding="utf-8"))

def armar_perfil_desde_notas(notas: dict, cant_simult: int) -> dict:
    promedio = float(np.mean(list(notas.values()))) if notas else 6.5
    aprobadas_prev = len([n for n in notas.values() if n >= 6])

    return {
        "promedio_previo": promedio,
        "cant_materias_aprobadas_previas": aprobadas_prev,
        "cant_materias_cursando_simultaneo": cant_simult,
    }

def predecir_probabilidades(perfil: dict, plan: dict, notas: dict, recursadas_set: set):
    modelo = joblib.load(MODEL)
    filas = []
    for m in plan["materias"]:
        mid = m["id"]
        corrs = m["correlativas"]
        
        notas_ajustadas = []
        for c in corrs:
            if c in notas:
                nota = notas[c]
                if c in recursadas_set and nota >= 6:
                    nota = nota - 1.5
                notas_ajustadas.append(nota)
        
        nota_corr = float(np.mean(notas_ajustadas)) if notas_ajustadas else perfil["promedio_previo"]

        es_recur = 1 if (mid in recursadas_set or (mid in notas and notas[mid] < 6)) else 0

        fila = {
            **perfil,
            "nivel": m["nivel"],
            "nota_promedio_correlativas": nota_corr,
            "es_recursada": es_recur,
        }
        filas.append(fila)

    X = pd.DataFrame(filas)[NUM_FEATS + CAT_FEATS]
    probs = modelo.predict_proba(X)[:, 1]
    return {m["id"]: float(p) for m, p in zip(plan["materias"], probs)}

def probabilidad_a_peso(p: float) -> float:
    return float(-np.log(max(p, EPS)))

def construir_pesos(perfil: dict, notas: dict, recursadas: list):
    plan = cargar_plan()
    recursadas_set = set(recursadas)
    probs = predecir_probabilidades(perfil, plan, notas, recursadas_set)
    return {
        mid: {
            "p_aprobar": probs[mid],
            "peso": probabilidad_a_peso(probs[mid]),
        }
        for mid in probs
    }