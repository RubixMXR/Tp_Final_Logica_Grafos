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
    "nivel",
]
CAT_FEATS = ["es_recursada"]
EPS = 1e-6

def cargar_plan():
    return json.load(open(PLAN, encoding="utf-8"))

def armar_perfil_desde_notas(notas: dict, cant_simult: int, recursadas: list) -> dict:
    promedio = float(np.mean(list(notas.values()))) if notas else 6.5
    aprobadas_prev = len([n for n in notas.values() if n >= 6])
    
    tiene_desaprobadas = any(n < 6 for n in notas.values())
    es_recursada = 1 if (recursadas or tiene_desaprobadas) else 0

    return {
        "promedio_previo": promedio,
        "cant_materias_aprobadas_previas": aprobadas_prev,
        "cant_materias_cursando_simultaneo": cant_simult,
        "es_recursada": es_recursada,
    }

def predecir_probabilidades(perfil: dict, plan: dict):
    modelo = joblib.load(MODEL)
    filas = []
    for m in plan["materias"]:
        fila = {**perfil, "nivel": m["nivel"]}
        filas.append(fila)
    X = pd.DataFrame(filas)[NUM_FEATS + CAT_FEATS]
    probs = modelo.predict_proba(X)[:, 1]
    return {m["id"]: float(p) for m, p in zip(plan["materias"], probs)}

def probabilidad_a_peso(p: float) -> float:
    return float(-np.log(max(p, EPS)))

def construir_pesos(perfil: dict):
    plan = cargar_plan()
    probs = predecir_probabilidades(perfil, plan)
    return {
        mid: {
            "p_aprobar": probs[mid],
            "peso": probabilidad_a_peso(probs[mid]),
        }
        for mid in probs
    }

if __name__ == "__main__":
    notas = {"AMI": 8, "AGA": 7, "SPN": 6, "AED": 9, "ING1": 8}
    perfil = armar_perfil_desde_notas(notas, cant_simult=4, recursadas=[])
    print("Perfil derivado:", perfil)
    pesos = construir_pesos(perfil)
    for mid, v in list(pesos.items())[:5]:
        print(f"{mid}: P={v['p_aprobar']:.3f}  peso={v['peso']:.3f}")