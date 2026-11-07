from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from pesos_grafo import cargar_plan, construir_pesos, armar_perfil_desde_notas
from dijkstra import calcular_secuencia

app = FastAPI(title="Planificador Curricular API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PerfilIn(BaseModel):
    notas: dict[str, float] = {}   # materia_id -> nota
    cant_materias_cursando_simultaneo: int = 4
    es_recursada: int = 0

def armar(perfil_in: PerfilIn):
    perfil = armar_perfil_desde_notas(
        perfil_in.notas,
        perfil_in.cant_materias_cursando_simultaneo,
        perfil_in.es_recursada,
    )
    aprobadas = list(perfil_in.notas.keys())
    return perfil, aprobadas

@app.get("/api/plan")
def get_plan():
    return cargar_plan()

@app.post("/api/pesos")
def post_pesos(perfil_in: PerfilIn):
    perfil, _ = armar(perfil_in)
    return construir_pesos(perfil)

@app.post("/api/secuencia")
def post_secuencia(perfil_in: PerfilIn):
    perfil, aprobadas = armar(perfil_in)
    return calcular_secuencia(perfil, aprobadas)