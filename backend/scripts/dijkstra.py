import heapq
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pesos_grafo import cargar_plan, construir_pesos

INICIO = "__INICIO__"

def construir_grafo(plan, pesos, aprobadas):
    grafo = {INICIO: []}
    for m in plan["materias"]:
        grafo.setdefault(m["id"], [])

    for m in plan["materias"]:
        mid = m["id"]
        if all(c in aprobadas for c in m["correlativas"]) and mid not in aprobadas:
            grafo[INICIO].append((mid, pesos[mid]["peso"]))
        for c in m["correlativas"]:
            grafo[c].append((mid, pesos[mid]["peso"]))
    return grafo

def dijkstra(grafo, inicio):
    dist = {inicio: 0.0}
    prev = {}
    pq = [(0.0, inicio)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")): continue
        for v, peso in grafo.get(u, []):
            nueva = d + peso
            if nueva < dist.get(v, float("inf")):
                dist[v] = nueva
                prev[v] = u
                heapq.heappush(pq, (nueva, v))
    return dist, prev

def calcular_secuencia(perfil, aprobadas):
    plan = cargar_plan()
    pesos = construir_pesos(perfil)
    aprobadas_set = set(aprobadas)
    grafo = construir_grafo(plan, pesos, aprobadas_set)

    dist, prev = dijkstra(grafo, INICIO)

    pendientes = []
    for m in plan["materias"]:
        mid = m["id"]
        if mid in aprobadas_set or mid not in dist:
            continue

        # LA CORRECCIÓN: Validar que TODAS las correlativas estén aprobadas 
        # O tengan una distancia menor (lo que significa que aparecen antes en la secuencia)
        valid = True
        for c in m["correlativas"]:
            if c in aprobadas_set:
                continue
            if c in dist and dist[c] < dist[mid]:
                continue
            valid = False
            break

        if valid:
            pendientes.append(mid)

    pendientes.sort(key=lambda mid: dist[mid])

    return {
        "pesos": pesos,
        "secuencia": pendientes,
        "distancias": {mid: float(dist[mid]) for mid in pendientes},
        "riesgo_total": float(sum(dist[mid] for mid in pendientes)),
    }

if __name__ == "__main__":
    notas = {"AMI": 8, "AGA": 7, "SPN": 6, "AED": 9, "ING1": 8}
    from pesos_grafo import armar_perfil_desde_notas
    perfil = armar_perfil_desde_notas(notas, cant_simult=4, recursada=0)
    res = calcular_secuencia(perfil, list(notas.keys()))
    print("Secuencia corregida:")
    for i, mid in enumerate(res["secuencia"][:10], 1):
        print(f"  {i:2d}. {mid:5s}  riesgo = {res['distancias'][mid]:.3f}")