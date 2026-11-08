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
        if d > dist.get(u, float("inf")):
            continue
        for v, peso in grafo.get(u, []):
            nueva = d + peso
            if nueva < dist.get(v, float("inf")):
                dist[v] = nueva
                prev[v] = u
                heapq.heappush(pq, (nueva, v))
    return dist, prev

def calcular_secuencia(perfil, aprobadas, notas=None, recursadas=None):
    if notas is None:
        notas = {}
    if recursadas is None:
        recursadas = []

    plan = cargar_plan()
    pesos = construir_pesos(perfil, notas, recursadas)
    aprobadas_set = set(aprobadas)
    grafo = construir_grafo(plan, pesos, aprobadas_set)
    dist, prev = dijkstra(grafo, INICIO)

    materias_map = {m["id"]: m for m in plan["materias"]}
    pendientes_pool = [mid for mid in dist if mid != INICIO and mid not in aprobadas_set]

    validas = set(aprobadas_set)
    final = []

    changed = True
    while changed:
        changed = False
        for mid in sorted(pendientes_pool, key=lambda x: dist[x]):
            if mid in validas or mid in final:
                continue
            m = materias_map[mid]
            if all(c in validas for c in m["correlativas"]):
                final.append(mid)
                nota = notas.get(mid)
                if nota is None or nota >= 6:
                    validas.add(mid)
                changed = True

    final.sort(key=lambda mid: dist[mid])

    return {
        "pesos": pesos,
        "secuencia": final,
        "distancias": {mid: float(dist[mid]) for mid in final},
        "riesgo_total": float(sum(dist[mid] for mid in final)),
    }