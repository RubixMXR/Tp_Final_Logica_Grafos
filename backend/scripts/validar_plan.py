import json
from pathlib import Path
import networkx as nx

PLAN = Path(__file__).parent.parent / "data" / "plan_estudios.json"

plan = json.load(open(PLAN, encoding="utf-8"))
ids = {m["id"] for m in plan["materias"]}
assert len(ids) == len(plan["materias"]), "IDs duplicados"

G = nx.DiGraph()
for m in plan["materias"]:
    G.add_node(m["id"], **m)
    for c in m["correlativas"]:
        assert c in ids, f"{m['id']} tiene correlativa inexistente: {c}"
        G.add_edge(c, m["id"])

assert nx.is_directed_acyclic_graph(G), "¡Hay un ciclo en las correlatividades!"

for m in plan["materias"]:
    for c in m["correlativas"]:
        corr = next(x for x in plan["materias"] if x["id"] == c)
        assert corr["nivel"] <= m["nivel"], (
            f"{m['id']} (nivel {m['nivel']}) depende de {c} (nivel {corr['nivel']})"
        )

grados = [len(m["correlativas"]) for m in plan["materias"]]
print(f"OK: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas, DAG válido")
print(f"Correlativas por materia: min={min(grados)}, max={max(grados)}, avg={sum(grados)/len(grados):.2f}")