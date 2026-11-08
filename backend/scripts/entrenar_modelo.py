import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE    = Path(__file__).parent.parent
CSV     = BASE / "data" / "cursadas.csv"
MODEL   = BASE / "models" / "modelo.pkl"
METRICS = BASE / "models" / "metricas.json"

df = pd.read_csv(CSV, encoding="utf-8")
df["target"] = (df["resultado"] == "aprobó").astype(int)

num_feats = [
    "promedio_previo",
    "cant_materias_aprobadas_previas",
    "cant_materias_cursando_simultaneo",
    "nota_promedio_correlativas",
    "nivel",
]
cat_feats = ["es_recursada"]

X = df[num_feats + cat_feats]
y = df["target"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(Xtr, ytr)

yhat = pipe.predict(Xte)

metricas = {
    "n_train": int(len(Xtr)),
    "n_test":  int(len(Xte)),
    "accuracy":  float(accuracy_score(yte, yhat)),
    "precision": float(precision_score(yte, yhat)),
    "recall":    float(recall_score(yte, yhat)),
    "f1":        float(f1_score(yte, yhat)),
    "confusion_matrix": confusion_matrix(yte, yhat).tolist(),
}
joblib.dump(pipe, MODEL)
METRICS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Modelo guardado en {MODEL}")
print(json.dumps(metricas, indent=2))