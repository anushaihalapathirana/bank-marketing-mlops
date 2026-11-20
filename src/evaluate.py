
import os
import json
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

# Load the selected model
model_uri = os.environ["MODEL_URI"]
model = mlflow.sklearn.load_model(model_uri)

# Load dataset
df = pd.read_csv(
    ROOT / "data/raw/bank-additional-full.csv",
    sep=";"
)

X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

# Same chronological split as training
n = len(X)
val_end = int(n * 0.85)

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

# Final predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "pr_auc": average_precision_score(
        y_test, y_prob
    ),
    "positive_rate": float(y_test.mean()),
    "n_test": len(y_test),
}

print("Final test metrics:")
print(json.dumps(metrics, indent=2))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save results
output = ROOT / "reports"
output.mkdir(exist_ok=True)

with open(output / "test_metrics.json", "w") as f:
    json.dump(
        {"model_uri": model_uri, "metrics": metrics},
        f,
        indent=2
    )
