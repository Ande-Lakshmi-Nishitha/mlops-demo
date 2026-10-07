"""Stage 3: evaluate on the held-out set, log to MLflow, enforce quality gate."""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    test = pd.read_csv("data/processed/test.csv")
    X, y = test.drop(columns="target"), test["target"]
    model = joblib.load("models/model.joblib")
    preds = model.predict(X)

    metrics = {
        "test_accuracy": float(accuracy_score(y, preds)),
        "test_f1_macro": float(f1_score(y, preds, average="macro")),
    }

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/scores.json").write_text(json.dumps(metrics, indent=2))

    # attach metrics to the same MLflow run created by train.py
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run_id = Path("models/run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("metrics/scores.json")

    print(metrics)
    if metrics["test_accuracy"] < params["min_accuracy"]:
        print(
            f"QUALITY GATE FAILED: {metrics['test_accuracy']:.3f} "
            f"< {params['min_accuracy']}"
        )
        sys.exit(1)
    print("Quality gate passed.")


if __name__ == "__main__":
    main()
