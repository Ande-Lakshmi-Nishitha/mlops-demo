"""Stage 2: train an MLP classifier and log everything to MLflow."""
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    train = pd.read_csv("data/processed/train.csv")
    X, y = train.drop(columns="target"), train["target"]

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layers"]),
            learning_rate_init=params["learning_rate"],
            max_iter=params["max_iter"],
            random_state=params["seed"],
        ),
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("digits-mlp")
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(
            {**params, "hidden_layers": str(params["hidden_layers"])}
        )
        model.fit(X, y)
        mlp = model[-1]
        mlflow.log_metric("train_accuracy", model.score(X, y))
        mlflow.log_metric("final_loss", float(mlp.loss_))
        for step, loss in enumerate(mlp.loss_curve_):
            mlflow.log_metric("loss", float(loss), step=step)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        mlflow.log_artifact("models/model.joblib")
        Path("models/run_id.txt").write_text(run.info.run_id)
        print("run_id:", run.info.run_id)


if __name__ == "__main__":
    main()
