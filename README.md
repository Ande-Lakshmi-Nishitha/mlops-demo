# MLOps demo: CI for an ML model (Git + GitHub Actions + DVC + MLflow)

A small neural network (scikit-learn `MLPClassifier`) that classifies handwritten digits,
wrapped in a reproducible pipeline with automated testing and a CI quality gate.

## Layout

| Path | Purpose |
|---|---|
| `src/prepare.py`, `train.py`, `evaluate.py` | Pipeline stages |
| `params.yaml` | All hyperparameters + the accuracy gate (`evaluate.min_accuracy`) |
| `dvc.yaml` / `dvc.lock` | Pipeline definition and the exact hashes of each run |
| `tests/` | pytest: data validation and model sanity tests |
| `.github/workflows/ci.yml` | GitHub Actions: lint, test, `dvc repro`, quality gate |
| `mlflow.db` | Local MLflow tracking store (git-ignored) |

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc repro                      # prepare -> train -> evaluate
pytest -v
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Experiment workflow

```bash
git checkout -b exp/wider-network
# edit params.yaml, e.g. hidden_layers: [128, 64]
dvc repro
dvc metrics diff               # compare against the last commit
git add -A && git commit -m "Try wider network"
git push -u origin exp/wider-network   # open a PR; CI runs automatically
```

## Quality gate

`evaluate.py` exits with code 1 if test accuracy < `evaluate.min_accuracy`.
That fails `dvc repro` in CI, which blocks the PR from merging.
