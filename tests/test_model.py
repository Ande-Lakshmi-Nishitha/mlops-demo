from pathlib import Path

import joblib
import pandas as pd
import pytest

MODEL = Path("models/model.joblib")
TEST_CSV = Path("data/processed/test.csv")

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and TEST_CSV.exists()),
    reason="run `dvc repro` first to create the model",
)


def test_model_predicts_valid_labels():
    model = joblib.load(MODEL)
    X = pd.read_csv(TEST_CSV).drop(columns="target")
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds) <= set(range(10))


def test_predict_proba_sums_to_one():
    model = joblib.load(MODEL)
    X = pd.read_csv(TEST_CSV).drop(columns="target").head(20)
    probs = model.predict_proba(X)
    assert probs.shape == (20, 10)
    assert probs.sum(axis=1) == pytest.approx(1.0)
