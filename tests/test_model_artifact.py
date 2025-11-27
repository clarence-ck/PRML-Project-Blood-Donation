"""Smoke tests for the persisted best model artifact."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from tests.sample_data import load_sample_frame

MODEL_PATH = Path("models/best_model.pkl")


def test_best_model_exists() -> None:
    assert MODEL_PATH.exists(), "models/best_model.pkl is missing; run training first."


def test_best_model_predicts() -> None:
    X, y = load_sample_frame()
    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.all((probs >= 0) & (probs <= 1))
    else:
        pytest.fail("Model pipeline lacks predict_proba, which the API requires.")
