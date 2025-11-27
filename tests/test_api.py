"""API contract tests for the FastAPI inference service."""

from __future__ import annotations

import json
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from app.main import EXPECTED_FEATURES, app
from tests.sample_data import load_sample_features

client = TestClient(app)


@pytest.fixture(scope="session")
def sample_payload() -> Dict[str, Dict[str, object]]:
    features = load_sample_features()
    missing = [col for col in EXPECTED_FEATURES if col not in features]
    if missing:
        raise AssertionError(
            f"Test setup error: sample row missing expected features: {missing}"
        )
    return {"features": features}


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success(sample_payload: Dict[str, Dict[str, object]]) -> None:
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body and 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_predict_missing_feature(sample_payload: Dict[str, Dict[str, object]]) -> None:
    payload = json.loads(json.dumps(sample_payload))
    first_key = EXPECTED_FEATURES[0]
    payload["features"].pop(first_key)

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "missing_features"
    assert first_key in body["detail"]["details"]
