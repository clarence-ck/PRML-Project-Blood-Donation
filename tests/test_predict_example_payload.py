from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import EXPECTED_FEATURES, app


client = TestClient(app)


def test_predict_example_json_payload_round_trip() -> None:
    """Ensure predict_example.json stays in sync with the API schema and works end-to-end."""

    path = Path("predict_example.json")
    assert path.exists(), "predict_example.json is missing; keep the example file in the repo."

    payload = json.loads(path.read_text())
    assert "features" in payload, "predict_example.json must contain a 'features' object."

    features = payload["features"]
    missing = [col for col in EXPECTED_FEATURES if col not in features]
    assert not missing, f"predict_example.json missing expected feature columns: {missing}"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "probability" in body and 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
