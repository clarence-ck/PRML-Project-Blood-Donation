"""FastAPI wrapper that serves the trained donor-return model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
PREDICTION_THRESHOLD = 0.5


# Mirror FEATURE_COLUMNS from src.features.preprocess to keep inference schema aligned.
EXPECTED_FEATURES: list[str] = [
    "age",
    "days_since_last_donation",
    "donor_tenure_days",
    "mean_donation_interval_days",
    "last_donation_volume_ml",
    "avg_donation_volume",
    "donation_count_outram",
    "donation_count_dhoby_ghaut",
    "donation_count_woodlands",
    "donation_count_jurong_east",
    "donation_count_punggol",
    "recency_ratio",
    "volume_consistency",
    "primary_location_pct",
    "donations_per_year",
    "risk_recency",
    "education_level_encoded",
    "gender_Male",
    "had_adverse_reaction_ever",
    "last_blood_quality_screen",
    "blood_group_A-",
    "blood_group_AB+",
    "blood_group_AB-",
    "blood_group_B+",
    "blood_group_B-",
    "blood_group_O+",
    "blood_group_O-",
]


class PredictionRequest(BaseModel):
    """Incoming payload containing model-ready features."""

    features: Dict[str, Union[float, int, bool]] = Field(
        ...,
        description=(
            "Mapping of feature name to numeric value. Must include the same columns "
            "produced by process_data.py"
        ),
    )


class PredictionResponse(BaseModel):
    probability: float
    prediction: int


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best_model.pkl not found. Ensure models/best_model.pkl exists before starting the API."
        )
    return joblib.load(MODEL_PATH)


MODEL = _load_model()
app = FastAPI(title="Blood Donor Return API", version="1.0.0")


def _validate_and_frame(features: Dict[str, Union[float, int, bool]]) -> pd.DataFrame:
    missing = [col for col in EXPECTED_FEATURES if col not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_features", "details": missing},
        )

    ordered_row = []
    for col in EXPECTED_FEATURES:
        value = features[col]
        if isinstance(value, bool):
            ordered_row.append(int(value))
        else:
            ordered_row.append(value)

    return pd.DataFrame([ordered_row], columns=EXPECTED_FEATURES)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    df = _validate_and_frame(request.features)
    try:
        proba = float(MODEL.predict_proba(df)[0, 1])
    except AttributeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Model does not support predict_proba") from exc

    prediction = int(proba >= PREDICTION_THRESHOLD)
    return PredictionResponse(probability=proba, prediction=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
