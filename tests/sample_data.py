"""Utility helpers for loading sample feature rows in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.features.preprocess import FEATURE_COLUMNS

PROCESSED_PATH = Path("processed_donor_features.csv")
TARGET_COLUMN = "target_available"


def _load_processed_subset(n_rows: int = 5) -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            "processed_donor_features.csv not found. Run process_data.py before tests."
        )
    return pd.read_csv(PROCESSED_PATH, nrows=n_rows)


def load_sample_features() -> Dict[str, object]:
    """Return a dict of model-ready features for API payload tests."""

    df = _load_processed_subset(n_rows=1)
    row = df.iloc[0]
    features = {}
    for col in FEATURE_COLUMNS:
        value = row[col]
        if hasattr(value, "item"):
            value = value.item()
        features[col] = value
    return features


def load_sample_frame() -> Tuple[pd.DataFrame, pd.Series]:
    """Return a small feature matrix and target series for artifact tests."""

    df = _load_processed_subset(n_rows=10)
    if TARGET_COLUMN not in df.columns:
        raise ValueError("processed dataset missing target_available column")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return X, y
