"""Feature metadata and preprocessing builders."""

from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

# Columns engineered by process_data.py
NUMERIC_FEATURES: List[str] = [
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
]
PASSTHROUGH_FEATURES: List[str] = [
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
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + PASSTHROUGH_FEATURES


# Columns that should use RobustScaler due to heavier tails / outliers
# (identified via IQR analysis in process_data.log_scaling_strategy).
ROBUST_NUMERIC_FEATURES: List[str] = [
    "days_since_last_donation",
    "donation_count_dhoby_ghaut",
    "donation_count_outram",
    "donation_count_woodlands",
    "donations_per_year",
    "mean_donation_interval_days",
    "recency_ratio",
]

STANDARD_NUMERIC_FEATURES: List[str] = [
    col for col in NUMERIC_FEATURES if col not in ROBUST_NUMERIC_FEATURES
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected feature columns: {missing}")


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Create a ColumnTransformer that optionally scales numeric columns."""

    transformers = []
    if scale_numeric and NUMERIC_FEATURES:
        if ROBUST_NUMERIC_FEATURES:
            transformers.append(
                ("num_robust", RobustScaler(), ROBUST_NUMERIC_FEATURES)
            )
        if STANDARD_NUMERIC_FEATURES:
            transformers.append(
                ("num_standard", StandardScaler(), STANDARD_NUMERIC_FEATURES)
            )

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
