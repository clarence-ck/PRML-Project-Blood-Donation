"""Feature engineering pipeline for the blood donor dataset.

This script ingests the synthetic donor CSV, removes PII, engineers
model-ready features such as age, donation recency, availability target,
volume metrics, and encodes categorical variables. Running
``python process_data.py`` regenerates ``processed_donor_features.csv``
for downstream notebooks and training scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# ---------------------------------------------------------------------------
# Configuration defaults and schema metadata
# ---------------------------------------------------------------------------
DEFAULT_DATA_FILE = Path("blood_donor_dataset.csv")
DEFAULT_OUTPUT_FILE = Path("processed_donor_features.csv")
REFERENCE_DATE = pd.to_datetime("2025-11-30")

ORIGINAL_COLS = [
    "donor_id",
    "name",
    "email",
    "contact_number",
    "password",
    "date_of_birth",
    "gender",
    "education_level",
    "blood_group",
    "first_donation_date",
    "last_donation_date",
    "number_of_donation",
    "total_donation_volume_ml",
    "last_donation_volume_ml",
    "donation_count_outram",
    "donation_count_dhoby_ghaut",
    "donation_count_woodlands",
    "donation_count_jurong_east",
    "donation_count_punggol",
    "had_adverse_reaction_ever",
    "last_blood_quality_screen",
    "availability",
]
PII_COL_SET = {"donor_id", "name", "email", "contact_number", "password"}
PII_COLS = [col for col in ORIGINAL_COLS if col in PII_COL_SET]
DATE_COLUMNS = ["date_of_birth", "first_donation_date", "last_donation_date"]
TRANSFORMED_COLS = [
    "date_of_birth",
    "first_donation_date",
    "last_donation_date",
    "availability",
    "total_donation_volume_ml",
    "education_level",
    "number_of_donation",
]
COLUMN_ORDER = [
    "age",
    "education_level_encoded",
    "gender_Male",
    "had_adverse_reaction_ever",
    "blood_group_A-",
    "blood_group_AB+",
    "blood_group_AB-",
    "blood_group_B+",
    "blood_group_B-",
    "blood_group_O+",
    "blood_group_O-",
    "days_since_last_donation",
    "donor_tenure_days",
    "mean_donation_interval_days",
    "last_donation_volume_ml",
    "avg_donation_volume",
    "last_blood_quality_screen",
    "donation_count_outram",
    "donation_count_dhoby_ghaut",
    "donation_count_woodlands",
    "donation_count_jurong_east",
    "donation_count_punggol",
]
# Ordinal mapping aligns education levels to increasing integers.
EDUCATION_MAPPING = {
    "Primary": 1,
    "Secondary": 2,
    "Diploma": 3,
    "University": 4,
    "Postgraduate": 5,
}

# ---------------------------------------------------------------------------
# Data loading and exploratory helpers
# ---------------------------------------------------------------------------
def load_dataset(file_path: Path | str) -> pd.DataFrame:
    """Load the donor dataset from disk."""

    return pd.read_csv(file_path)


def log_unique_non_numeric_columns(df: pd.DataFrame) -> None:
    """Print unique values for all non-numeric, non-PII columns."""

    print("--- Unique values for non-numeric columns ---")
    for column in df.columns:
        if column in PII_COLS or column in DATE_COLUMNS:
            continue

        series = df[column]
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            unique_values = series.dropna().unique().tolist()
            print(f"{column}: {unique_values}")


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
def add_date_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Convert date columns and engineer age, recency, and tenure features."""

    df = df.copy()
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column])

    df["age"] = (reference_date - df["date_of_birth"]).dt.days / 365.25
    df["days_since_last_donation"] = (reference_date - df["last_donation_date"]).dt.days
    df["donor_tenure_days"] = (
        df["last_donation_date"] - df["first_donation_date"]
    ).dt.days

    return df


def add_feature_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical and boolean values."""

    df = df.copy()
    df["avg_donation_volume"] = (
        df["total_donation_volume_ml"] / df["number_of_donation"]
    )

    interval_denominator = (df["number_of_donation"] - 1).clip(lower=1)
    df["mean_donation_interval_days"] = (
        df["donor_tenure_days"] / interval_denominator
    )

    df["target_available"] = df["availability"].map({"Yes": True, "No": False})

    df["had_adverse_reaction_ever"] = df["had_adverse_reaction_ever"].astype(int)
    df["last_blood_quality_screen"] = df["last_blood_quality_screen"].map(
        {"Pass": 1, "Fail": 0}
    )
    df["education_level_encoded"] = df["education_level"].map(EDUCATION_MAPPING)

    df = pd.get_dummies(df, columns=["gender", "blood_group"], drop_first=True)

    # Derived behavioral ratios suggested by EDA diagnostics
    tenure_days = df["donor_tenure_days"].astype("float64")
    tenure_days = tenure_days.replace(0.0, np.nan)
    recency_ratio = df["days_since_last_donation"].astype("float64").div(tenure_days)
    df["recency_ratio"] = recency_ratio.clip(lower=0).fillna(0.0)

    avg_volume = df["avg_donation_volume"].astype("float64").replace(0.0, np.nan)
    volume_consistency = df["last_donation_volume_ml"].astype("float64").div(avg_volume)
    df["volume_consistency"] = volume_consistency.clip(lower=0).fillna(0.0)

    location_cols = [
        col for col in df.columns if col.startswith("donation_count_")
    ]
    if location_cols:
        total_locations = df[location_cols].sum(axis=1).astype("float64")
        primary = df[location_cols].max(axis=1).astype("float64")
        total_locations = total_locations.replace(0.0, np.nan)
        df["primary_location_pct"] = (primary / total_locations).fillna(0.0)
    else:
        df["primary_location_pct"] = 0.0

    tenure_years = tenure_days / 365.25
    donations_per_year = df["number_of_donation"].astype("float64").div(tenure_years)
    df["donations_per_year"] = donations_per_year.fillna(0.0)

    df["risk_recency"] = (
        df["had_adverse_reaction_ever"] * df["days_since_last_donation"]
    )

    return df


def process_data(
    file_path: Path | str = DEFAULT_DATA_FILE,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.DataFrame:
    """Load, clean, and transform donor data into model-ready features.
    
    Feature engineering:
    1. Temporal features: age, recency (days since last), tenure (engagement span)
    2. Behavioral features: donation frequency, average volume, interval patterns
    3. Categorical encoding: one-hot for gender/blood group, ordinal for education
    4. Spatial feature: total donations across all locations (simple aggregation)
    """

    df = load_dataset(file_path)
    log_unique_non_numeric_columns(df)

    # Drop PII
    df = df.drop(columns=PII_COLS, errors="ignore")
    
    # Temporal features
    df = add_date_features(df, reference_date)
    
    # Basic encodings + behavioral features
    df = add_feature_encodings(df)
    
    # Drop transformed columns
    df_processed = df.drop(columns=TRANSFORMED_COLS, errors="ignore")

    # Reorder columns
    ordered_cols = [col for col in COLUMN_ORDER if col in df_processed.columns]
    remaining_cols = [col for col in df_processed.columns if col not in ordered_cols]
    if ordered_cols:
        df_processed = df_processed[ordered_cols + remaining_cols]

    # Ensure target column is last
    if "target_available" in df_processed.columns:
        cols = [c for c in df_processed.columns if c != "target_available"]
        df_processed = df_processed[cols + ["target_available"]]

    return df_processed


def save_processed_data(df: pd.DataFrame, output_path: Path | str) -> None:
    """Persist processed features and log a quick summary."""

    df.to_csv(output_path, index=False)
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    print(f"Saved processed data -> {output_path} ({n_rows:,} rows × {n_cols} cols)")
    print(f"Numeric ({len(numeric_cols)}): {', '.join(numeric_cols)}")
    print(f"Boolean ({len(bool_cols)}): {', '.join(bool_cols)}")


def log_scaling_strategy(df: pd.DataFrame) -> None:
    """Print numeric ranges and scaling guidance (post-EDA)."""

    print("\n" + "-" * 60)
    print("📏 FEATURE SCALING STRATEGY (POST-PROCESSING)")

    key_numerics = [
        "number_of_donation",
        "total_donation_volume_ml",
        "days_since_last_donation",
        "donor_tenure_days",
    ]
    available_keys = [c for c in key_numerics if c in df.columns]
    if available_keys:
        print("   Key numeric features (range | mean | median):")
        for col in available_keys:
            series = df[col].dropna()
            col_min, col_max = series.min(), series.max()
            col_mean, col_median = series.mean(), series.median()
            print(
                f"      • {col}: range=[{col_min:.0f}, {col_max:.0f}], "
                f"mean={col_mean:.1f}, median={col_median:.1f}"
            )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    indicator_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) <= 2]
    continuous_cols = [col for col in numeric_cols if col not in indicator_cols]

    if indicator_cols:
        print("\n   Indicator / binary columns (no scaling needed):")
        print("      • " + ", ".join(sorted(indicator_cols)))

    recommendations: dict[str, list[str]] = {"robust": [], "standard": []}
    outlier_pct: dict[str, float] = {}
    for col in continuous_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            recommendations["standard"].append(col)
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct = (series.lt(lower) | series.gt(upper)).sum() / len(series) * 100
        outlier_pct[col] = pct
        if pct >= 5:
            recommendations["robust"].append(col)
        else:
            recommendations["standard"].append(col)

    if recommendations["robust"]:
        print("\n   Use RobustScaler / capping for skewed columns:")
        for col in sorted(recommendations["robust"]):
            pct = outlier_pct.get(col, 0.0)
            print(f"      • {col} (IQR outliers ≈ {pct:.1f}% of rows)")
    else:
        print("\n   No columns exceed the 5% IQR-outlier threshold for RobustScaler.")

    if recommendations["standard"]:
        print("\n   Use StandardScaler for remaining continuous columns:")
        print("      • " + ", ".join(sorted(recommendations["standard"])))

    print("\n   Criteria: indicator columns skipped; ≥5% IQR outliers → RobustScaler; others → StandardScaler.")
    print("-" * 60)


def main() -> None:
    try:
        df_processed = process_data()
        log_scaling_strategy(df_processed)
        save_processed_data(df_processed, DEFAULT_OUTPUT_FILE)
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("  1) Run 'python -m src.models.tune'  # optional but recommended for hyperparameter search")
        print("  2) Run 'python -m src.pipeline.run' to train and evaluate models")
        print("="*60)
    except Exception as exc:  # pragma: no cover - top-level logging only
        print(f"An error occurred during processing: {exc}")


if __name__ == "__main__":
    main()