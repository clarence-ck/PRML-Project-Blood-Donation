"""Schema validation tests for the processed dataset."""

from __future__ import annotations

from src.data.splits import TARGET_COLUMN, load_processed_dataset
from src.features.preprocess import FEATURE_COLUMNS, NUMERIC_FEATURES

BOOLEAN_FEATURES = {
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
}


def test_processed_dataset_schema() -> None:
    df = load_processed_dataset()

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing expected column: {col}"

    numeric_cols = df[NUMERIC_FEATURES].select_dtypes(include=["number"]).columns.tolist()
    assert set(NUMERIC_FEATURES) == set(numeric_cols), "Numeric feature dtypes mismatch"

    bool_values = {0, 1, True, False}
    for col in BOOLEAN_FEATURES:
        assert col in df.columns, f"Missing boolean feature column: {col}"
        uniques = set(df[col].dropna().unique())
        assert uniques.issubset(bool_values), f"Column {col} must be binary"

    target_unique = df[TARGET_COLUMN].dropna().unique()
    assert set(target_unique).issubset({0, 1, True, False})
