"""Dataset loading and stratified split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.preprocess import FEATURE_COLUMNS

DEFAULT_PROCESSED_FILE = Path("processed_donor_features.csv")
TARGET_COLUMN = "target_available"


@dataclass
class DatasetSplits:
    """Structured container for train/validation/test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def sizes(self) -> Tuple[int, int, int]:
        return (len(self.X_train), len(self.X_val), len(self.X_test))


def load_processed_dataset(path: Path | str = DEFAULT_PROCESSED_FILE) -> pd.DataFrame:
    """Load the processed donor feature matrix and validate schema."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {resolved_path.resolve()}")

    df = pd.read_csv(resolved_path)
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Processed dataset is missing expected columns: {missing_cols}")
    return df


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Create stratified train/val/test partitions from the processed dataset."""

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum < 1")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=random_state,
    )

    test_fraction = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def load_and_split(
    path: Path | str = DEFAULT_PROCESSED_FILE,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Convenience wrapper: load the processed CSV and perform the split."""

    df = load_processed_dataset(path)
    return stratified_split(df, val_size=val_size, test_size=test_size, random_state=random_state)
