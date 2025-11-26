"""Subgroup fairness analysis for the best donor-return model.

Evaluates the BEST MODEL ONLY across key demographic subgroups and
saves metrics plus simple comparison plots.

Intended usage:

    python -m src.analysis.subgroups

or via the post-training orchestrator:

    python -m src.analysis.run_all

which will call ``run_subgroup_analysis`` if this module exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.splits import load_and_split
from src.evaluation.metrics import compute_classification_metrics

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
SUBGROUP_SUMMARY_PATH = REPORTS_DIR / "subgroup_analysis.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Minimum number of samples required for a subgroup to be reported
MIN_SAMPLES_PER_GROUP = 100

BLOOD_GROUP_DUMMIES = [
    "blood_group_A-",
    "blood_group_AB+",
    "blood_group_AB-",
    "blood_group_B+",
    "blood_group_B-",
    "blood_group_O+",
    "blood_group_O-",
]
BASE_BLOOD_GROUP = "A+"  # reference group when all dummies are zero


def _load_best_pipeline():
    """Load the best trained pipeline from disk.

    Raises
    ------
    FileNotFoundError
        If ``models/best_model.pkl`` does not exist. In that case,
        run ``python -m src.pipeline.run`` first.
    """

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best pipeline not found at {BEST_MODEL_PATH}. "
            "Run 'python -m src.pipeline.run' first."
        )
    return joblib.load(BEST_MODEL_PATH)


def _derive_gender(X: pd.DataFrame) -> pd.Series:
    """Derive categorical gender labels from encoded features.

    Expects a ``gender_Male`` column created by process_data.py; values
    of 1 map to "Male" and 0 map to "Female".
    """

    if "gender_Male" not in X.columns:
        return pd.Series("Unknown", index=X.index, dtype="object")

    return X["gender_Male"].map({1: "Male", 0: "Female"}).astype("category")


def _derive_blood_group(X: pd.DataFrame) -> pd.Series:
    """Reconstruct blood group labels from one-hot encoded columns.

    Uses BLOOD_GROUP_DUMMIES with BASE_BLOOD_GROUP as the reference
    category when all dummies are zero.
    """

    groups = pd.Series(BASE_BLOOD_GROUP, index=X.index, dtype="object")

    for col in BLOOD_GROUP_DUMMIES:
        if col in X.columns:
            mask = X[col] == 1
            label = col.replace("blood_group_", "")
            groups.loc[mask] = label

    return groups.astype("category")


def _derive_age_band(X: pd.DataFrame) -> pd.Series:
    """Bin numeric age into coarse bands.

    Bands: 18-30, 31-40, 41-50, 51+ (inclusive on lower bound).
    """

    if "age" not in X.columns:
        return pd.Series("Unknown", index=X.index, dtype="object")

    age = X["age"].astype(float)

    def _to_band(a: float) -> str:
        if a < 31:
            return "18-30"
        if a < 41:
            return "31-40"
        if a < 51:
            return "41-50"
        return "51+"

    return age.map(_to_band).astype("category")


def _compute_overall_metrics(pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute metrics for the full dataset using the best model."""

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y, y_pred, y_proba)
    return metrics.as_dict()


def _compute_subgroup_metrics(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per subgroup value.

    Subgroups with fewer than MIN_SAMPLES_PER_GROUP samples are skipped.
    """

    results: Dict[str, Dict[str, float]] = {}
    groups = groups.astype("object")

    for value, idx in groups.groupby(groups).groups.items():
        if len(idx) < MIN_SAMPLES_PER_GROUP:
            continue

        X_g = X.loc[idx]
        y_g = y.loc[idx]

        y_proba = pipeline.predict_proba(X_g)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_classification_metrics(y_g, y_pred, y_proba)
        results[str(value)] = metrics.as_dict()

    return results


def _plot_subgroup_metric(
    subgroup_metrics: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    filename: str,
) -> str:
    """Bar plot of a single metric across subgroups.

    Returns the path to the saved PNG file.
    """

    if not subgroup_metrics:
        return ""

    sns.set_style("whitegrid")

    labels = list(subgroup_metrics.keys())
    values = [subgroup_metrics[label].get(metric, np.nan) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="black")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Subgroup")
    # Leave headroom above the tallest bar so value labels do not collide with
    # the plot title or the top of the axis.
    valid_values = [v for v in values if not np.isnan(v)]
    upper = max(valid_values) if valid_values else 1.0
    ax.set_ylim(0, min(1.05, upper + 0.05))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


def run_subgroup_analysis() -> Dict[str, object]:
    """Run subgroup fairness analysis for the best model.

    Evaluates performance on the TEST set across gender, age bands,
    and blood groups.
    """

    print("=" * 60)
    print("SUBGROUP FAIRNESS ANALYSIS (BEST MODEL - TEST SET)")
    print("=" * 60)

    pipeline = _load_best_pipeline()
    splits = load_and_split()

    X_test = splits.X_test
    y_test = splits.y_test

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    overall_metrics = _compute_overall_metrics(pipeline, X_test, y_test)

    gender = _derive_gender(X_test)
    age_band = _derive_age_band(X_test)
    blood_group = _derive_blood_group(X_test)

    subgroup_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    gender_metrics = _compute_subgroup_metrics(pipeline, X_test, y_test, gender)
    age_band_metrics = _compute_subgroup_metrics(pipeline, X_test, y_test, age_band)
    blood_group_metrics = _compute_subgroup_metrics(pipeline, X_test, y_test, blood_group)

    subgroup_results["gender"] = gender_metrics
    subgroup_results["age_band"] = age_band_metrics
    subgroup_results["blood_group"] = blood_group_metrics

    print("\nOverall test-set recall:", f"{overall_metrics['recall']:.4f}")
    print("Subgroup recall (test set):")
    for name, metrics_dict in (
        ("gender", gender_metrics),
        ("age_band", age_band_metrics),
        ("blood_group", blood_group_metrics),
    ):
        print(f"  {name}:")
        for subgroup, m in metrics_dict.items():
            print(f"    - {subgroup}: recall={m['recall']:.4f}, f1={m['f1']:.4f}")

    figure_paths = {
        "gender_recall": _plot_subgroup_metric(
            gender_metrics,
            metric="recall",
            title="Recall by gender (test set)",
            filename="subgroup_recall_gender.png",
        ),
        "age_band_recall": _plot_subgroup_metric(
            age_band_metrics,
            metric="recall",
            title="Recall by age band (test set)",
            filename="subgroup_recall_age_band.png",
        ),
        "blood_group_recall": _plot_subgroup_metric(
            blood_group_metrics,
            metric="recall",
            title="Recall by blood group (test set)",
            filename="subgroup_recall_blood_group.png",
        ),
    }

    summary = {
        "overall_metrics": overall_metrics,
        "subgroup_results": subgroup_results,
        "figure_paths": figure_paths,
        "min_samples_per_group": MIN_SAMPLES_PER_GROUP,
        "split": "test",
    }

    with open(SUBGROUP_SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\nSubgroup analysis summary written to:")
    print(f"  {SUBGROUP_SUMMARY_PATH}")
    print("Subgroup recall plots saved to:")
    for key, path in figure_paths.items():
        if path:
            print(f"  {key}: {path}")

    return summary


if __name__ == "__main__":
    run_subgroup_analysis()
