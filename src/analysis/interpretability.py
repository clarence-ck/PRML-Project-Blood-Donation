"""Interpretability analysis for the best donor-return model.

Computes model-agnostic permutation importance for the BEST MODEL ONLY
and saves a feature-importance plot plus a small JSON summary.

Intended usage:

    python -m src.analysis.interpretability

or via the post-training orchestrator:

    python -m src.analysis.run_all

which will call ``run_interpretability_analysis`` if this module exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.data.splits import load_and_split

# ---------------------------------------------------------------------------
# Paths and directories
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
INTERP_DIR = REPORTS_DIR / "interpretability"
SUMMARY_PATH = INTERP_DIR / "interpretability_summary.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INTERP_DIR.mkdir(parents=True, exist_ok=True)


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


def compute_permutation_importance_for_best_model(
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, object]:
    """Compute permutation importance for the best model on the validation set.

    Parameters
    ----------
    n_repeats : int, default=10
        Number of shuffling rounds per feature.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    summary : dict
        Dictionary with feature names and importance statistics.
    """

    pipeline = _load_best_pipeline()
    splits = load_and_split()

    X_val = splits.X_val
    y_val = splits.y_val

    # Ensure we are working with a DataFrame for clearer column names
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val)

    print("=" * 60)
    print("INTERPRETABILITY ANALYSIS (BEST MODEL - PERMUTATION IMPORTANCE)")
    print("=" * 60)
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {X_val.shape[1]}")

    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="f1",
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    feature_names = list(getattr(X_val, "columns", range(len(importances_mean))))

    df_importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Basic summary for JSON output
    summary = {
        "n_features": int(len(df_importances)),
        "top_features": df_importances.head(20).to_dict(orient="records"),
        "n_repeats": int(n_repeats),
        "scoring": "f1",
    }

    return {"importances": df_importances, "summary": summary}


def plot_feature_importances(
    df_importances: pd.DataFrame,
    top_n: int = 20,
) -> str:
    """Plot top-N features by permutation importance.

    Parameters
    ----------
    df_importances : pd.DataFrame
        DataFrame with columns ['feature', 'importance_mean', 'importance_std'].
    top_n : int, default=20
        Number of top features to display.

    Returns
    -------
    path : str
        Path to the saved PNG file.
    """

    sns.set_style("whitegrid")

    top = df_importances.head(top_n).iloc[::-1]  # reverse for nicer horizontal plot

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))

    ax.barh(
        top["feature"],
        top["importance_mean"],
        color="steelblue",
        alpha=0.8,
    )

    ax.set_xlabel("Permutation importance (mean decrease in F1)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Top feature importances - best model (validation set)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_importances_best_model.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def run_interpretability_analysis() -> Dict[str, object]:
    """Run permutation-importance-based interpretability for the best model.

    Returns
    -------
    results : dict
        Summary dictionary including top features and plot paths.
    """

    results = compute_permutation_importance_for_best_model()
    df_importances = results["importances"]
    summary = results["summary"]

    print("\nTop 10 features by permutation importance (validation set):")
    for row in summary["top_features"][:10]:
        print(
            f"  - {row['feature']}: mean={row['importance_mean']:.4f} "+
            f"± {row['importance_std']:.4f}"
        )

    fig_path = plot_feature_importances(df_importances, top_n=20)

    summary_with_paths = {
        **summary,
        "figure_paths": {
            "feature_importances": fig_path,
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary_with_paths, fp, indent=2)

    print("\nInterpretability summary written to:")
    print(f"  {SUMMARY_PATH}")
    print("Feature importance plot saved to:")
    print(f"  {fig_path}")

    return summary_with_paths


if __name__ == "__main__":
    run_interpretability_analysis()
