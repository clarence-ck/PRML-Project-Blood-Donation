"""Enhanced model evaluation module with comprehensive visualizations.

Generates:
- Confusion matrices (visual heatmaps)
- ROC curves with AUC
- Precision-Recall curves with AP
- Calibration curves
- Decile lift charts
- Threshold analysis
- Model comparison tables
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    brier_score_loss,
)

from src.data.splits import load_and_split
from src.evaluation.metrics import compute_classification_metrics

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
EVAL_DIR = Path("reports/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = EVAL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = EVAL_DIR / "evaluation_summary.json"


def load_best_pipeline():
    """Load the best trained pipeline."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best pipeline not found at {BEST_MODEL_PATH}. "
            "Run 'python -m src.pipeline.run' first."
        )
    return joblib.load(BEST_MODEL_PATH)


def plot_confusion_matrix(y_true, y_pred, split_label: str) -> str:
    """Generate visual confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        # Use default integer count annotations for clarity
        values_format="d",
        display_labels=["No", "Yes"],
    )
    
    ax.set_title(f"Confusion Matrix ({split_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.grid(False)

    plt.tight_layout()
    filename = FIGURES_DIR / f"confusion_matrix_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def plot_roc_curve(y_true, y_proba, split_label: str) -> str:
    """Generate ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    disp = RocCurveDisplay.from_predictions(
        y_true,
        y_proba,
        ax=ax,
        name=split_label,
        color="steelblue",
        linewidth=2,
    )
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    
    ax.set_title(f"ROC Curve ({split_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = FIGURES_DIR / f"roc_curve_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def plot_precision_recall_curve(y_true, y_proba, split_label: str) -> str:
    """Generate Precision-Recall curve with AP annotation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    disp = PrecisionRecallDisplay.from_predictions(
        y_true,
        y_proba,
        ax=ax,
        name=split_label,
        color="coral",
        linewidth=2,
    )
    
    # Add baseline (proportion of positive class)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.2f})")
    
    ax.set_title(f"Precision-Recall Curve ({split_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = FIGURES_DIR / f"pr_curve_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def plot_calibration_curve(y_true, y_proba, split_label: str) -> str:
    """Generate calibration curve (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="uniform"
    )
    
    # Plot calibration curve
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=2,
        label=split_label,
        color="steelblue",
        markersize=8,
    )
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly Calibrated")
    
    ax.set_title(f"Calibration Curve ({split_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = FIGURES_DIR / f"calibration_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def compute_decile_stats(y_true, y_proba) -> pd.DataFrame:
    """Compute performance metrics by probability decile."""
    df = pd.DataFrame({"true": y_true, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"], 10, labels=False, duplicates="drop")
    
    summary = (
        df.groupby("decile")["true"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n", "sum": "positives", "mean": "hit_rate"})
        .sort_index(ascending=False)
    )
    
    overall_rate = df["true"].mean()
    summary["lift"] = summary["hit_rate"] / overall_rate
    summary["decile_rank"] = summary.index.max() - summary.index
    
    return summary.reset_index(drop=True)


def plot_decile_lift(decile_df: pd.DataFrame, split_label: str) -> str:
    """Generate decile lift chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(decile_df)))
    bars = ax.bar(
        decile_df["decile_rank"],
        decile_df["lift"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    
    # Add value labels
    for idx, row in decile_df.iterrows():
        ax.text(
            row["decile_rank"],
            row["lift"] + 0.05,
            f"{row['lift']:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    
    # Add baseline
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Baseline (Lift = 1.0)")
    
    ax.set_xlabel("Decile (0 = Top 10%)", fontsize=12)
    ax.set_ylabel("Lift vs Overall Rate", fontsize=12)
    ax.set_title(f"Lift by Probability Decile ({split_label})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    filename = FIGURES_DIR / f"decile_lift_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def plot_threshold_analysis(y_true, y_proba, split_label: str) -> str:
    """Plot precision, recall, and F1 vs classification threshold."""
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precision[:-1], label="Precision", linewidth=2, color="steelblue")
    ax.plot(thresholds, recall[:-1], label="Recall", linewidth=2, color="coral")
    ax.plot(thresholds, f1_scores[:-1], label="F1-Score", linewidth=2, color="green")
    
    # Mark default threshold
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Default Threshold (0.5)")
    
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Metrics vs Threshold ({split_label})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = FIGURES_DIR / f"threshold_analysis_{split_label}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return str(filename)


def evaluate_split(pipeline, X, y, split_label: str) -> Dict[str, object]:
    """Comprehensive evaluation for a single data split."""
    print(f"\n  Evaluating {split_label} split...")
    
    # Predictions
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Core metrics
    metrics = compute_classification_metrics(y, y_pred, y_proba)
    brier = brier_score_loss(y, y_proba)
    
    # Generate visualizations
    print(f"    - Confusion matrix...")
    cm_path = plot_confusion_matrix(y, y_pred, split_label)
    
    print(f"    - ROC curve...")
    roc_path = plot_roc_curve(y, y_proba, split_label)
    
    print(f"    - PR curve...")
    pr_path = plot_precision_recall_curve(y, y_proba, split_label)
    
    print(f"    - Calibration curve...")
    cal_path = plot_calibration_curve(y, y_proba, split_label)
    
    print(f"    - Decile lift...")
    decile_df = compute_decile_stats(y, y_proba)
    decile_path = plot_decile_lift(decile_df, split_label)
    
    print(f"    - Threshold analysis...")
    threshold_path = plot_threshold_analysis(y, y_proba, split_label)
    
    return {
        "metrics": metrics.as_dict(),
        "brier_score": float(brier),
        "plots": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "pr_curve": pr_path,
            "calibration": cal_path,
            "decile_lift": decile_path,
            "threshold_analysis": threshold_path,
        },
        "decile_summary": decile_df.to_dict(orient="records"),
    }


def run_evaluation_analysis() -> Dict[str, object]:
    """Run complete evaluation analysis."""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and data
    print("\nLoading best pipeline and data splits...")
    pipeline = load_best_pipeline()
    splits = load_and_split()
    
    # Evaluate validation and test splits
    results = {
        "validation": evaluate_split(pipeline, splits.X_val, splits.y_val, "validation"),
        "test": evaluate_split(pipeline, splits.X_test, splits.y_test, "test"),
    }
    
    # Save summary
    print("\nSaving evaluation summary...")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    # Print key metrics
    print("\nKEY METRICS (Test Set):")
    test_metrics = results["test"]["metrics"]
    print(f"  - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall:    {test_metrics['recall']:.4f}")
    print(f"  - F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  - ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  - Brier:     {results['test']['brier_score']:.4f}")
    print(f"  - Lift (top decile): {test_metrics['lift']:.2f}")
    
    return results


if __name__ == "__main__":
    run_evaluation_analysis()
