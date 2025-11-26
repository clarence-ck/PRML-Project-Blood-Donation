"""Orchestrate the donor intention modeling pipeline using modular components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.splits import load_and_split
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.candidates import candidate_builders

MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RESULTS_PATH = MODELS_DIR / "all_model_results.json"
TUNED_DIR = MODELS_DIR / "tuned"


def evaluate_pipeline(pipeline, X, y) -> ClassificationMetrics:
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return compute_classification_metrics(y, y_pred, y_proba)


def evaluate_with_cross_validation(pipeline, X, y, cv_folds: int = 4) -> Dict[str, Dict[str, float]]:
    """Evaluate pipeline using stratified cross-validation.
    
    Returns dict with mean and std for each metric.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }
    
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    # Aggregate results
    aggregated = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        aggregated[metric] = {
            'mean': float(test_scores.mean()),
            'std': float(test_scores.std()),
            'min': float(test_scores.min()),
            'max': float(test_scores.max()),
        }
    
    return aggregated


def resolve_pipeline(name: str, builder) -> object:
    tuned_path = TUNED_DIR / f"{name}.pkl"
    if tuned_path.exists():
        print(f"Loading tuned pipeline for {name} from {tuned_path}")
        return joblib.load(tuned_path)
    return builder()


def run() -> None:
    splits = load_and_split()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_name = None
    best_pipeline = None
    best_val_recall = -float("inf")
    best_val_lift = -float("inf")
    results_summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    model_selection_data: list[dict] = []

    builders = candidate_builders()

    for name, builder in builders.items():
        print(f"\nTraining {name}...")
        pipeline = resolve_pipeline(name, builder)

        # Cross-validation on training data
        print("  Running 4-fold cross-validation...")
        cv_metrics = evaluate_with_cross_validation(pipeline, splits.X_train, splits.y_train, cv_folds=4)
        
        # Train on full training set
        pipeline.fit(splits.X_train, splits.y_train)

        val_metrics = evaluate_pipeline(pipeline, splits.X_val, splits.y_val)
        test_metrics = evaluate_pipeline(pipeline, splits.X_test, splits.y_test)

        results_summary[name] = {
            "cross_validation": cv_metrics,
            "validation": val_metrics.as_dict(),
            "test": test_metrics.as_dict(),
        }

        # Log CV metrics
        print(f"  [cross-validation] mean ± std:")
        for metric, values in cv_metrics.items():
            print(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

        def _log_metrics(split: str, metrics_obj: ClassificationMetrics) -> None:
            metrics_dict = metrics_obj.as_dict()
            print(f"  [{split}] metrics:")
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                print(f"    {key}: {metrics_dict[key]:.4f}")
            print("    confusion_matrix (rows=actual, cols=predicted):")
            headers = "             Pred No   Pred Yes"
            actual_labels = ["Actual No ", "Actual Yes"]
            print(f"{headers}")
            for label, row in zip(actual_labels, metrics_obj.confusion):
                print(f"      {label}{row[0]:9d}{row[1]:10d}")
            print(
                f"    top_decile_rate: {metrics_obj.decile_stats['top_decile_rate']:.4f} | "
                f"bottom_decile_rate: {metrics_obj.decile_stats['bottom_decile_rate']:.4f} | "
                f"lift: {metrics_obj.decile_stats['lift']:.3f}"
            )

        _log_metrics("validation", val_metrics)
        _log_metrics("test", test_metrics)

        model_selection_data.append(
            {
                "name": name,
                "pipeline": pipeline,
                "val_recall": float(val_metrics.recall),
                "val_lift": float(val_metrics.decile_stats["lift"]),
            }
        )

    if not model_selection_data:
        raise RuntimeError("No models were trained successfully.")

    recalls = [m["val_recall"] for m in model_selection_data]
    lifts = [m["val_lift"] for m in model_selection_data]
    rec_min, rec_max = min(recalls), max(recalls)
    lift_min, lift_max = min(lifts), max(lifts)

    def _normalize(value: float, vmin: float, vmax: float) -> float:
        if vmax == vmin:
            return 0.0
        return (value - vmin) / (vmax - vmin)

    best_score = 0.0
    best_name = None
    best_pipeline = None
    best_val_recall = 0.0
    best_val_lift = 0.0

    for info in model_selection_data:
        recall_norm = _normalize(info["val_recall"], rec_min, rec_max)
        lift_norm = _normalize(info["val_lift"], lift_min, lift_max)
        score = 0.5 * recall_norm + 0.5 * lift_norm

        print(
            f"  [selection] model='{info['name']}', "
            f"score={score:.4f}, recall_norm={recall_norm:.4f}, "
            f"lift_norm={lift_norm:.4f}, val_recall={info['val_recall']:.4f}, "
            f"val_lift={info['val_lift']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_name = info["name"]
            best_pipeline = info["pipeline"]
            best_val_recall = info["val_recall"]
            best_val_lift = info["val_lift"]
        elif score == best_score:
            if info["val_recall"] > best_val_recall or (
                info["val_recall"] == best_val_recall
                and info["val_lift"] > best_val_lift
            ):
                best_score = score
                best_name = info["name"]
                best_pipeline = info["pipeline"]
                best_val_recall = info["val_recall"]
                best_val_lift = info["val_lift"]

    if best_pipeline is None:
        raise RuntimeError("No models were trained successfully.")

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump({"best_model": best_name, "results": results_summary}, fp, indent=2)

    print(
        f"\nBest model: {best_name} "
        f"(recall={best_val_recall:.4f}, lift={best_val_lift:.3f})"
    )
    print(
        "  Selection rule: choose the model with the highest composite score "
        "(0.5 × normalized recall + 0.5 × normalized lift) on the validation set."
    )
    print(
        "  Tie-breaking: if multiple models share the same composite score, "
        "prefer higher validation recall; if recall is also equal, prefer "
        "higher validation lift."
    )
    print(
        "  Explanation: this balances capturing many willing donors (recall) "
        "with ranking efficiency (lift) when choosing the production model."
    )
    print(f"Saved pipeline to {BEST_MODEL_PATH}")
    print(f"Evaluation summary written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
