"""Hyperparameter tuning for donor availability models.

Runs RandomizedSearchCV followed by GridSearchCV for each candidate pipeline,
then saves the tuned estimator artifacts for downstream selection/evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingGridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.data.splits import load_and_split
from src.models.candidates import candidate_builders

RESULTS_DIR = Path("models") / "tuned"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
SCORING = "f1"
SUBSAMPLE_FRACTION = 0.5


def _logspace_values(center: float, span: float = 1.0) -> list[float]:
    values = [center / span, center, center * span]
    return sorted({max(v, 1e-6) for v in values})


PARAM_CONFIGS: Dict[str, Dict[str, object]] = {
    "logistic_regression": {
        "random": {
            "model__C": np.logspace(-2, 2, 20),
        },
        "random_iter": 12,
        "grid_generator": lambda best: {
            "model__C": _logspace_values(best.get("model__C", 1.0), span=2.0),
        },
    },
    "random_forest": {
        "random": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__max_depth": [best.get("model__max_depth", None), 20],
            "model__min_samples_leaf": [1, best.get("model__min_samples_leaf", 2)],
            "model__max_features": [best.get("model__max_features", "sqrt")],
        },
    },
    "adaboost": {
        "random": {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.1, 0.2, 0.5, 0.8],
        },
        "random_iter": 12,
        "grid_generator": lambda best: {
            "model__n_estimators": [best.get("model__n_estimators", 400), 500],
            "model__learning_rate": _logspace_values(best.get("model__learning_rate", 0.5), span=2.0),
        },
    },
    "svm_linear": {
        "random": {
            "model__estimator__C": np.logspace(-3, 1, 20),
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__estimator__C": _logspace_values(
                best.get("model__estimator__C", 1.0), span=2.0
            ),
        },
    },
    "xgboost": {
        "random": {
            "model__n_estimators": [300, 450],
            "model__max_depth": [4, 6],
            "model__subsample": [0.7, 0.85],
            "model__colsample_bytree": [0.7, 0.85],
            "model__learning_rate": [0.05, 0.1, 0.15],
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__max_depth": [best.get("model__max_depth", 6)],
            "model__learning_rate": _logspace_values(best.get("model__learning_rate", 0.1), span=1.5),
            "model__subsample": [best.get("model__subsample", 0.8)],
            "model__colsample_bytree": [best.get("model__colsample_bytree", 0.8)],
        },
    },
    "lightgbm": {
        "random": {
            "model__num_leaves": [31, 63, 95],
            "model__max_depth": [-1, 15],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.75, 0.9],
            "model__colsample_bytree": [0.75, 0.9],
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__num_leaves": [best.get("model__num_leaves", 63), 80],
            "model__learning_rate": _logspace_values(best.get("model__learning_rate", 0.05), span=1.5),
        },
    },
    "logistic_regression_smote": {
        # Same search space as logistic_regression; underlying step name is still "model"
        "random": {
            "model__C": np.logspace(-2, 2, 20),
        },
        "random_iter": 12,
        "grid_generator": lambda best: {
            "model__C": _logspace_values(best.get("model__C", 1.0), span=2.0),
        },
    },
    "random_forest_smote": {
        # Mirror random_forest search space for the SMOTE-enabled variant
        "random": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__max_depth": [best.get("model__max_depth", None), 20],
            "model__min_samples_leaf": [1, best.get("model__min_samples_leaf", 2)],
            "model__max_features": [best.get("model__max_features", "sqrt")],
        },
    },
    "xgboost_smote": {
        # Mirror xgboost search space for the SMOTE-enabled variant
        "random": {
            "model__n_estimators": [300, 450],
            "model__max_depth": [4, 6],
            "model__subsample": [0.7, 0.85],
            "model__colsample_bytree": [0.7, 0.85],
            "model__learning_rate": [0.05, 0.1, 0.15],
        },
        "random_iter": 16,
        "grid_generator": lambda best: {
            "model__max_depth": [best.get("model__max_depth", 6)],
            "model__learning_rate": _logspace_values(best.get("model__learning_rate", 0.1), span=1.5),
            "model__subsample": [best.get("model__subsample", 0.8)],
            "model__colsample_bytree": [best.get("model__colsample_bytree", 0.8)],
        },
    },
}


def run_random_search(name: str, estimator, config: dict, X, y):
    rand_params = config.get("random")
    if not rand_params:
        return None
    n_iter = config.get("random_iter", 20)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=rand_params,
        n_iter=n_iter,
        scoring=SCORING,
        n_jobs=-1,
        cv=CV,
        random_state=42,
        verbose=1,
    )
    search.fit(X, y)
    return search


def run_grid_search(name: str, estimator, grid_params: dict, X, y):
    if not grid_params:
        return None
    search = HalvingGridSearchCV(
        estimator=estimator,
        param_grid=grid_params,
        scoring=SCORING,
        n_jobs=-1,
        cv=CV,
        verbose=1,
        factor=3,
        aggressive_elimination=False,
    )
    search.fit(X, y)
    return search


def main() -> None:
    splits = load_and_split()
    X_train_full, y_train_full = splits.X_train, splits.y_train

    if SUBSAMPLE_FRACTION < 1.0:
        X_search, _, y_search, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=SUBSAMPLE_FRACTION,
            stratify=y_train_full,
            random_state=42,
        )
    else:
        X_search, y_search = X_train_full, y_train_full

    builders = candidate_builders()

    for name, builder in builders.items():
        if name not in PARAM_CONFIGS:
            print(f"Skipping tuning for {name}: no parameter config defined.")
            continue

        print(f"\n=== Tuning {name} ===")
        config = PARAM_CONFIGS[name]
        base_estimator = builder()

        best_estimator = base_estimator
        best_params: Dict[str, object] = {}

        random_search = run_random_search(name, base_estimator, config, X_search, y_search)
        if random_search:
            best_estimator = random_search.best_estimator_
            best_params = random_search.best_params_
            print(f"  RandomizedSearchCV best F1: {random_search.best_score_:.4f}")

        if "grid_generator" in config:
            grid_params = config["grid_generator"](best_params)
        elif "grid" in config:
            grid_params = config["grid"]
        else:
            raise ValueError(f"Model '{name}' must define grid_generator or grid for compulsory grid search.")

        if not grid_params:
            raise ValueError(f"Model '{name}' produced empty grid parameters; cannot skip grid search.")

        grid_search = run_grid_search(name, builder(), grid_params, X_search, y_search)
        if grid_search:
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"  HalvingGridSearchCV best F1: {grid_search.best_score_:.4f}")

        final_estimator = builder()
        if best_params:
            final_estimator.set_params(**best_params)
        final_estimator.fit(X_train_full, y_train_full)

        artifact_path = RESULTS_DIR / f"{name}.pkl"
        joblib.dump(final_estimator, artifact_path)
        print(f"  Saved tuned estimator to {artifact_path}")

        summary = {
            "best_params": best_params,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "cv_folds": CV.get_n_splits(),
        }
        if random_search:
            summary["random_search_best_score"] = random_search.best_score_
        if grid_params:
            summary["grid_search_params"] = grid_params
        summary_path = RESULTS_DIR / f"{name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"  Summary written to {summary_path}")


if __name__ == "__main__":
    main()
