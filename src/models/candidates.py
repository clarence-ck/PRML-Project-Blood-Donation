"""Factory for scikit-learn compatible model pipelines."""

from __future__ import annotations

from typing import List, Tuple

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.features.preprocess import build_preprocessor
from src.features.imbalance import create_smote_pipeline


def _make_pipeline(estimator, *, scale_numeric: bool) -> Pipeline:
    steps = []
    if scale_numeric:
        steps.append(("preprocess", build_preprocessor(scale_numeric=True)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _candidate_builders() -> dict[str, callable]:
    builders: dict[str, callable] = {
        "logistic_regression": lambda: _make_pipeline(
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs",
            ),
            scale_numeric=True,
        ),
        "random_forest": lambda: _make_pipeline(
            RandomForestClassifier(
                n_estimators=600,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            scale_numeric=False,
        ),
        "adaboost": lambda: _make_pipeline(
            AdaBoostClassifier(
                n_estimators=400,
                learning_rate=0.8,
                random_state=42,
            ),
            scale_numeric=False,
        ),
        "svm_linear": lambda: _make_pipeline(
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=5000,
                    dual=False,
                    random_state=42,
                ),
                method="sigmoid",
                cv=3,
                n_jobs=-1,
            ),
            scale_numeric=True,
        ),
    }

    builders["xgboost"] = lambda: _make_pipeline(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_estimators=400,
            max_depth=6,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            n_jobs=-1,
            random_state=42,
        ),
        scale_numeric=False,
    )

    builders["lightgbm"] = lambda: _make_pipeline(
        LGBMClassifier(
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=600,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        scale_numeric=False,
    )

    # SMOTE variants (resampling instead of class weights)
    builders["logistic_regression_smote"] = lambda: create_smote_pipeline(
        LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=42,
        ),
        scale_numeric=True,
        smote_variant="smote",
    )

    builders["random_forest_smote"] = lambda: create_smote_pipeline(
        RandomForestClassifier(
            n_estimators=600,
            n_jobs=-1,
            random_state=42,
        ),
        scale_numeric=False,
        smote_variant="smote",
    )

    builders["xgboost_smote"] = lambda: create_smote_pipeline(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_estimators=400,
            max_depth=6,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
        ),
        scale_numeric=False,
        smote_variant="smote",
    )

    return builders


def candidate_builders() -> dict[str, callable]:
    """Expose candidate builders for reuse (e.g., tuning, benchmarking)."""

    return _candidate_builders()


def candidate_pipelines() -> List[Tuple[str, Pipeline]]:
    return [(name, builder()) for name, builder in candidate_builders().items()]
