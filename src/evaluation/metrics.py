"""Evaluation helpers for donor intention models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_array(values: Iterable) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D array for evaluations")
    return arr


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray
    decile_stats: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion.tolist(),
            **self.decile_stats,
        }


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: Iterable[float],
) -> ClassificationMetrics:
    y_true_arr = _safe_array(y_true)
    y_pred_arr = _safe_array(y_pred)
    y_proba_arr = _safe_array(y_proba)

    metrics_tuple = (
        accuracy_score(y_true_arr, y_pred_arr),
        precision_score(y_true_arr, y_pred_arr, zero_division=0),
        recall_score(y_true_arr, y_pred_arr, zero_division=0),
        f1_score(y_true_arr, y_pred_arr, zero_division=0),
        roc_auc_score(y_true_arr, y_proba_arr),
        average_precision_score(y_true_arr, y_proba_arr),
        confusion_matrix(y_true_arr, y_pred_arr),
    )

    deciles = top_bottom_decile_stats(y_true_arr, y_proba_arr)

    return ClassificationMetrics(
        accuracy=metrics_tuple[0],
        precision=metrics_tuple[1],
        recall=metrics_tuple[2],
        f1=metrics_tuple[3],
        roc_auc=metrics_tuple[4],
        pr_auc=metrics_tuple[5],
        confusion=metrics_tuple[6],
        decile_stats=deciles,
    )


def top_bottom_decile_stats(y_true: Iterable[int], y_proba: Iterable[float]) -> Dict[str, float]:
    y_true_arr = _safe_array(y_true)
    y_proba_arr = _safe_array(y_proba)
    order = np.argsort(y_proba_arr)
    decile_size = max(1, len(order) // 10)
    bottom_idx = order[:decile_size]
    top_idx = order[-decile_size:]
    bottom_rate = float(y_true_arr[bottom_idx].mean())
    top_rate = float(y_true_arr[top_idx].mean())
    lift = top_rate / bottom_rate if bottom_rate > 0 else float("inf")
    return {
        "top_decile_rate": top_rate,
        "bottom_decile_rate": bottom_rate,
        "lift": lift,
    }
