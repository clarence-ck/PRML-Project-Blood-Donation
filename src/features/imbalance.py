"""Class imbalance handling utilities and SMOTE pipeline variants.

Provides resampling techniques to address class imbalance:
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- BorderlineSMOTE
- Random oversampling
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features.preprocess import build_preprocessor

def create_smote_pipeline(estimator, *, scale_numeric: bool, smote_variant: str = "smote") -> ImbPipeline:
    """Create a pipeline with SMOTE resampling.
    
    Args:
        estimator: The classifier to use
        scale_numeric: Whether to scale numeric features
        smote_variant: Type of SMOTE ('smote')
    
    Returns:
        Pipeline with preprocessing, SMOTE, and classifier
    """
    steps = []
    
    # Add preprocessing if needed
    if scale_numeric:
        steps.append(("preprocess", build_preprocessor(scale_numeric=True)))

    # Only plain SMOTE is supported in the current configuration
    if smote_variant != "smote":
        raise ValueError(f"Unsupported smote_variant '{smote_variant}'. Only 'smote' is supported.")

    steps.append(("resample", SMOTE(random_state=42, k_neighbors=5)))
    
    # Add estimator
    steps.append(("model", estimator))
    
    return ImbPipeline(steps=steps)


def analyze_class_balance(y: pd.Series | np.ndarray) -> dict:
    """Analyze class distribution and imbalance ratio.
    
    Args:
        y: Target variable
    
    Returns:
        Dictionary with class counts, percentages, and imbalance ratio
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    class_dist = {}
    for cls, count in zip(unique, counts):
        class_dist[int(cls)] = {
            "count": int(count),
            "percentage": float(count / total * 100),
        }
    
    imbalance_ratio = max(counts) / min(counts)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    return {
        "class_distribution": class_dist,
        "imbalance_ratio": float(imbalance_ratio),
        "minority_class": int(minority_class),
        "majority_class": int(majority_class),
        "total_samples": int(total),
    }


def compute_smote_samples_needed(y: pd.Series | np.ndarray, target_ratio: float = 1.0) -> int:
    """Calculate how many synthetic samples SMOTE should generate.
    
    Args:
        y: Target variable
        target_ratio: Desired minority/majority ratio (1.0 = balanced)
    
    Returns:
        Number of synthetic minority samples to generate
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    unique, counts = np.unique(y, return_counts=True)
    minority_count = min(counts)
    majority_count = max(counts)
    
    target_minority = int(majority_count * target_ratio)
    samples_needed = max(0, target_minority - minority_count)
    
    return samples_needed


def get_resampling_strategy(y: pd.Series | np.ndarray, target_ratio: float = 1.0) -> dict:
    """Generate sampling strategy dictionary for imbalanced-learn.
    
    Args:
        y: Target variable
        target_ratio: Desired minority/majority ratio
    
    Returns:
        Dictionary mapping class labels to target counts
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) != 2:
        raise ValueError("Only binary classification is supported")
    
    minority_idx = np.argmin(counts)
    majority_idx = np.argmax(counts)
    
    minority_class = unique[minority_idx]
    majority_class = unique[majority_idx]
    
    majority_count = counts[majority_idx]
    target_minority = int(majority_count * target_ratio)
    
    strategy = {
        int(majority_class): int(majority_count),
        int(minority_class): int(target_minority),
    }
    
    return strategy
