"""Success Criteria Validator - validates all workflow.md requirements.

Checks:
1. EDA completed
2. All models trained
3. Cross-validation performed
4. Visualizations generated
5. Explainability implemented
6. Subgroup analysis done
7. Final report created
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def check_eda_completed() -> Dict[str, object]:
    """Criterion 1: Comprehensive EDA performed."""
    eda_dir = REPORTS_DIR / "eda"
    eda_summary = eda_dir / "eda_summary.json"
    figures_dir = eda_dir / "figures"
    
    passed = (
        eda_summary.exists() and
        figures_dir.exists() and
        len(list(figures_dir.glob("*.png"))) >= 5  # At least 5 visualizations
    )
    
    return {
        "criterion": "EDA - Comprehensive exploratory data analysis",
        "passed": passed,
        "evidence": {
            "eda_summary_exists": eda_summary.exists(),
            "figures_generated": len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0,
        }
    }


def check_all_models_trained() -> Dict[str, object]:
    """Criterion 2: All required models trained."""
    required_models = [
        "logistic_regression",
        "svm_linear",
        "random_forest",
        "xgboost",
        "lightgbm",
        "adaboost",
    ]
    
    pipeline_results = MODELS_DIR / "all_model_results.json"
    
    if not pipeline_results.exists():
        return {
            "criterion": "All required models trained",
            "passed": False,
            "evidence": {"pipeline_results_exists": False}
        }
    
    with open(pipeline_results, "r") as fp:
        raw_results = json.load(fp)

    # Support both new schema {"best_model": ..., "results": {...}} and
    # older flat schemas where the top level directly maps model_name → metrics.
    if isinstance(raw_results, dict) and "results" in raw_results and isinstance(
        raw_results["results"], dict
    ):
        results = raw_results["results"]
    else:
        results = {k: v for k, v in raw_results.items() if isinstance(v, dict)}

    trained_models = list(results.keys())
    missing_models = [m for m in required_models if m not in trained_models]
    
    passed = len(missing_models) == 0
    
    return {
        "criterion": "All required models trained (6 models)",
        "passed": passed,
        "evidence": {
            "trained_models": trained_models,
            "missing_models": missing_models,
            "count": len(trained_models),
        }
    }


def check_cross_validation_performed() -> Dict[str, object]:
    """Criterion 3: Cross-validation performed in the training pipeline.

    This project uses 4-fold stratified CV in both hyperparameter tuning
    (``src.models.tune``) and the main training pipeline (``src.pipeline.run``).
    This check simply verifies that cross_validation results are present in the
    saved model results JSON.
    """
    pipeline_results = MODELS_DIR / "all_model_results.json"
    
    if not pipeline_results.exists():
        return {
            "criterion": "Cross-validation performed in training pipeline",
            "passed": False,
            "evidence": {"pipeline_results_exists": False}
        }
    
    with open(pipeline_results, "r") as fp:
        raw_results = json.load(fp)

    # Support both new schema {"best_model": ..., "results": {...}} and
    # older flat schemas where the top level directly maps model_name → metrics.
    if isinstance(raw_results, dict) and "results" in raw_results and isinstance(
        raw_results["results"], dict
    ):
        results = raw_results["results"]
    else:
        results = {k: v for k, v in raw_results.items() if isinstance(v, dict)}

    # Check if any model has CV results
    has_cv = any("cross_validation" in model_results for model_results in results.values())
    
    return {
        "criterion": "Cross-validation performed in training pipeline",
        "passed": has_cv,
        "evidence": {
            "cross_validation_present": has_cv,
            "models_with_cv": [name for name, res in results.items() if "cross_validation" in res],
        }
    }


def check_visualizations_generated() -> Dict[str, object]:
    """Criterion 4: Key visualizations (confusion matrix, ROC, PR, calibration)."""
    required_viz_types = [
        "confusion_matrix",
        "roc_curve",
        "pr_curve",
        "calibration",
        "decile_lift",
    ]
    
    # Check evaluation figures
    eval_figures = REPORTS_DIR / "evaluation" / "figures"
    
    if not eval_figures.exists():
        return {
            "criterion": "Visualizations (CM, ROC, PR, calibration, decile lift)",
            "passed": False,
            "evidence": {"eval_figures_dir_exists": False}
        }
    
    generated_files = list(eval_figures.glob("*.png"))
    generated_names = [f.stem for f in generated_files]
    
    # Check if each type exists
    found_types = []
    for viz_type in required_viz_types:
        if any(viz_type in name for name in generated_names):
            found_types.append(viz_type)
    
    passed = len(found_types) >= len(required_viz_types)
    
    return {
        "criterion": "Visualizations (CM, ROC, PR, calibration, decile lift)",
        "passed": passed,
        "evidence": {
            "required_types": required_viz_types,
            "found_types": found_types,
            "total_figures": len(generated_files),
        }
    }


def check_explainability_implemented() -> Dict[str, object]:
    """Criterion 5: Explainability (feature importance and related plots)."""
    figures_dir = REPORTS_DIR / "figures"
    
    if not figures_dir.exists():
        # Try old location
        figures_dir = REPORTS_DIR / "evaluation" / "figures"
    
    core_required = ["feature_importances"]
    optional_artifacts = ["partial_dependence"]
    
    if not figures_dir.exists():
        return {
            "criterion": "Explainability (feature importance and related plots)",
            "passed": False,
            "evidence": {"figures_dir_exists": False}
        }
    
    generated_files = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.html"))
    generated_names = [f.stem for f in generated_files]
    
    found_core = []
    for artifact in core_required:
        if any(artifact in name for name in generated_names):
            found_core.append(artifact)
    
    found_optional = []
    for artifact in optional_artifacts:
        if any(artifact in name for name in generated_names):
            found_optional.append(artifact)
    
    # Pass criterion if core feature-importance artifact exists; additional plots are optional.
    passed = len(found_core) == len(core_required)
    
    return {
        "criterion": "Explainability (feature importance and related plots)",
        "passed": passed,
        "evidence": {
            "core_required": core_required,
            "found_core": found_core,
            "optional_artifacts": optional_artifacts,
            "found_optional": found_optional,
        }
    }


def check_subgroup_analysis_done() -> Dict[str, object]:
    """Criterion 6: Subgroup analysis across demographics."""
    subgroup_summary = REPORTS_DIR / "subgroup_analysis.json"
    figures_dir = REPORTS_DIR / "figures"
    
    if not subgroup_summary.exists():
        return {
            "criterion": "Subgroup analysis (demographics)",
            "passed": False,
            "evidence": {"subgroup_summary_exists": False}
        }
    
    with open(subgroup_summary, "r") as fp:
        results = json.load(fp)
    
    required_dimensions = ["blood_group", "gender", "age_band"]
    subgroup_results = results.get("subgroup_results", {})
    found_dimensions = [dim for dim in required_dimensions if dim in subgroup_results]
    
    # Check for subgroup plots
    subgroup_plots = 0
    if figures_dir.exists():
        subgroup_plots = len(list(figures_dir.glob("subgroup_*.png")))
    
    passed = len(found_dimensions) >= len(required_dimensions)
    
    return {
        "criterion": "Subgroup analysis (demographics)",
        "passed": passed,
        "evidence": {
            "required_dimensions": required_dimensions,
            "found_dimensions": found_dimensions,
            "subgroup_plots_generated": subgroup_plots,
        }
    }


def check_final_report_generated() -> Dict[str, object]:
    """Criterion 7: Final comprehensive report."""
    final_report = REPORTS_DIR / "FINAL_REPORT.md"
    
    passed = final_report.exists()
    
    size = 0
    if passed:
        size = final_report.stat().st_size
    
    return {
        "criterion": "Final comprehensive report",
        "passed": passed,
        "evidence": {
            "report_exists": passed,
            "report_size_bytes": size,
        }
    }


def validate_all_criteria() -> Dict[str, object]:
    """Run all success criteria validations."""
    print("=" * 60)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 60)
    
    criteria_checks = [
        ("1", check_eda_completed),
        ("2", check_all_models_trained),
        ("3", check_cross_validation_performed),
        ("4", check_visualizations_generated),
        ("5", check_explainability_implemented),
        ("6", check_subgroup_analysis_done),
        ("7", check_final_report_generated),
    ]
    
    results = {}
    passed_count = 0
    
    for criterion_id, check_func in criteria_checks:
        print(f"\n  Checking Criterion {criterion_id}...")
        result = check_func()
        results[criterion_id] = result
        
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"    {status}: {result['criterion']}")
        
        if result["passed"]:
            passed_count += 1
    
    summary = {
        "validation_results": results,
        "total_criteria": len(criteria_checks),
        "passed_criteria": passed_count,
        "all_passed": passed_count == len(criteria_checks),
    }
    
    # Save results
    output_path = REPORTS_DIR / "success_criteria_validation.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nResults: {passed_count}/{len(criteria_checks)} criteria passed")
    print(f"Validation report saved to: {output_path}")
    
    if summary["all_passed"]:
        print("\n🎉 ALL SUCCESS CRITERIA MET! 🎉")
    else:
        print("\n⚠️  Some criteria not met. Review evidence in validation report.")
    
    return summary


if __name__ == "__main__":
    validate_all_criteria()
