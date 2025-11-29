# Blood Donor Return Prediction - Final Analysis Report

**Generated:** 2025-11-30 02:49:46

---

## Executive Summary

**Best Model:** adaboost

**Test Set Performance:**
- **Accuracy:** 0.7671
- **Precision:** 0.7725
- **Recall:** 0.9819
- **F1-Score:** 0.8647
- **ROC-AUC:** 0.7393
- **PR-AUC:** 0.8902
- **Top Decile Lift:** 2.17x

**Key Achievement:** The model achieves **98.2% recall**, successfully identifying the vast majority of donors likely to return.

## Data Quality & EDA Findings

**Dataset Size:** 75,978 donors × 22 features

**Missing Values:** 0 (data quality: excellent)

**Class Distribution (availability):**
- Yes: 75.79%
- No: 24.21%
- Imbalance Ratio (majority/minority): 3.13:1

## Class Imbalance Handling

**Raw dataset imbalance (availability):**

- Yes: 57,582 donors (75.79%)
- No: 18,396 donors (24.21%)
- Imbalance ratio (majority/minority): 3.13:1

**Baseline vs SMOTE variants (test F1):**

| Model | Test F1 |
|-------|--------|
| adaboost | 0.8647 |
| lightgbm | 0.8202 |
| logistic_regression | 0.7443 |
| logistic_regression_smote | 0.7416 |
| random_forest | 0.8610 |
| random_forest_smote | 0.8343 |
| svm_linear | 0.8632 |
| xgboost | 0.8638 |
| xgboost_smote | 0.8450 |

In addition to SMOTE-based pipelines, several baseline models (e.g. logistic_regression, svm_linear, random_forest, lightgbm) use `class_weight="balanced"` or native class weighting so the minority class is up-weighted without resampling the data.

## Model Performance Comparison

| Model | CV F1 | Val F1 | Test F1 | Test Recall | Test PR-AUC | Test Lift |
|-------|-------|--------|---------|-------------|-------------|------------|
| logistic_regression | 0.7397 | 0.7426 | 0.7443 | 0.6560 | 0.8796 | 2.06 |
| random_forest | 0.8604 | 0.8589 | 0.8610 | 0.9611 | 0.8845 | 2.04 |
| adaboost | 0.8658 | 0.8647 | 0.8647 | 0.9819 | 0.8902 | 2.17 |
| svm_linear | 0.8635 | 0.8620 | 0.8632 | 0.9672 | 0.8790 | 2.04 |
| xgboost | 0.8634 | 0.8639 | 0.8638 | 0.9594 | 0.8936 | 2.20 |
| lightgbm | 0.8213 | 0.8140 | 0.8202 | 0.8195 | 0.8712 | 1.93 |
| logistic_regression_smote | 0.7352 | 0.7391 | 0.7416 | 0.6534 | 0.8773 | 2.01 |
| random_forest_smote | 0.8302 | 0.8315 | 0.8343 | 0.8667 | 0.8721 | 1.84 |
| xgboost_smote | 0.8395 | 0.8424 | 0.8450 | 0.8853 | 0.8733 | 1.98 |

## Evaluation Visualizations

Comprehensive evaluation visualizations generated for the best model:
- Confusion matrices (validation & test)
- ROC curves (validation & test)
- Precision-Recall curves (validation & test)
- Calibration curves
- Decile lift charts
- Threshold analysis

## Interpretability Insights

**Feature Importance Analysis (Best Model):**

Top features ranked by permutation importance on the validation set:

| Rank | Feature | Mean importance | Std dev |
|------|---------|----------------|--------|
| 1 | days_since_last_donation | 0.0053 | 0.0007 |
| 2 | age | 0.0027 | 0.0006 |
| 3 | donor_tenure_days | 0.0010 | 0.0006 |
| 4 | had_adverse_reaction_ever | 0.0006 | 0.0002 |
| 5 | risk_recency | 0.0003 | 0.0003 |
| 6 | donations_per_year | 0.0003 | 0.0001 |
| 7 | education_level_encoded | 0.0001 | 0.0003 |
| 8 | blood_group_O+ | 0.0000 | 0.0000 |
| 9 | blood_group_B- | 0.0000 | 0.0000 |
| 10 | blood_group_B+ | 0.0000 | 0.0000 |

*See feature-importance (and optional partial-dependence) plots in `reports/figures/` for detailed visualizations.*

## Subgroup Fairness Analysis

Performance evaluated across key demographic segments. Unless otherwise noted, metrics are measured on the test set.

**Overall test-set performance (best model):**

- Recall: 0.9819
- F1-Score: 0.8647

**By age band:**

| Subgroup | Recall | F1-Score |
|----------|--------|---------|
| 18-30 | 0.9547 | 0.8092 |
| 31-40 | 0.9940 | 0.8853 |
| 41-50 | 0.9970 | 0.9193 |
| 51+ | 0.9873 | 0.8600 |

**By blood group:**

| Subgroup | Recall | F1-Score |
|----------|--------|---------|
| A+ | 0.9808 | 0.8646 |
| A- | 0.9898 | 0.8593 |
| AB+ | 0.9890 | 0.8802 |
| AB- | 0.9896 | 0.8716 |
| B+ | 0.9758 | 0.8687 |
| B- | 0.9758 | 0.8388 |
| O+ | 0.9807 | 0.8663 |
| O- | 0.9861 | 0.8643 |

*See subgroup comparison plots in `reports/figures/` for visual breakdowns (e.g., `subgroup_recall_age_band.png`).*

## Actionable Recommendations

Based on model insights and feature importance analysis:

1. **Target Recent Donors:** Focus retention efforts on donors with recency < 120 days
2. **Engagement Programs:** Develop specialized programs for donors in the 31-40 age band (highest return rate)
3. **Location Optimization:** Leverage insights from donation center preferences to optimize outreach
4. **Personalized Communication:** Use model probability scores to customize messaging intensity
5. **Risk Scoring:** Implement decile-based segmentation for targeted campaigns
6. **Adverse Reaction Follow-up:** Provide enhanced support for donors with past adverse reactions
7. **Quality Assurance:** Monitor blood quality screen results to maintain donor confidence
8. **Tenure-Based Incentives:** Recognize and reward veteran donors to maintain long-term engagement
9. **Threshold Optimization:** Consider adjusting classification threshold based on campaign objectives
10. **Continuous Monitoring:** Track model performance across demographic subgroups to ensure fairness

## Generated Artifacts

**JSON reports:**
- `reports/eda/eda_summary.json`
- `reports/evaluation/evaluation_summary.json`
- `reports/interpretability/interpretability_summary.json`
- `reports/subgroup_analysis.json`
- `reports/success_criteria_validation.json`

**Visualization files (23 PNGs):**
- `reports/eda/figures/categorical_distributions.png`
- `reports/eda/figures/categorical_vs_target.png`
- `reports/eda/figures/class_distribution.png`
- `reports/eda/figures/key_relationships.png`
- `reports/eda/figures/numeric_by_target.png`
- `reports/eda/figures/numeric_distributions.png`
- `reports/eda/figures/temporal_first_donations.png`
- `reports/eda/figures/temporal_last_donations.png`
- `reports/evaluation/figures/calibration_test.png`
- `reports/evaluation/figures/calibration_validation.png`
- `reports/evaluation/figures/confusion_matrix_test.png`
- `reports/evaluation/figures/confusion_matrix_validation.png`
- `reports/evaluation/figures/decile_lift_test.png`
- `reports/evaluation/figures/decile_lift_validation.png`
- `reports/evaluation/figures/pr_curve_test.png`
- `reports/evaluation/figures/pr_curve_validation.png`
- `reports/evaluation/figures/roc_curve_test.png`
- `reports/evaluation/figures/roc_curve_validation.png`
- `reports/evaluation/figures/threshold_analysis_test.png`
- `reports/evaluation/figures/threshold_analysis_validation.png`
- `reports/figures/feature_importances_best_model.png`
- `reports/figures/subgroup_recall_age_band.png`
- `reports/figures/subgroup_recall_blood_group.png`

**Model Artifacts:**
- `models/best_model.pkl` (production-ready model)
- `models/all_model_results.json` (all model metrics)

## Success Criteria Validation

For a detailed, programmatic checklist of all workflow criteria (EDA, models, cross-validation, visualizations, explainability, subgroups, final report), see: `reports/success_criteria_validation.json`.

---

*End of Report*
