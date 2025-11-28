# **Blood Donor Return Prediction Workflow**

1. Generate or refresh the synthetic donor dataset `blood_donor_dataset.csv` with realistic donation patterns and availability labels (`Dataset.py`).
2. Run EDA on the raw synthetic dataset `blood_donor_dataset.csv` to understand structure, distributions, missing values, and class balance (`eda.py`).
3. Define preprocessing and scaling based on EDA, then implement it in the feature pipeline (`process_data.py`, `src/features/preprocess.py`).
4. Apply domain-specific feature engineering (temporal, behavioral, location-based features) to create model-ready columns (`process_data.py`).
5. Diagnose and mitigate class imbalance using class weights, SMOTE, and stratified sampling (`eda.py`, `src/features/imbalance.py`, `src/models/candidates.py`).
6. Create a stratified **train/validation/test** split (70/15/15) to preserve class distribution across all sets (`src/data/splits.py`).
7. Train (and optionally hyperparameter-tune) all candidate models with 4-fold stratified CV, including Logistic Regression, Linear SVM, Random Forest, AdaBoost, XGBoost, LightGBM, and SMOTE variants (`src/models/tune.py`, `src/pipeline/run.py`).
8. Evaluate all candidates on validation and test sets (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, top-decile lift) and select the best model using a recall + lift based rule (`src/pipeline/run.py`, `src/evaluation/metrics.py`).
9. Generate confusion matrices, ROC/PR curves, calibration plots, decile-lift charts, and threshold-analysis curves for the best model (`src/analysis/evaluation.py`).
10. Compute permutation-based feature importance for the best model to understand key drivers of predictions (`src/analysis/interpretability.py`).
11. Perform subgroup analysis on the best model to compare performance across age bands, gender, and blood groups (`src/analysis/subgroups.py`).
12. Turn the analysis into actionable recommendations for donor retention and long-term blood supply planning (`src/analysis/report_generator.py`).
13. Generate a final markdown report that consolidates all results, visualizations, and recommendations (`reports/FINAL_REPORT.md`, `src/analysis/report_generator.py`).
14. Validate that all seven success criteria are met and save a machine-readable checklist (`src/analysis/validate_criteria.py`, `reports/success_criteria_validation.json`).