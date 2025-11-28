# Blood Donor Return Prediction

A comprehensive machine learning pipeline for predicting blood donor return likelihood, enabling targeted retention campaigns and optimized blood supply management.

---

## 📋 Overview

This project builds an end-to-end ML system that predicts whether blood donors will return for future donations based on their demographic profile, donation history, and behavioral patterns. The model helps blood banks:

- **Identify high-risk donors** requiring intervention
- **Prioritize outreach efforts** to maximize retention
- **Optimize resource allocation** for engagement programs
- **Improve long-term blood supply** stability

**Key Achievements:**
- 9 candidate models trained (baseline + SMOTE variants)
- 98.2% recall on donor return prediction (test set)
- 2.17× lift in top decile (test set ranking quality)
- Comprehensive fairness auditing across demographics
- Production-ready model with full interpretability

---

## 🏗️ Architecture

```
Blood Donor ML Pipeline
│
├── Data Generation
│   └── Synthetic dataset with realistic donation patterns
│
├── Exploratory Analysis
│   └── Raw data profiling → informs preprocessing strategy
│
├── Feature Engineering
│   └── Temporal, behavioral, demographic features
│
├── Model Training
│   ├── Baseline models (class_weight="balanced")
│   └── SMOTE variants (synthetic oversampling)
│
└── Post-Training Analysis
    ├── Performance evaluation (ROC, PR, calibration)
    ├── Interpretability (feature importance)
    ├── Fairness auditing (subgroup analysis)
    └── Automated reporting
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd PRML-Project-Blood-Donation

# Create environment
conda env create -f environment.yml
conda activate prml-project

# Or with pip
pip install -r requirements.txt
```

### Complete Workflow Execution

Run these commands in sequence:

```bash
# 1. Generate synthetic donor dataset
python Dataset.py

# 2. Exploratory data analysis (analyze raw data)
python eda.py

# 3. Feature engineering (based on EDA insights)
python process_data.py

# 4. Train all models and select the best
python -m src.pipeline.run

# 5. Generate comprehensive analysis reports
python -m src.analysis.run_all
```

**Total runtime:** ~10-20 minutes

---

## 📊 Workflow Details

### Step 1: Data Generation (`Dataset.py`)

Generates **75,978 synthetic donor records** with realistic constraints:

- **Demographics:** Age, gender, education, blood type
- **Donation history:** First/last donation dates, frequency, volumes
- **Quality metrics:** Adverse reactions, blood quality screens
- **Target variable:** Probabilistic availability based on behavioral scoring

**Output:** `blood_donor_dataset.csv`

**Key constraints enforced:**
- 12-week minimum interval between donations (medical safety)
- Age eligibility (≥18 years at first donation)
- Realistic blood group distribution (O+ 36%, A+ 28%, etc.)
- Temporal span: January 2020 - October 2025

---

### Step 2: Exploratory Data Analysis (`eda.py`)

Analyzes the **raw dataset BEFORE feature engineering** to inform preprocessing decisions.

**What it discovers:**
- Class distribution: 75.79% return, 24.21% don't (3.13:1 imbalance)
- Missing values: None (clean synthetic data)
- Outliers: ~2-5% in age/volume features
- Temporal patterns: Consistent donation trends over 33 months
- Correlations: Recency is the strongest predictor

**Output:** `reports/eda/` (15-20 visualizations + JSON summary)

**Key insights:**
- **Class imbalance detected** → Need SMOTE or class weights
- **No missing values** → No imputation needed
- **Outliers present** → Can be handled in pipelines
- **Recency matters** → Engineer days_since_last_donation feature

---

### Step 3: Feature Engineering (`process_data.py`)

Transforms raw data into model-ready features **based on EDA findings**.

**Operations:**
1. **Drop PII:** Remove donor_id, name, email, contact, password
2. **Engineer temporal features:**
   - All temporal variables are computed relative to a fixed snapshot `REFERENCE_DATE = 2025-11-30` so that both training and inference share the same "today".
   - `age` = (REFERENCE_DATE - date_of_birth) / 365.25
   - `days_since_last_donation` = days since last donation (recency)
   - `donor_tenure_days` = last_donation - first_donation (engagement span)
   - `mean_donation_interval_days` = tenure / (count - 1)
3. **Engineer volume features:**
   - `avg_donation_volume` = total_volume / number_donations
4. **Encode categoricals:**
   - Education: Ordinal encoding (1-5)
   - Gender: One-hot encoding (Male=1, Female=0)
   - Blood group: One-hot encoding (7 dummies, A+ baseline)
5. **Binary encoding:**
   - `had_adverse_reaction_ever` → 0/1
   - `last_blood_quality_screen` → Pass=1, Fail=0
6. **Create target:**
   - `target_available` = availability mapped to True/False

**Output:** `processed_donor_features.csv` (75,978 rows with engineered features and target)

**Core modeling feature set (features + target):**
- 16 numeric feature columns (age, recency, tenure, donation intervals, volumes, ratios, location counts, frequency, risk interaction)
- 11 indicator/ordinal feature columns (encoded education, gender and blood group one-hot indicators, reaction and quality flags)
- 1 boolean target (`target_available`)

---

### Step 4: Model Training (`src.pipeline.run`)

Trains **9 candidate models** with rigorous evaluation, selects the best based on performance criteria.

#### Candidates Trained

**Baseline Models (class_weight="balanced"):**
1. Logistic Regression
2. Random Forest (600 trees)
3. AdaBoost (400 estimators)
4. Linear SVM (calibrated)
5. XGBoost (400 trees, gradient boosting)
6. LightGBM (600 trees, optimized)

**SMOTE Variants (synthetic oversampling):**
7. Logistic Regression + SMOTE
8. Random Forest + SMOTE
9. XGBoost + SMOTE

#### Training Process

For each candidate:
1. **Load data:** Split into 70% train, 15% validation, 15% test (stratified)
2. **Cross-validation:** 4-fold stratified CV on training set
3. **Train:** Fit on full training set (53k samples)
4. **Evaluate:** Compute metrics on validation and test sets

#### Metrics Computed

- **Classification (per candidate):** Accuracy, Precision, Recall, F1-Score
- **Ranking (per candidate):** ROC-AUC, PR-AUC
- **Business (per candidate):** Decile lift (top 10% capture rate)
- **Calibration (best model only):** Brier score (computed in `src.analysis.evaluation`)

Calibration is computed only for the selected best model: candidates are ranked by recall and top-decile lift, while Brier is used as a calibration check in post-training analysis.

Cross-validation metrics printed in the console are reported as `mean ± standard deviation` across the 4 folds, so you can see both the average performance and how much it varies between folds.

#### Model Selection Criteria

1. **Primary metric:** Highest composite score on the validation set, defined as
   `0.5 × normalized_recall + 0.5 × normalized_lift`, where recall and decile lift are
   each min-max normalized across the 9 candidates.
2. **Tie-breakers:** If composite scores are effectively equal, prefer the model with
   higher validation recall; if recall is also tied, prefer higher validation lift.

**Output:**
- `models/best_model.pkl` (production-ready model)
- `models/all_model_results.json` (all metrics for all 9 models)

---

### Step 5: Post-Training Analysis (`src.analysis.run_all`)

Generates evaluation plots, interpretability outputs, subgroup metrics, and the final report/fairness checks.

#### 5.1 Model Evaluation (`src.analysis.evaluation`)

- Confusion matrices, ROC & PR curves, calibration plots, decile-lift charts, and threshold analysis for the best model.
- Saved under `reports/evaluation/figures/` with a summary JSON in `reports/evaluation/`.

#### 5.2 Interpretability Analysis (`src.analysis.interpretability`)

- Permutation-based feature importance for the best model on the validation set.
- Outputs `reports/figures/feature_importances_best_model.png` and `reports/interpretability/interpretability_summary.json`.

#### 5.3 Subgroup Fairness Analysis (`src.analysis.subgroups`)

- Evaluates recall/F1 across blood-group, gender, and age-band subgroups on the test set.
- Outputs `reports/subgroup_analysis.json` and `reports/figures/subgroup_*.png`.

#### 5.4 Automated Reporting (`src.analysis.report_generator`)

Synthesizes all findings into a comprehensive markdown report.

**Report sections:**
1. Executive summary (best model, key metrics)
2. EDA findings (data quality, class balance)
3. Model performance comparison (all 9 candidates)
4. Evaluation visualizations (ROC, PR, confusion matrices)
5. Interpretability insights (top features via feature importance)
6. Subgroup fairness analysis (demographic performance)
7. Actionable recommendations (10 strategic insights)
8. Generated artifacts (file paths)
9. Success criteria validation (requirements checklist)

**Output:** `reports/FINAL_REPORT.md`

#### 5.5 Success Validation (`src.analysis.validate_criteria`)

Automated verification that all project requirements are met:
- ✅ EDA completed
- ✅ All required models trained
- ✅ Cross-validation performed
- ✅ Visualizations generated
- ✅ Explainability implemented
- ✅ Subgroup analysis done
- ✅ Final report created

**Output:** `reports/success_criteria_validation.json`

---

## 📈 Results

### Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 76.7% |
| **Precision** | 77.2% |
| **Recall** | 98.2% |
| **F1-Score** | 86.5% |
| **ROC-AUC** | 73.9% |
| **PR-AUC** | 89.0% |
| **Top Decile Lift** | 2.17× |

### Key Features (by Importance)

1. **days_since_last_donation** (recency is critical)
2. **donor_tenure_days** (engagement history)
3. **age** (demographic factor)
4. **mean_donation_interval_days** (frequency pattern)
5. **donation_count_outram** (location preference)

### Fairness Audit

- ✅ No significant bias detected across blood groups
- ✅ Gender performance is equitable (Male ≈ Female)
- ✅ Age bands within 5% performance range

---

## 📁 Repository Structure

```
PRML-Project-Blood-Donation/
│
├── Dataset.py                      # Synthetic data generator
├── eda.py                          # Exploratory data analysis
├── process_data.py                 # Feature engineering pipeline
├── environment.yml                 # Conda environment spec
│
├── src/
│   ├── data/
│   │   └── splits.py              # Train/val/test partitioning
│   ├── features/
│   │   ├── preprocess.py          # Preprocessing utilities
│   │   └── imbalance.py           # SMOTE pipeline factory
│   ├── models/
│   │   ├── candidates.py          # Model definitions (9 candidates)
│   │   └── tune.py                # Hyperparameter tuning for candidate models
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation metrics
│   ├── pipeline/
│   │   └── run.py                 # Training orchestrator
│   └── analysis/
│       ├── evaluation.py          # Performance plots
│       ├── interpretability.py    # Feature-importance analysis 
│       ├── subgroups.py           # Fairness auditing 
│       ├── report_generator.py    # Markdown report builder
│       ├── validate_criteria.py   # Requirements checker
│       └── run_all.py             # Post-training orchestrator
│
├── models/
│   ├── best_model.pkl             # Production model
│   └── all_model_results.json     # All metrics
│
├── reports/
│   ├── eda/                       # EDA outputs
│   ├── evaluation/                # Evaluation plots
│   ├── figures/                   # Feature-importance and subgroup plots
│   ├── subgroup_analysis.json     # Fairness results
│   ├── FINAL_REPORT.md            # Comprehensive report
│   └── success_criteria_validation.json
│
└── blood_donor_dataset.csv        # Generated raw data
└── processed_donor_features.csv   # Engineered features
```

---

## 🔧 Advanced Usage

### Hyperparameter Tuning

To optimize model hyperparameters before final training (optional but recommended):

```bash
python -m src.models.tune
```
This performs RandomizedSearchCV followed by HalvingGridSearchCV for each candidate model, using 4-fold stratified cross-validation with F1-score as the optimization metric, and saves tuned estimators under `models/tuned/`.

Typical end-to-end training sequence:

```bash
python -m src.models.tune    # one-time (or occasional, optional but recommended) hyperparameter search
python -m src.pipeline.run   # train/evaluate all models and select the best
```

### Individual Analysis Modules

Run specific analyses without the full pipeline:

```bash
# Generate evaluation plots only
python -m src.analysis.evaluation

# Run interpretability analysis
python -m src.analysis.interpretability

# Perform subgroup fairness audit
python -m src.analysis.subgroups

# Generate final report
python -m src.analysis.report_generator

# Validate requirements
python -m src.analysis.validate_criteria
```

### Production Deployment

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

# New donor data (same 27 features: all feature columns except `target_available`)
new_donor = pd.DataFrame({
    'age': [35],
    'days_since_last_donation': [45],
    'donor_tenure_days': [730],
    # ... all 27 features ...
})

# Predict probability
prob = model.predict_proba(new_donor)[:, 1][0]
print(f"Return probability: {prob:.1%}")

# Classify
prediction = model.predict(new_donor)[0]
print(f"Predicted: {'Will return' if prediction == 1 else 'May not return'}")

# Use for targeted outreach
if prob > 0.8:
    print("Action: High-priority retention campaign")
elif prob > 0.5:
    print("Action: Standard outreach")
else:
    print("Action: Intensive engagement program")
```

---


## 🤝 Contributing

This is an academic project for PRML coursework. For questions or suggestions, please contact the project maintainers.

---

## 📄 License

This project is for educational purposes only.

---

## 🙏 Acknowledgments

- Synthetic data generation inspired by real-world blood bank operations
- Feature engineering guided by domain expertise in transfusion medicine
- Model selection criteria aligned with operational blood supply planning needs

---

## 📞 Support

For issues or questions:
1. Check the `reports/FINAL_REPORT.md` for detailed analysis
2. Review `reports/success_criteria_validation.json` for requirement checklist
3. Examine console logs during execution for debugging

---

**Built with ❤️ for improving blood supply management through machine learning**
