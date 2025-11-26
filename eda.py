"""Comprehensive Exploratory Data Analysis (EDA) module.

IMPORTANT: Run this on RAW data BEFORE feature engineering to inform preprocessing decisions.

Generates visualizations and statistical summaries to understand:
- Data structure and quality (before transformations)
- Feature distributions (raw features)
- Class balance (to decide on imbalance handling)
- Missing values and outliers (to guide cleaning strategy)
- Temporal patterns (to engineer time-based features)

Insights from this EDA should inform process_data.py decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

EDA_DIR = Path("reports/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = EDA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = EDA_DIR / "eda_summary.json"


def load_raw_data(file_path: Path | str = Path("blood_donor_dataset.csv")) -> pd.DataFrame:
    """Load the raw donor dataset for exploratory analysis."""
    return pd.read_csv(file_path)


def data_overview(df: pd.DataFrame) -> Dict[str, object]:
    """Generate dataset overview statistics."""
    overview = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "column_names": df.columns.tolist(),
    }
    return overview


def missing_values_analysis(df: pd.DataFrame) -> Dict[str, object]:
    """Analyze missing values."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    result = {
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
        "missing_percentage": missing_pct[missing_pct > 0].to_dict(),
    }
    
    # Visualize if there are missing values
    if result["total_missing_values"] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_pct[missing_pct > 0].sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="coral"
        )
        ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
        ax.set_ylabel("Percentage Missing (%)")
        ax.set_xlabel("Column")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missing_values.png")
        plt.close()
    
    return result


def class_distribution_analysis(df: pd.DataFrame, target_col: str = "availability") -> Dict[str, object]:
    """Analyze target variable distribution."""
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found"}
    
    value_counts = df[target_col].value_counts()
    value_pct = (value_counts / len(df) * 100).round(2)
    
    result = {
        "counts": value_counts.to_dict(),
        "percentages": value_pct.to_dict(),
        "imbalance_ratio": float(value_counts.max() / value_counts.min()),
    }
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    value_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
    axes[0].set_title("Class Distribution (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel(target_col.replace("_", " ").title())
    for i, v in enumerate(value_counts):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")
    
    # Pie chart
    axes[1].pie(
        value_counts, labels=value_counts.index, autopct="%1.1f%%",
        colors=["steelblue", "coral"], startangle=90
    )
    axes[1].set_title("Class Distribution (Percentage)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close()
    
    return result


def numeric_distributions(df: pd.DataFrame, exclude_cols: List[str] = None) -> None:
    """Plot distributions for all numeric columns."""
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not numeric_cols:
        return
    
    n_cols = min(3, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel("Frequency")
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png")
    plt.close()


def categorical_distributions(df: pd.DataFrame, cat_cols: List[str] = None) -> None:
    """Plot distributions for categorical columns."""
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if not cat_cols:
        return
    
    n_cols = min(2, len(cat_cols))
    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(cat_cols):
        if idx < len(axes):
            value_counts = df[col].value_counts()
            axes[idx].bar(range(len(value_counts)), value_counts.values, color="coral", edgecolor="black")
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha="right")
            axes[idx].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
            axes[idx].set_ylabel("Count")
            axes[idx].grid(alpha=0.3, axis="y")
    
    # Hide unused subplots
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "categorical_distributions.png")
    plt.close()


def correlation_analysis(df: pd.DataFrame, target_col: str = "target_available") -> Dict[str, object]:
    """Analyze correlations between features and with target."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {"error": "No numeric columns found"}
    
    corr_matrix = numeric_df.corr()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, annot=False, cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()
    
    result = {}
    
    # Top correlations with target
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col, errors="ignore").sort_values(ascending=False)
        result["top_positive_correlations"] = target_corr.head(10).to_dict()
        result["top_negative_correlations"] = target_corr.tail(10).to_dict()
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = pd.concat([target_corr.head(15), target_corr.tail(5)]).sort_values()
        colors = ["coral" if x < 0 else "steelblue" for x in top_features]
        top_features.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Top Features Correlated with {target_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Correlation Coefficient")
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "target_correlations.png")
        plt.close()
    
    return result


def outlier_analysis(df: pd.DataFrame) -> Dict[str, object]:
    """Detect outliers using IQR method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_summary = {}
    outlier_plots_cols = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)
        outlier_pct = (n_outliers / len(df) * 100)
        
        if n_outliers > 0:
            outlier_summary[col] = {
                "count": int(n_outliers),
                "percentage": float(round(outlier_pct, 2)),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
            outlier_plots_cols.append(col)
    
    # Box plots for columns with outliers
    if outlier_plots_cols:
        n_cols = 3
        n_rows = int(np.ceil(len(outlier_plots_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(outlier_plots_cols[:len(axes)]):
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(f"{col}\n({outlier_summary[col]['count']} outliers)", fontsize=10)
            axes[idx].set_ylabel(col.replace("_", " ").title())
            axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(outlier_plots_cols), len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "outlier_boxplots.png")
        plt.close()
    
    return outlier_summary


def temporal_analysis(df: pd.DataFrame) -> None:
    """Analyze temporal patterns in donation data."""
    date_cols = []
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass
    
    if not date_cols:
        return
    
    # Donations over time
    if "first_donation_date" in date_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        df["first_donation_date"].dt.to_period("M").value_counts().sort_index().plot(
            kind="line", ax=ax, marker="o", color="steelblue"
        )
        ax.set_title("First Donations Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of First Donations")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "temporal_first_donations.png")
        plt.close()
    
    if "last_donation_date" in date_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        df["last_donation_date"].dt.to_period("M").value_counts().sort_index().plot(
            kind="line", ax=ax, marker="o", color="coral"
        )
        ax.set_title("Last Donations Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Last Donations")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "temporal_last_donations.png")
        plt.close()


def numeric_by_target(df: pd.DataFrame, target_col: str = "availability") -> None:
    """Plot numeric feature distributions stratified by target class."""
    if target_col not in df.columns:
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_numerics = ['number_of_donation', 'total_donation_volume_ml', 'last_donation_volume_ml']
    key_numerics = [col for col in key_numerics if col in numeric_cols]
    
    if not key_numerics:
        return
    
    n_cols = 2
    n_rows = int(np.ceil(len(key_numerics) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(key_numerics):
        if idx < len(axes):
            # Overlapping histograms
            for target_val in df[target_col].unique():
                subset = df[df[target_col] == target_val][col].dropna()
                axes[idx].hist(subset, bins=30, alpha=0.6, 
                             label=f"{target_col}={target_val}",
                             edgecolor="black", linewidth=0.5)
            
            axes[idx].set_title(f"{col} by {target_col}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel("Frequency")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(key_numerics), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_by_target.png", dpi=300)
    plt.close()


def categorical_vs_target(df: pd.DataFrame, target_col: str = "availability") -> None:
    """Plot categorical variables vs target to show relationships."""
    if target_col not in df.columns:
        return
    
    cat_cols = ['gender', 'blood_group', 'education_level', 'last_blood_quality_screen']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    if not cat_cols:
        return
    
    n_cols = 2
    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(cat_cols):
        if idx < len(axes):
            # Crosstab for stacked bar chart
            ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            ct.plot(kind='bar', stacked=True, ax=axes[idx], 
                   color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=0.5)
            axes[idx].set_title(f"{col} vs {target_col}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel("Percentage (%)")
            axes[idx].legend(title=target_col, loc='best')
            axes[idx].grid(alpha=0.3, axis='y')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=300)
    plt.close()


def key_relationships(df: pd.DataFrame) -> None:
    """Plot availability rate heatmaps for key feature pairs."""

    def _plot_rate_heatmap(data: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes,
                           x_label: str, y_label: str, title: str,
                           x_bins: int | np.ndarray = 20,
                           y_bins: int | np.ndarray = 20) -> None:
        cols = [x_col, y_col, 'availability']
        if not set(cols).issubset(data.columns):
            ax.axis('off')
            ax.set_title(f"Missing columns for {title}")
            return

        plot_df = data[cols].dropna()
        if plot_df.empty:
            ax.axis('off')
            ax.set_title(f"No data for {title}")
            return

        x_vals = plot_df[x_col].to_numpy()
        y_vals = plot_df[y_col].to_numpy()
        availability_vals = (plot_df['availability'] == 'Yes').astype(float).to_numpy()

        if isinstance(x_bins, int):
            x_bins = np.linspace(x_vals.min(), x_vals.max(), x_bins + 1)
        if isinstance(y_bins, int):
            y_bins = np.linspace(y_vals.min(), y_vals.max(), y_bins + 1)

        counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_bins, y_bins])
        yes_counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_bins, y_bins],
                                          weights=availability_vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.divide(yes_counts, counts, out=np.zeros_like(yes_counts), where=counts > 0)
        mask = counts <= 0
        rates = np.nan_to_num(rates)
        rates_masked = np.ma.masked_array(rates, mask=mask)

        cmap = plt.colormaps['RdYlGn'].copy()
        cmap.set_bad(color='lightgrey')

        mesh = ax.pcolormesh(x_bins, y_bins, rates_masked.T, cmap=cmap, vmin=0, vmax=1, shading='auto')
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        subtitle = f"{title}\n(gray = no donors)"
        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.2)
        plt.colorbar(mesh, ax=ax, label='Availability rate (Yes share)')

    if 'date_of_birth' in df.columns:
        df['age'] = (pd.Timestamp('2025-11-30') - pd.to_datetime(df['date_of_birth'])).dt.days / 365.25

    if 'last_donation_date' in df.columns:
        df['days_since_last'] = (pd.Timestamp('2025-11-30') - pd.to_datetime(df['last_donation_date'])).dt.days

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    _plot_rate_heatmap(
        df, 'number_of_donation', 'total_donation_volume_ml', axes[0],
        'Number of Donations', 'Total Volume (ml)',
        'Availability rate vs donations & volume', x_bins=np.arange(1, 22), y_bins=20,
    )

    _plot_rate_heatmap(
        df, 'age', 'number_of_donation', axes[1],
        'Age (years)', 'Number of Donations',
        'Availability rate vs age & frequency', x_bins=20, y_bins=np.arange(1, 22),
    )

    _plot_rate_heatmap(
        df, 'days_since_last', 'number_of_donation', axes[2],
        'Days Since Last Donation', 'Number of Donations',
        'Availability rate vs recency & frequency', x_bins=20, y_bins=np.arange(1, 22),
    )

    if 'total_donation_volume_ml' in df.columns and 'number_of_donation' in df.columns:
        df['avg_volume'] = df['total_donation_volume_ml'] / df['number_of_donation']

    _plot_rate_heatmap(
        df, 'avg_volume', 'number_of_donation', axes[3],
        'Average Volume per Donation (ml)', 'Number of Donations',
        'Availability rate vs avg volume & frequency', x_bins=20, y_bins=np.arange(1, 22),
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "key_relationships.png", dpi=300)
    plt.close()


def run_eda_analysis() -> Dict[str, object]:
    """Run complete EDA pipeline on RAW data BEFORE feature engineering.
    
    This analysis informs preprocessing decisions in process_data.py:
    - Class imbalance → use SMOTE/class weights
    - Missing values → imputation strategy
    - Outliers → capping/winsorization
    - Distributions → scaling/normalization needs
    - Temporal patterns → feature engineering ideas
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (RAW DATA)")
    print("=" * 60)
    
    # Load ONLY raw data
    print("\nLoading raw dataset...")
    df_raw = load_raw_data()
    
    summary = {}
    
    # Data overview
    print("  [1/9] Generating data overview...")
    summary["data_overview"] = data_overview(df_raw)
    
    # Missing values
    print("  [2/9] Analyzing missing values...")
    summary["missing_values"] = missing_values_analysis(df_raw)
    
    # Class distribution (CRITICAL for imbalance handling)
    print("  [3/9] Analyzing class distribution...")
    summary["class_distribution"] = class_distribution_analysis(df_raw, "availability")
    
    # Numeric distributions
    print("  [4/9] Plotting numeric feature distributions...")
    numeric_distributions(df_raw, exclude_cols=["donor_id"])
    
    # Categorical distributions
    print("  [5/9] Plotting categorical feature distributions...")
    categorical_distributions(
        df_raw, cat_cols=["gender", "blood_group", "education_level", "availability"]
    )
    
    # Temporal patterns (inform feature engineering)
    print("  [6/9] Analyzing temporal patterns...")
    temporal_analysis(df_raw)
    
    # Numeric features stratified by target
    print("  [7/9] Plotting numeric features by target class...")
    numeric_by_target(df_raw, "availability")
    
    # Categorical vs target relationships
    print("  [8/9] Analyzing categorical vs target relationships...")
    categorical_vs_target(df_raw, "availability")
    
    # Key feature relationships
    print("  [9/9] Plotting key feature relationships...")
    key_relationships(df_raw.copy())
    
    # Save summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    
    # Print comprehensive key findings
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    # Dataset size
    n_rows = summary['data_overview']['shape']['rows']
    n_cols = summary['data_overview']['shape']['columns']
    print(f"\n📊 Dataset: {n_rows:,} rows × {n_cols} columns")
    
    # Data types breakdown
    dtypes = summary['data_overview']['dtypes']
    n_numeric = sum(1 for dt in dtypes.values() if 'int' in dt or 'float' in dt)
    n_object = sum(1 for dt in dtypes.values() if 'object' in dt)
    print(f"   - Numeric features: {n_numeric}")
    print(f"   - Categorical features: {n_object}")
    
    # Missing values
    print(f"\n🔍 Data Quality:")
    print(f"   - Missing values: {summary['missing_values']['total_missing_values']}")
    
    # Outliers
    outliers = summary.get("outlier_analysis", {})
    if outliers:
        total_outliers = sum(info['count'] for info in outliers.values())
        print(f"   - Features with outliers: {len(outliers)}")
        print(f"   - Total outlier records: {total_outliers:,}")
    
    # Class distribution
    class_dist = summary["class_distribution"]
    print(f"\n⚖️  Target Variable (availability):")
    if "percentages" in class_dist:
        for k, v in class_dist["percentages"].items():
            count = class_dist["counts"][k]
            print(f"   - '{k}': {count:,} ({v}%)")
        imbalance_ratio = class_dist['imbalance_ratio']
        print(f"   - Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.0:
            print(f"   → RECOMMENDATION: Use SMOTE or class_weight='balanced'")
    
    # Categorical variable details
    print(f"\n🏷️  Categorical Variables & Encoding Strategy:")
    
    # Gender
    if 'gender' in df_raw.columns:
        gender_vals = df_raw['gender'].unique().tolist()
        print(f"   - gender: {gender_vals}")
        print(f"     → Strategy: One-hot encoding with drop_first=True")
        print(f"     → Reason: Nominal (no order), 2 categories → creates 1 binary feature")
        print(f"     → Avoids dummy variable trap (multicollinearity for linear models)")
    
    # Blood group
    if 'blood_group' in df_raw.columns:
        blood_vals = sorted(df_raw['blood_group'].unique().tolist())
        print(f"   - blood_group: {blood_vals}")
        print(f"     → Strategy: One-hot encoding with drop_first=True")
        print(f"     → Reason: Nominal (no order), {len(blood_vals)} categories → creates {len(blood_vals)-1} binary features")
        print(f"     → Drop first category as reference to avoid multicollinearity")
    
    # Education
    if 'education_level' in df_raw.columns:
        edu_vals = df_raw['education_level'].unique().tolist()
        edu_order = ['Primary', 'Secondary', 'Diploma', 'University', 'Postgraduate']
        print(f"   - education_level: {edu_vals}")
        print(f"     → Strategy: Ordinal encoding")
        print(f"     → Mapping: {', '.join([f'{i+1}={e}' for i, e in enumerate(edu_order)])}")
        print(f"     → Reason: ORDERED categories with inherent ranking")
        print(f"     → Preserves ordinality while avoiding {len(edu_vals)-1} dummy variables")
    
    # Suggested derived features
    print(f"\n💡 Suggested Derived Features (Low Multicollinearity Risk):")
    print(f"   1. Recency ratio:")
    print(f"      • recency_ratio = days_since_last_donation / donor_tenure_days")
    print(f"      • Captures recent vs stale engagement pattern")
    print(f"   2. Volume consistency:")
    print(f"      • volume_consistency = last_donation_volume_ml / avg_donation_volume")
    print(f"      • Detects deviation from donor's personal baseline")
    print(f"   3. Location loyalty:")
    print(f"      • primary_location_pct = max(location_counts) / sum(location_counts)")
    print(f"      • Measures donor's location preference strength")
    print(f"   4. Donation rate:")
    print(f"      • donations_per_year = number_of_donation / (donor_tenure_days / 365.25)")
    print(f"      • Annual donation frequency")
    print(f"   5. Risk-recency interaction:")
    print(f"      • risk_recency = had_adverse_reaction_ever × days_since_last_donation")
    print(f"      • Higher value = risky donor who hasn't donated recently")
    print(f"\n   ⚠️  AVOID for linear models (causes multicollinearity):")
    print(f"      ✗ Polynomial features: age², number_of_donation² (redundant with originals)")
    print(f"      ✗ Reciprocal pairs: A/B and B/A both present")
    print(f"      ✗ Sum + parts: total_donations AND each location_count simultaneously")
    print(f"      ✗ Highly correlated pairs: total_volume + number_donations (r > 0.9)")
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Total visualizations: {len(list(FIGURES_DIR.glob('*.png')))}")
    print("\n" + "="*60)
    print("NEXT STEP: Run 'python process_data.py' to engineer features.")
    print("="*60)
    
    return summary


if __name__ == "__main__":
    run_eda_analysis()
