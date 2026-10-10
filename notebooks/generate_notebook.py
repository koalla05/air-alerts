import json
from pathlib import Path

def create_notebook():
    project_root = Path(__file__).resolve().parent.parent
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""# Ukraine Air Raid Alerts: Model Training & Evaluation Pipeline
> **Implementation aligned with PLAN.md (Phase 1 MVP)**
>
> **Task A (Active Alert):** Predict remaining duration ($P10, P50, P90$) and probability of alert ending within 15 / 30 / 60 minutes given elapsed time $t$.
>
> **Task B (Quiet Region):** Predict time until next alert ($P10, P50, P90$) and probability of alert starting within 1 / 3 / 6 hours given quiet elapsed time $t_{quiet}$.
>
> *Disclaimer: Statistical experiment for research and analysis. Never replaces official alert sirens or civil defense alerts.*""")

    # Section 1: Environment & Imports
    add_md("## 1. Setup & Imports")
    add_code("""%load_ext autoreload
%autoreload 2

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to python path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lifelines import KaplanMeierFitter

# Import project modules
from alertpred.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REGION_MAPPING,
    OBLAST_NEIGHBORS,
    TASK_A_HORIZONS_MIN,
    TASK_B_HORIZONS_MIN,
)
from alertpred.data.cleaning import (
    load_raw_data,
    clean_alerts,
    build_gap_episodes,
    create_time_split,
)
from alertpred.features.builder import (
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
)
from alertpred.models.baseline import (
    EmpiricalConditionalBaseline,
    KaplanMeierBaseline,
)
from alertpred.models.quantile_lgbm import (
    LightGBMQuantileModel,
    LightGBMProbabilityModel,
    AlertPredictionPipeline,
)
from alertpred.evaluation.metrics import (
    evaluate_quantile_predictions,
    evaluate_classification_probabilities,
)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print(f"Project root: {project_root}")
print(f"Raw data path: {RAW_DATA_PATH} (exists: {RAW_DATA_PATH.exists()})")""")

    # Section 2: Data Ingestion & Cleaning
    add_md("""## 2. Data Ingestion & Cleaning
According to **PLAN.md Section 3.4**, raw alerts data requires:
1. Filtering by `level == 'oblast'` for macroscopic regional forecasting.
2. Parsing timestamps to UTC.
3. Filtering out technical glitches (< 1 minute) and unbounded anomalies.
4. Deduplicating overlapping intervals.
5. Mapping Ukrainian oblast names to standardized IDs.""")

    add_code("""# Load raw data
df_raw = load_raw_data(RAW_DATA_PATH)
print(f"Total raw records: {len(df_raw):,}")
display(df_raw.head())""")

    add_code("""# Clean alerts
df_alerts = clean_alerts(df_raw, filter_level='oblast', min_duration_sec=60)
print(f"Cleaned alert episodes: {len(df_alerts):,}")
print(f"Number of regions: {df_alerts['region_id'].nunique()}")
print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")
display(df_alerts.head())""")

    add_code("""# Build Task B Quiet Gap episodes (time between consecutive alerts per region)
df_gaps = build_gap_episodes(df_alerts)
print(f"Quiet gap episodes: {len(df_gaps):,}")
display(df_gaps.head())""")

    # Section 3: Expanded EDA
    add_md("""## 3. Exploratory Data Analysis (EDA)
Comprehensive exploration of:
1. **Alert Frequency Across Oblasts:** Ranking, volume share, and monthly temporal intensity.
2. **Alert Duration Distributions (Mean vs. Median):** Skewness, dispersion, and diurnal hourly patterns.
3. **Spatial Context & Neighbor Co-Occurrence:** Simultaneous alert rates among direct neighbors, pairwise correlation heatmap, and propagation trajectory analysis.
4. **Quiet Gaps (Task B):** Inter-arrival quiet durations across frontline vs. rear oblasts.""")

    add_code("""# 3.1 Alert Frequency Analysis by Oblast
total_days = (df_alerts['started_at'].max() - df_alerts['started_at'].min()).total_seconds() / (24 * 3600)
months_approx = total_days / 30.4375

freq_stats = df_alerts.groupby('region_en').size().reset_index(name='total_alerts')
freq_stats['alerts_per_month'] = freq_stats['total_alerts'] / months_approx
freq_stats['alerts_per_week'] = freq_stats['total_alerts'] / (total_days / 7.0)
freq_stats['pct_national'] = 100.0 * freq_stats['total_alerts'] / len(df_alerts)
freq_stats = freq_stats.sort_values(by='total_alerts', ascending=False).reset_index(drop=True)

display(freq_stats)

# Plot Horizontal Bar Chart of Alert Frequency
plt.figure(figsize=(12, 8))
sns.barplot(
    data=freq_stats,
    y='region_en',
    x='total_alerts',
    palette='viridis'
)
plt.title(f"Total Alert Frequency Across Ukrainian Oblasts (Total: {len(df_alerts):,} episodes)")
plt.xlabel("Total Alert Count")
plt.ylabel("")
for i, v in enumerate(freq_stats['total_alerts']):
    plt.text(v + 60, i, f"{v:,} ({freq_stats['pct_national'].iloc[i]:.1f}%)", va='center', fontsize=9)
plt.tight_layout()
plt.show()

# Monthly Timeline of Nationwide Alert Counts
df_alerts['year_month'] = df_alerts['started_at'].dt.to_period('M').astype(str)
monthly_counts = df_alerts.groupby('year_month').size()

plt.figure(figsize=(14, 4))
monthly_counts.plot(kind='bar', color='#1f77b4', edgecolor='black', alpha=0.85)
plt.title("National Monthly Alert Episode Volume (2022 - 2026)")
plt.xlabel("Year - Month")
plt.ylabel("Monthly Alerts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")

    add_code("""# 3.2 Duration Analysis: Mean vs. Median & Dispersion
duration_stats = df_alerts.groupby('region_en')['duration_min'].agg(
    count='count',
    mean_min='mean',
    median_min='median',
    std_min='std',
    p10_min=lambda s: np.percentile(s, 10),
    p90_min=lambda s: np.percentile(s, 90),
    iqr_min=lambda s: np.percentile(s, 75) - np.percentile(s, 25),
    max_hours=lambda s: s.max() / 60.0
).sort_values(by='mean_min', ascending=False)

display(duration_stats.round(2))

# Side-by-Side Comparison: Mean vs. Median Duration
dur_plot_df = duration_stats.reset_index().melt(
    id_vars='region_en',
    value_vars=['mean_min', 'median_min'],
    var_name='Metric',
    value_name='Duration (min)'
)
dur_plot_df['Metric'] = dur_plot_df['Metric'].map({'mean_min': 'Mean Duration', 'median_min': 'Median Duration'})

plt.figure(figsize=(12, 9))
sns.barplot(
    data=dur_plot_df,
    y='region_en',
    x='Duration (min)',
    hue='Metric',
    palette=['#d95f02', '#1b9e77']
)
plt.title("Mean vs. Median Alert Duration by Oblast (Highlighting Right-Skewed Multi-Hour Alerts)")
plt.xlabel("Duration in Minutes")
plt.ylabel("")
plt.legend(title="")
plt.tight_layout()
plt.show()

# Boxplot across Macro-Regions & Diurnal Hour-of-Day Pattern
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sample_regions = ['Donetsk Oblast', 'Kharkiv Oblast', 'Kyiv City', 'Odesa Oblast', 'Lviv Oblast', 'Zakarpattia Oblast']
sns.boxplot(
    data=df_alerts[df_alerts['region_en'].isin(sample_regions) & (df_alerts['duration_min'] <= 240)],
    x='region_en',
    y='duration_min',
    palette='Set2',
    ax=axes[0]
)
axes[0].set_title("Alert Duration Distribution (<= 4 Hours)")
axes[0].set_ylabel("Duration (minutes)")
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', rotation=25)

# Diurnal Pattern (Hour of Day in Europe/Kyiv Local Time)
df_alerts['start_kyiv_hour'] = df_alerts['started_at'].dt.tz_convert('Europe/Kyiv').dt.hour
sns.histplot(df_alerts['start_kyiv_hour'], bins=24, kde=True, color='crimson', ax=axes[1])
axes[1].set_title("Alert Starts by Hour of Day (Europe/Kyiv Local Time)")
axes[1].set_xlabel("Local Hour (0 - 23)")
axes[1].set_ylabel("Total Alert Count")
plt.tight_layout()
plt.show()""")

    add_code("""# 3.3 Spatial Context: Simultaneous Alerts in Neighboring Oblasts
start_time = df_alerts['started_at'].min().floor('h')
end_time = df_alerts['finished_at'].max().ceil('h')
hourly_grid = pd.date_range(start_time, end_time, freq='1h', tz='UTC')

region_ids = sorted(df_alerts['region_id'].unique())
reg_to_idx = {r: i for i, r in enumerate(region_ids)}
matrix = np.zeros((len(hourly_grid), len(region_ids)), dtype=bool)

for _, row in df_alerts.iterrows():
    c = reg_to_idx[row['region_id']]
    s_idx = max(0, int((row['started_at'] - start_time).total_seconds() // 3600))
    e_idx = min(len(hourly_grid), int((row['finished_at'] - start_time).total_seconds() // 3600) + 1)
    matrix[s_idx:e_idx, c] = True

# Compute Simultaneous Alert Percentage with Direct Neighbors
neighbor_sim_results = []
for reg in region_ids:
    c = reg_to_idx[reg]
    reg_active = matrix[:, c]
    total_active_hours = int(reg_active.sum())
    
    direct_neighbors = [n for n in OBLAST_NEIGHBORS.get(reg, []) if n in reg_to_idx]
    if not direct_neighbors or total_active_hours == 0:
        continue
    
    n_indices = [reg_to_idx[n] for n in direct_neighbors]
    any_neighbor_active = matrix[:, n_indices].any(axis=1)
    co_active_hours = int((reg_active & any_neighbor_active).sum())
    
    reg_en = df_alerts[df_alerts['region_id'] == reg]['region_en'].iloc[0]
    
    neighbor_sim_results.append({
        'region_id': reg,
        'region_en': reg_en,
        'active_hours': total_active_hours,
        'co_active_neighbor_hours': co_active_hours,
        'pct_neighbor_simultaneous': 100.0 * co_active_hours / total_active_hours,
        'neighbor_count': len(direct_neighbors),
    })

df_neighbor_sim = pd.DataFrame(neighbor_sim_results).sort_values('pct_neighbor_simultaneous', ascending=False).reset_index(drop=True)
display(df_neighbor_sim[['region_en', 'active_hours', 'co_active_neighbor_hours', 'pct_neighbor_simultaneous', 'neighbor_count']])

# Bar Plot: Neighbor Co-Occurrence Percentage
plt.figure(figsize=(12, 8))
sns.barplot(
    data=df_neighbor_sim,
    y='region_en',
    x='pct_neighbor_simultaneous',
    palette='mako'
)
mean_sim = df_neighbor_sim['pct_neighbor_simultaneous'].mean()
plt.axvline(x=mean_sim, color='red', linestyle='--', label=f"National Average ({mean_sim:.1f}%)")
plt.title("Neighbor Co-Occurrence: % of Alert Hours Where >= 1 Direct Neighbor is Also in Alert")
plt.xlabel("% of Active Hours with Concurrent Neighbor Alert")
plt.ylabel("")
plt.xlim(50, 100)
plt.legend(loc='lower right')
for i, v in enumerate(df_neighbor_sim['pct_neighbor_simultaneous']):
    plt.text(v + 0.4, i, f"{v:.1f}%", va='center', fontsize=9)
plt.tight_layout()
plt.show()

# Pairwise Correlation Heatmap Across All 24 Regions
id_to_en = df_alerts.drop_duplicates('region_id').set_index('region_id')['region_en'].to_dict()
clean_names = [id_to_en.get(r, r).replace(' Oblast', '').replace(' Autonomous Republic of ', '') for r in region_ids]
df_hourly = pd.DataFrame(matrix.astype(int), columns=clean_names)
corr_matrix = df_hourly.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    cmap='YlOrRd',
    annot=False,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation (Concurrent Active Alerts)'}
)
plt.title("Pairwise Simultaneous Alert Correlation Matrix Across Ukrainian Regions")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()""")

    add_code("""# 3.4 Inter-Alert Quiet Gap Distribution (Task B)
plt.figure(figsize=(12, 5))
gap_hours = df_gaps['gap_duration_min'] / 60.0

sns.histplot(gap_hours[gap_hours <= 72], bins=72, color='navy', kde=True)
plt.title("Distribution of Quiet Gaps Between Alerts (<= 72 Hours)")
plt.xlabel("Quiet Gap Between Alerts (Hours)")
plt.ylabel("Number of Quiet Episodes")
plt.tight_layout()
plt.show()

gap_summary = df_gaps.groupby('region_en')['gap_duration_min'].agg(
    episode_count='count',
    median_quiet_hours=lambda s: s.median() / 60.0,
    p25_quiet_hours=lambda s: np.percentile(s, 25) / 60.0,
    p75_quiet_hours=lambda s: np.percentile(s, 75) / 60.0,
).sort_values(by='median_quiet_hours').reset_index()

print("Median Quiet Hours Between Alerts by Oblast (Frontline vs. Rear Oblasts):")
display(gap_summary)""")

    # Section 4: Chronological Train / Test Split
    add_md("""## 4. Chronological Train / Test Split
**PLAN.md Section 5.6:** *Never shuffle time series data.* We perform a strict point-in-time split (e.g. 80% train, 20% test on recent months) to prevent data leakage.""")

    add_code("""# Split alerts (Task A) and gaps (Task B)
alerts_train, alerts_test = create_time_split(df_alerts, timestamp_col='started_at', train_ratio=0.80)
gaps_train, gaps_test = create_time_split(df_gaps, timestamp_col='gap_start', train_ratio=0.80)

split_date = alerts_test['started_at'].min()
print(f"Train/Test split cutoff timestamp: {split_date}")
print(f"Task A (Alerts): Train = {len(alerts_train):,}, Test = {len(alerts_test):,}")
print(f"Task B (Gaps):   Train = {len(gaps_train):,}, Test = {len(gaps_test):,}")""")

    # Section 5: Baseline Models
    add_md("""## 5. Baseline Models (PLAN.md Section 5.1)
The core insight: When an alert has already lasted $t$ minutes, what is the conditional distribution of remaining time:
$$P(\\text{remaining} \\le \\Delta \\mid \\text{duration} > t)$$
We implement:
1. **Empirical Conditional Quantile Baseline**: Non-parametric conditional quantiles ($P10, P50, P90$) and probabilities.
2. **Kaplan-Meier Survival Baseline**: Survival curve $S(t) = P(D > t)$ with conditional probability:
$$P(\\text{ends within } \\Delta \\mid D > t) = 1 - \\frac{S(t + \\Delta)}{S(t)}$$""")

    add_code("""# 5.1 Fit Empirical Baseline on Training Data
baseline_empirical = EmpiricalConditionalBaseline().fit(alerts_train)

# Fit Kaplan-Meier baseline
km_baseline = KaplanMeierBaseline().fit(alerts_train)

# Compare predictions for Kyiv City when alert has lasted 30 minutes
pred_kyiv_30m = baseline_empirical.predict_remaining('kyiv_city', elapsed_min=30.0)
km_prob_30m_end_in_15m = km_baseline.predict_survival_prob('kyiv_city', elapsed_min=30.0, delta_min=15.0)

print("Empirical Baseline Prediction (Kyiv City, elapsed = 30m):")
print(f"  Predicted remaining median: {pred_kyiv_30m['p50']:.1f} min")
print(f"  Uncertainty range [P10, P90]: [{pred_kyiv_30m['p10']:.1f}, {pred_kyiv_30m['p90']:.1f}] min")
print(f"  Probability ends within 15 min: {pred_kyiv_30m['probabilities']['within_15m']:.1%}")
print(f"  Kaplan-Meier prob ends within 15 min: {km_prob_30m_end_in_15m:.1%}")""")

    add_code("""# 5.2 Plot Kaplan-Meier Survival Curves for Selected Regions
plt.figure(figsize=(10, 5))
for reg_id in ['kyiv_city', 'kharkiv', 'lviv']:
    kmf = km_baseline.kmfs[reg_id]
    plt.plot(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'], label=reg_id, lw=2)

plt.title("Alert Survival Curves S(t) = P(Duration > t)")
plt.xlabel("Elapsed Alert Minutes (t)")
plt.ylabel("Survival Probability S(t)")
plt.xlim(0, 180)
plt.ylim(0, 1)
plt.legend()
plt.show()""")

    # Section 6: Feature Engineering & As-Of Moment Sampling
    add_md("""## 6. Feature Engineering & As-Of Moment Sampling
**PLAN.md Section 5.6:** In production, users query the app at arbitrary moments *during* an ongoing alert.
To train models accurately:
- We sample "as-of" snapshot moments $t \\in [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]$ inside each historical alert episode.
- At snapshot $t$:
  - Feature: $elapsed\\_min = t$
  - Target: $remaining\\_min = \\text{total duration} - t$
  - Classification Targets: $ends\\_within\\_15m, ends\\_within\\_30m, ends\\_within\\_60m$
- We compute point-in-time cyclical time features (hour sine/cosine, weekday, month, weekend).""")

    add_code("""# Sample as-of snapshots for Task A
train_a_samples = sample_as_of_alert_moments(alerts_train)
test_a_samples = sample_as_of_alert_moments(alerts_test)

# Sample as-of snapshots for Task B
train_b_samples = sample_as_of_gap_moments(gaps_train)
test_b_samples = sample_as_of_gap_moments(gaps_test)

print(f"Task A (Alert Duration) Snapshots -> Train: {len(train_a_samples):,}, Test: {len(test_a_samples):,}")
print(f"Task B (Quiet Gaps) Snapshots     -> Train: {len(train_b_samples):,}, Test: {len(test_b_samples):,}")
display(train_a_samples.head())""")

    # Section 7: LightGBM Model Training
    add_md("""## 7. Machine Learning Models: LightGBM Quantile & Probability
We train:
1. **LightGBM Quantile Regressors** with $\\alpha \\in [0.10, 0.50, 0.90]$ to predict remaining duration and intervals.
2. **LightGBM Binary Classifiers** to output calibrated probabilities:
   - Task A: $P(\\text{ends within } 15, 30, 60 \\text{ min})$
   - Task B: $P(\\text{starts within } 1, 3, 6 \\text{ hours})$""")

    add_code("""# Initialize end-to-end pipeline
pipeline = AlertPredictionPipeline()

# Train Task A (Alert Remaining Duration)
print("--> Training Task A Models...")
pipeline.fit_task_a(train_a_samples)

# Train Task B (Quiet Gap Until Next Alert)
print("--> Training Task B Models...")
pipeline.fit_task_b(train_b_samples)

print("All models successfully trained!")""")

    # Section 8: Model Evaluation & Comparison
    add_md("""## 8. Model Evaluation & Benchmark Comparison
Comparing **LightGBM** vs. **Empirical Baseline** on the held-out test split:
- **MAE / Median AE:** point accuracy in minutes
- **Pinball Loss:** quantile calibration
- **P10-P90 Coverage:** fraction of true outcomes inside the predicted interval (target: ~80%)
- **Brier Score & Accuracy:** probability calibration""")

    add_code("""# Evaluate Task A Baseline
baseline_preds = baseline_empirical.predict_batch(test_a_samples, elapsed_col='elapsed_min')
eval_baseline_a = evaluate_quantile_predictions(
    y_true=test_a_samples['target_remaining_min'].values,
    p10=baseline_preds['baseline_p10'].values,
    p50=baseline_preds['baseline_p50'].values,
    p90=baseline_preds['baseline_p90'].values
)

# Evaluate Task A LightGBM
lgb_preds_a = pipeline.task_a_quantile.predict(test_a_samples)
eval_lgb_a = evaluate_quantile_predictions(
    y_true=test_a_samples['target_remaining_min'].values,
    p10=lgb_preds_a['pred_p10'].values,
    p50=lgb_preds_a['pred_p50'].values,
    p90=lgb_preds_a['pred_p90'].values
)

# Benchmark Comparison Table
comparison_df = pd.DataFrame([
    {"Model": "Empirical Baseline", **eval_baseline_a},
    {"Model": "LightGBM Quantile", **eval_lgb_a},
])
display(comparison_df[['Model', 'mae_p50_min', 'median_ae_p50_min', 'coverage_p10_p90', 'mean_pinball_loss']])""")

    add_code("""# Evaluate Probability Classifiers for Task A (Ends within 15/30/60 min)
lgb_probs_a = pipeline.task_a_probs.predict_proba(test_a_samples)
prob_evals = []
for h in [15, 30, 60]:
    metrics = evaluate_classification_probabilities(
        y_true_binary=test_a_samples[f'target_ends_within_{h}m'].values,
        y_prob=lgb_probs_a[f'prob_ends_within_{h}m'].values
    )
    metrics['Horizon'] = f"Ends within {h}m"
    prob_evals.append(metrics)

display(pd.DataFrame(prob_evals)[['Horizon', 'accuracy', 'brier_score', 'roc_auc']])""")

    # Section 9: Interactive Scenario Exploration
    add_md("""## 9. Interactive Scenario Explorer
Simulating live queries that will appear in the Streamlit web app.""")

    add_code("""# 9.1 Scenario: Active Alert in Kyiv running for 35 minutes
pred_active_kyiv = pipeline.predict_active_alert(
    region_id='kyiv_city',
    elapsed_min=35.0,
    timestamp=pd.Timestamp.now(tz='UTC')
)

print("=" * 60)
print("SCENARIO 1: ACTIVE ALERT IN KYIV (Elapsed: 35 minutes)")
print("=" * 60)
print(f"Predicted Remaining Duration (Median P50): {pred_active_kyiv['predicted_remaining_median_min']} min")
print(f"Predicted Uncertainty Range [P10, P90]   : {pred_active_kyiv['predicted_range_p10_p90_min']} min")
print("Probabilities:")
for k, v in pred_active_kyiv['probabilities'].items():
    print(f"  • {k}: {v:.1%}")""")

    add_code("""# 9.2 Scenario: Quiet Region in Lviv with no alert for 4 hours
pred_quiet_lviv = pipeline.predict_quiet_region(
    region_id='lviv',
    elapsed_quiet_min=240.0,
    prev_alert_duration_min=40.0,
    timestamp=pd.Timestamp.now(tz='UTC')
)

print("=" * 60)
print("SCENARIO 2: QUIET PERIOD IN LVIV (Quiet for: 4 hours)")
print("=" * 60)
print(f"Predicted Gap to Next Alert (Median P50): {pred_quiet_lviv['predicted_next_alert_gap_median_min']} min ({pred_quiet_lviv['predicted_next_alert_gap_median_min']/60:.1f} hrs)")
print(f"Predicted Uncertainty Range [P10, P90]  : {pred_quiet_lviv['predicted_range_p10_p90_min']} min")
print("Probabilities:")
for k, v in pred_quiet_lviv['probabilities'].items():
    print(f"  • {k}: {v:.1%}")""")

    add_code("""# 9.3 Visualizing Predicted Remaining Duration vs. Elapsed Time for Multiple Oblasts
elapsed_grid = np.linspace(0, 120, 25)
regions_to_plot = ['kyiv_city', 'kharkiv', 'odesa', 'lviv']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, reg in enumerate(regions_to_plot):
    p50_list, p10_list, p90_list = [], [], []
    for t in elapsed_grid:
        res = pipeline.predict_active_alert(reg, elapsed_min=t)
        p50_list.append(res['predicted_remaining_median_min'])
        p10_list.append(res['predicted_range_p10_p90_min'][0])
        p90_list.append(res['predicted_range_p10_p90_min'][1])
        
    ax = axes[idx]
    ax.plot(elapsed_grid, p50_list, 'b-', label='Predicted Median Remaining (P50)', lw=2)
    ax.fill_between(elapsed_grid, p10_list, p90_list, color='blue', alpha=0.2, label='P10 - P90 Uncertainty')
    ax.set_title(f"Region: {reg}")
    ax.set_xlabel("Elapsed Time Since Alert Start (min)")
    ax.set_ylabel("Predicted Remaining Time (min)")
    ax.legend()
    
plt.suptitle("Remaining Alert Duration vs. Elapsed Time (Conditional Curve)", fontsize=14)
plt.tight_layout()
plt.show()""")

    # Section 10: Model Serialization
    add_md("""## 10. Model Serialization & Export
Saving the trained pipeline artifact to `models/` directory for direct integration into the main project app.""")

    add_code("""# Save the trained pipeline
saved_path = pipeline.save(MODELS_DIR / "alertpred_pipeline.joblib")
print(f"Pipeline artifact saved to: {saved_path}")

# Verify deserialization
loaded_pipeline = AlertPredictionPipeline.load(saved_path)
test_pred = loaded_pipeline.predict_active_alert('kyiv_city', elapsed_min=15.0)
print("Reloaded pipeline test prediction successful!")
print(test_pred)""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_file = project_root / "notebooks" / "01_model_training_pipeline.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")

if __name__ == "__main__":
    create_notebook()
