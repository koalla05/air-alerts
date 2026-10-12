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

    add_md("""# Do Explicit Time-of-Day / Day-of-Week Buckets Improve Predictions?
> **Companion to notebooks 01-03.** The currently-served model already has
> *cyclical* time encodings (`hour_sin/cos`, `dow_sin/cos`) - smooth,
> continuous representations of time of day and day of week. This notebook
> tests a different hypothesis: does giving LightGBM an **explicit
> categorical** time-of-day bucket (night / morning / afternoon / evening,
> on Kyiv local time - not UTC, which the cyclical features use) and an
> explicit day-of-week category help it learn the dynamics better than the
> cyclical encoding alone?
>
> The intuition: a tree model *can* approximate a block boundary from
> `hour_sin`/`hour_cos` alone, but only by combining several splits across
> both dimensions at once. An explicit "is this a night alert" column makes
> that a single split.
>
> Same protocol as notebook 03: same model, same recency-weighting, same
> held-out recent test window, with and without the new features, starting
> from **today's actual production feature set** (region, elapsed time,
> cyclical time, and the neighbor-context features notebook 03 already
> justified adopting) - not an older baseline.
>
> *Disclaimer: statistical experiment for research and analysis. Decides
> nothing about the app by itself - see the verdict cell at the end.*""")

    add_md("## 1. Setup & Imports")
    add_code("""%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from alertpred.config import OBLAST_NEIGHBORS, RECENCY_HALF_LIFE_DAYS, TASK_A_HORIZONS_MIN
from alertpred.data.build_dataset import build_master_alerts_dataset
from alertpred.data.cleaning import create_time_split
from alertpred.features.builder import (
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
    compute_recency_weight,
    compute_neighbor_active_features,
    compute_time_bucket_features,
)
from alertpred.models.quantile_lgbm import LightGBMQuantileModel, LightGBMProbabilityModel
from alertpred.evaluation.metrics import evaluate_quantile_predictions, evaluate_classification_probabilities

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print(f"Project root: {project_root}")""")

    add_md("""## 2. Load the Master Dataset
Same pipeline as `train.py` / notebook 01.""")

    add_code("""df_alerts, df_gaps = build_master_alerts_dataset(use_api_backfill=True)
print(f"Alert episodes: {len(df_alerts):,}  |  Gap episodes: {len(df_gaps):,}")
print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")""")

    add_md("""## 3. Chronological Split, Recency Weights, As-Of Samples
Identical to `train.py`'s production methodology.""")

    add_code("""TRAIN_RATIO = 0.80

alerts_train, alerts_test = create_time_split(df_alerts, timestamp_col='started_at', train_ratio=TRAIN_RATIO)
gaps_train, gaps_test = create_time_split(df_gaps, timestamp_col='gap_start', train_ratio=TRAIN_RATIO)

reference_ts = df_alerts['started_at'].max()
alerts_train['recency_weight'] = compute_recency_weight(alerts_train['started_at'], reference_ts, RECENCY_HALF_LIFE_DAYS)
gaps_train['recency_weight'] = compute_recency_weight(gaps_train['gap_start'], reference_ts, RECENCY_HALF_LIFE_DAYS)

train_a = sample_as_of_alert_moments(alerts_train)
test_a = sample_as_of_alert_moments(alerts_test)
train_b = sample_as_of_gap_moments(gaps_train)
test_b = sample_as_of_gap_moments(gaps_test)

train_a = train_a.merge(alerts_train[['region_id', 'started_at', 'recency_weight']].rename(columns={'started_at': 'start_ts'}), on=['region_id', 'start_ts'], how='left')
train_b = train_b.merge(gaps_train[['region_id', 'gap_start', 'recency_weight']].rename(columns={'gap_start': 'gap_start_ts'}), on=['region_id', 'gap_start_ts'], how='left')

print(f"Task A -> train: {len(train_a):,}, test: {len(test_a):,}")
print(f"Task B -> train: {len(train_b):,}, test: {len(test_b):,}")""")

    add_md("""## 4. Feature Setup: Today's Production Baseline + the New Hypothesis
`neighbors_active_*`/`national_active_count` (notebook 03) are already part
of the production feature set, so they're computed here as part of the
*baseline*, not the thing being tested. `time_of_day`/`day_of_week`
(Kyiv-local, see `features.builder.compute_time_bucket_features`) are the
new hypothesis.""")

    add_code("""train_a = compute_neighbor_active_features(train_a, df_alerts, OBLAST_NEIGHBORS)
test_a = compute_neighbor_active_features(test_a, df_alerts, OBLAST_NEIGHBORS)
train_b = compute_neighbor_active_features(train_b, df_alerts, OBLAST_NEIGHBORS)
test_b = compute_neighbor_active_features(test_b, df_alerts, OBLAST_NEIGHBORS)

for df in (train_a, test_a, train_b, test_b):
    bucket_feats = compute_time_bucket_features(df['as_of_ts'])
    df['time_of_day'] = bucket_feats['time_of_day']
    df['day_of_week'] = bucket_feats['day_of_week']

print("Time-of-day distribution (test set, Task A):")
display(test_a['time_of_day'].value_counts())
print()
print("Day-of-week distribution (test set, Task A):")
display(test_a['day_of_week'].value_counts())""")

    add_md("""## 5. Task A: Production Baseline vs. + Time-of-Day/Day-of-Week
`time_of_day` and `day_of_week` are added to `categorical_cols` (alongside
`region_id`) so LightGBM treats them as categories with optimal splits,
the same way `region_id` already is - not as arbitrary ordinal numbers.""")

    add_code("""PRODUCTION_FEATURES_A = [
    'region_id', 'elapsed_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month',
    'neighbors_active_count', 'neighbors_active_frac', 'national_active_count',
]
TIME_BUCKET_FEATURES = ['time_of_day', 'day_of_week']

results = []

for label, feature_cols, cat_cols in [
    ('Production (cyclical only)', PRODUCTION_FEATURES_A, ['region_id']),
    ('+ Time-of-Day/Day-of-Week', PRODUCTION_FEATURES_A + TIME_BUCKET_FEATURES, ['region_id'] + TIME_BUCKET_FEATURES),
]:
    quant = LightGBMQuantileModel(feature_cols=feature_cols, categorical_cols=cat_cols)
    quant.fit(train_a, target_col='target_remaining_min', weight_col='recency_weight')
    pred = quant.predict(test_a)
    m = evaluate_quantile_predictions(test_a['target_remaining_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
    results.append({'task': 'A', 'variant': label, **m})

    if label.startswith('+'):
        importances_a = pd.Series(quant.models[0.5].feature_importances_, index=feature_cols).sort_values(ascending=False)

df_res_a = pd.DataFrame(results).set_index(['task', 'variant'])
display(df_res_a.round(2))""")

    add_code("""print("LightGBM P50 feature importances, Task A, with time-of-day/day-of-week included:")
display(importances_a)

plt.figure(figsize=(10, 5))
importances_a.sort_values().plot(kind='barh', color='#0ea5e9')
plt.title("Task A Feature Importance (P50 model, + time-of-day/day-of-week)")
plt.tight_layout()
plt.show()""")

    add_md("""## 6. Task B: Production Baseline vs. + Time-of-Day/Day-of-Week""")

    add_code("""PRODUCTION_FEATURES_B = [
    'region_id', 'elapsed_quiet_min', 'prev_alert_duration_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month',
    'neighbors_active_count', 'neighbors_active_frac', 'national_active_count',
]

for label, feature_cols, cat_cols in [
    ('Production (cyclical only)', PRODUCTION_FEATURES_B, ['region_id']),
    ('+ Time-of-Day/Day-of-Week', PRODUCTION_FEATURES_B + TIME_BUCKET_FEATURES, ['region_id'] + TIME_BUCKET_FEATURES),
]:
    quant = LightGBMQuantileModel(feature_cols=feature_cols, categorical_cols=cat_cols)
    quant.fit(train_b, target_col='target_remaining_gap_min', weight_col='recency_weight')
    pred = quant.predict(test_b)
    m = evaluate_quantile_predictions(test_b['target_remaining_gap_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
    results.append({'task': 'B', 'variant': label, **m})

    if label.startswith('+'):
        importances_b = pd.Series(quant.models[0.5].feature_importances_, index=feature_cols).sort_values(ascending=False)

df_res_b = pd.DataFrame([r for r in results if r['task'] == 'B']).set_index(['task', 'variant'])
display(df_res_b.round(2))
print("\\nFeature importances, Task B, with time-of-day/day-of-week included:")
display(importances_b)""")

    add_md("""## 7. Probability Classifiers: Does Calibration Improve?
Unlike notebook 03 (which found no benefit from neighbor features here and
so left the classifiers alone), this hypothesis is checked against the
classifiers too since there's no prior result ruling it out.""")

    add_code("""prob_results = []
for label, feature_cols, cat_cols in [
    ('Production (cyclical only)', PRODUCTION_FEATURES_A, ['region_id']),
    ('+ Time-of-Day/Day-of-Week', PRODUCTION_FEATURES_A + TIME_BUCKET_FEATURES, ['region_id'] + TIME_BUCKET_FEATURES),
]:
    probs = LightGBMProbabilityModel(target_horizons=TASK_A_HORIZONS_MIN, horizon_prefix='ends_within_', horizon_suffix='m', feature_cols=feature_cols, categorical_cols=cat_cols)
    probs.fit(train_a, weight_col='recency_weight')
    pred_proba = probs.predict_proba(test_a)
    for h in TASK_A_HORIZONS_MIN:
        target_col = f'target_ends_within_{h}m'
        prob_col = f'prob_ends_within_{h}m'
        m = evaluate_classification_probabilities(test_a[target_col].values, pred_proba[prob_col].values)
        prob_results.append({'variant': label, 'horizon_min': h, **m})

df_prob = pd.DataFrame(prob_results).set_index(['horizon_min', 'variant'])
display(df_prob.round(4))""")

    add_md("""## 8. Verdict
Same scoring as notebook 03: compares MAE and pinball loss (lower is
better) across every (task, variant) pair. A recommendation to weigh, not
a decision already made - this notebook changes nothing in `train.py` or
the app by itself.""")

    add_code("""summary = pd.DataFrame(results)
pivot = summary.pivot(index='task', columns='variant', values=['mae_p50_min', 'mean_pinball_loss', 'coverage_p10_p90'])
display(pivot.round(2))

for task in ['A', 'B']:
    base = summary[(summary['task'] == task) & (summary['variant'].str.startswith('Production'))].iloc[0]
    new = summary[(summary['task'] == task) & (summary['variant'].str.startswith('+'))].iloc[0]
    mae_delta_pct = 100 * (new['mae_p50_min'] - base['mae_p50_min']) / base['mae_p50_min']
    pinball_delta_pct = 100 * (new['mean_pinball_loss'] - base['mean_pinball_loss']) / base['mean_pinball_loss']
    print(f"Task {task}: + time-of-day/day-of-week {'improves' if mae_delta_pct < 0 else 'worsens'} MAE by {abs(mae_delta_pct):.1f}% "
          f"({'improves' if pinball_delta_pct < 0 else 'worsens'} pinball loss by {abs(pinball_delta_pct):.1f}%)")

print(f"\\nTask A new-feature importance rank (of {len(PRODUCTION_FEATURES_A) + len(TIME_BUCKET_FEATURES)} features): "
      f"{[f for f in importances_a.index if f in TIME_BUCKET_FEATURES]}")
print("(top of the list = most used by the model; compare position against elapsed_min/region_id/hour_sin)")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    out_file = project_root / "notebooks" / "04_time_of_day_features.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")

if __name__ == "__main__":
    create_notebook()
