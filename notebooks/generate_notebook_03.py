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

    add_md("""# Do Neighboring Regions' Alert Status Improve Predictions?
> **Companion to notebooks 01-02.** PLAN.md (Section 4.1) calls spatial
> context - whether neighboring oblasts are also under alert - "likely the
> most valuable feature group," but it was deferred in the first pass. The
> currently-served model (`AlertPredictionPipeline`, trained in `train.py`)
> only uses `[region_id, elapsed_min, hour_sin/cos, dow_sin/cos, is_weekend,
> month]` - each region is predicted independently, with no awareness of
> what's happening next door.
>
> This notebook tests the hypothesis directly: train the **same model**,
> **same recency-weighting**, **same train/test split** as `train.py`, with
> and without point-in-time neighbor features, and compare on held-out data.
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

from alertpred.config import OBLAST_NEIGHBORS, RECENCY_HALF_LIFE_DAYS
from alertpred.data.build_dataset import build_master_alerts_dataset
from alertpred.data.cleaning import create_time_split
from alertpred.features.builder import (
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
    compute_recency_weight,
    compute_neighbor_active_features,
)
from alertpred.models.quantile_lgbm import LightGBMQuantileModel, LightGBMProbabilityModel
from alertpred.evaluation.metrics import evaluate_quantile_predictions, evaluate_classification_probabilities

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print(f"Project root: {project_root}")""")

    add_md("""## 2. Load the Master Dataset
Same pipeline as `train.py` / notebook 01: bias-corrected oblast rollup +
alerts.in.ua backfill + volunteer gap-fill.""")

    add_code("""df_alerts, df_gaps = build_master_alerts_dataset(use_api_backfill=True)
print(f"Alert episodes: {len(df_alerts):,}  |  Gap episodes: {len(df_gaps):,}")
print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")""")

    add_md("""## 3. Chronological Split, Recency Weights, As-Of Samples
Identical to `train.py`'s production methodology (same 80/20 chronological
split, same recency half-life) so this experiment isolates the effect of
adding neighbor features - not a different training recipe.""")

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

# Recency weight doesn't survive as-of sampling's row-by-row construction - join it back
train_a = train_a.merge(alerts_train[['region_id', 'started_at', 'recency_weight']].rename(columns={'started_at': 'start_ts'}), on=['region_id', 'start_ts'], how='left')
train_b = train_b.merge(gaps_train[['region_id', 'gap_start', 'recency_weight']].rename(columns={'gap_start': 'gap_start_ts'}), on=['region_id', 'gap_start_ts'], how='left')

print(f"Task A -> train: {len(train_a):,}, test: {len(test_a):,}")
print(f"Task B -> train: {len(train_b):,}, test: {len(test_b):,}")""")

    add_md("""## 4. Compute Point-in-Time Neighbor Features
`compute_neighbor_active_features` answers, for each historical (region,
as-of-moment) snapshot: how many of this region's direct neighbors (per
`config.OBLAST_NEIGHBORS`) - and how many regions nationally - had an
*active* episode at that exact moment. It's a backward as-of join against
each region's own sorted episode table, so only episodes that had already
started by that moment can match - no future information leaks in.

Computed separately for Task A (against `df_alerts`, "is my neighbor
currently mid-alert") and Task B (also against `df_alerts` - the question
during a *quiet* period is still "is a neighbor currently alerting," since
that's a plausible precursor signal for this region's own next alert).""")

    add_code("""train_a = compute_neighbor_active_features(train_a, df_alerts, OBLAST_NEIGHBORS)
test_a = compute_neighbor_active_features(test_a, df_alerts, OBLAST_NEIGHBORS)
train_b = compute_neighbor_active_features(train_b, df_alerts, OBLAST_NEIGHBORS)
test_b = compute_neighbor_active_features(test_b, df_alerts, OBLAST_NEIGHBORS)

print("Task A neighbor feature summary (test set):")
display(test_a[['neighbors_active_count', 'neighbors_active_frac', 'national_active_count']].describe().round(2))""")

    add_md("""## 5. Task A: With vs. Without Neighbor Features
Two identical `LightGBMQuantileModel` + `LightGBMProbabilityModel` pairs,
both recency-weighted, differing only in `feature_cols`.""")

    add_code("""BASE_FEATURES_A = ['region_id', 'elapsed_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month']
NEIGHBOR_FEATURES = ['neighbors_active_count', 'neighbors_active_frac', 'national_active_count']

results = []

for label, feature_cols in [('Without Neighbors (current)', BASE_FEATURES_A), ('With Neighbors', BASE_FEATURES_A + NEIGHBOR_FEATURES)]:
    quant = LightGBMQuantileModel(feature_cols=feature_cols)
    quant.fit(train_a, target_col='target_remaining_min', weight_col='recency_weight')
    pred = quant.predict(test_a)
    m = evaluate_quantile_predictions(test_a['target_remaining_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
    results.append({'task': 'A', 'variant': label, **m})

    if label == 'With Neighbors':
        importances = pd.Series(quant.models[0.5].feature_importances_, index=feature_cols).sort_values(ascending=False)

df_res_a = pd.DataFrame(results).set_index(['task', 'variant'])
display(df_res_a.round(2))""")

    add_code("""print("LightGBM P50 feature importances, Task A, with neighbor features included:")
display(importances)

plt.figure(figsize=(10, 4))
importances.sort_values().plot(kind='barh', color='#0ea5e9')
plt.title("Task A Feature Importance (P50 model, with neighbor features)")
plt.tight_layout()
plt.show()""")

    add_md("""## 6. Task B: With vs. Without Neighbor Features
Same comparison for time-to-next-alert.""")

    add_code("""BASE_FEATURES_B = ['region_id', 'elapsed_quiet_min', 'prev_alert_duration_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month']

for label, feature_cols in [('Without Neighbors (current)', BASE_FEATURES_B), ('With Neighbors', BASE_FEATURES_B + NEIGHBOR_FEATURES)]:
    quant = LightGBMQuantileModel(feature_cols=feature_cols)
    quant.fit(train_b, target_col='target_remaining_gap_min', weight_col='recency_weight')
    pred = quant.predict(test_b)
    m = evaluate_quantile_predictions(test_b['target_remaining_gap_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
    results.append({'task': 'B', 'variant': label, **m})

    if label == 'With Neighbors':
        importances_b = pd.Series(quant.models[0.5].feature_importances_, index=feature_cols).sort_values(ascending=False)

df_res_b = pd.DataFrame([r for r in results if r['task'] == 'B']).set_index(['task', 'variant'])
display(df_res_b.round(2))
print("\\nFeature importances, Task B, with neighbor features included:")
display(importances_b)""")

    add_md("""## 7. Probability Models: Does "Ends/Starts Within Xh" Calibration Improve?
The quantile regressors answer "how long," but the app also shows threshold
probabilities ("ends within 30 min: 41%"). Check those separately since a
model can win on MAE but not on calibrated probabilities, or vice versa.""")

    add_code("""from alertpred.config import TASK_A_HORIZONS_MIN

prob_results = []
for label, feature_cols in [('Without Neighbors (current)', BASE_FEATURES_A), ('With Neighbors', BASE_FEATURES_A + NEIGHBOR_FEATURES)]:
    probs = LightGBMProbabilityModel(target_horizons=TASK_A_HORIZONS_MIN, horizon_prefix='ends_within_', horizon_suffix='m', feature_cols=feature_cols)
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
Compares MAE and pinball loss (lower is better for both) across every
(task, variant) pair, plus where neighbor features ranked in importance.
A real recommendation, not a decision already made - the next step if this
looks worthwhile is wiring `compute_neighbor_active_features` into
`train.py` the same way recency-weighting was wired in.""")

    add_code("""summary = pd.DataFrame(results)
pivot = summary.pivot(index='task', columns='variant', values=['mae_p50_min', 'mean_pinball_loss', 'coverage_p10_p90'])
display(pivot.round(2))

for task in ['A', 'B']:
    base = summary[(summary['task'] == task) & (summary['variant'].str.contains('Without'))].iloc[0]
    neigh = summary[(summary['task'] == task) & (summary['variant'].str.contains('With Neighbors'))].iloc[0]
    mae_delta_pct = 100 * (neigh['mae_p50_min'] - base['mae_p50_min']) / base['mae_p50_min']
    pinball_delta_pct = 100 * (neigh['mean_pinball_loss'] - base['mean_pinball_loss']) / base['mean_pinball_loss']
    direction = "improves" if mae_delta_pct < 0 else "worsens"
    print(f"Task {task}: adding neighbor features {direction} MAE by {abs(mae_delta_pct):.1f}% "
          f"({'improves' if pinball_delta_pct < 0 else 'worsens'} pinball loss by {abs(pinball_delta_pct):.1f}%)")

print(f"\\nTask A neighbor-feature importance rank (of {len(BASE_FEATURES_A) + len(NEIGHBOR_FEATURES)} features): "
      f"{[f for f in importances.index if f in NEIGHBOR_FEATURES]}")
print("(top of the list = most used by the model; compare position against elapsed_min/region_id)")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    out_file = project_root / "notebooks" / "03_neighbor_features.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")

if __name__ == "__main__":
    create_notebook()
