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
    add_md("""# Regime-Shift Handling: Hard Cutover vs. Recency-Weighted Training
> **Companion to notebook 01.** Ukraine's official alert reporting shifted from
> oblast-wide blanket alerts to localized raion/hromada alerts over
> Aug-Nov 2025 (see notebook 01, section 2.1b). That's not just a data gap -
> it's a change in what "an alert episode" structurally looks like, which
> changes the duration/gap distributions a model should learn.
>
> This notebook compares two ways to handle it, training the **same model
> classes** on **the same data pool**, evaluated on **the same held-out
> recent test window**, so the only thing that differs is the data strategy:
>
> 1. **Hard cutover** - train only on data from after the regime boundary.
> 2. **Recency-weighted** - train on the full history, down-weighting older
>    (pre-shift) episodes exponentially so the fit still leans toward
>    current behavior.
>
> *Disclaimer: statistical experiment for research and analysis. Decides
> nothing about the app by itself - see the verdict cell at the end.*""")

    # 1. Setup
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

from alertpred.config import MODELS_DIR
from alertpred.data.build_dataset import build_master_alerts_dataset
from alertpred.features.builder import sample_as_of_alert_moments, sample_as_of_gap_moments
from alertpred.models.baseline import EmpiricalConditionalBaseline
from alertpred.models.quantile_lgbm import LightGBMQuantileModel
from alertpred.evaluation.metrics import evaluate_quantile_predictions

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print(f"Project root: {project_root}")""")

    # 2. Load master dataset
    add_md("""## 2. Load the Master Dataset
Reuses the exact pipeline walked through step-by-step in notebook 01
(`alertpred.data.build_dataset.build_master_alerts_dataset`): raw CSV ->
bias-corrected oblast rollup -> alerts.in.ua backfill (if a token is
configured) -> merge. See notebook 01 for the detailed walkthrough and
validation of each step; this notebook starts from its result.""")

    add_code("""df_alerts, df_gaps = build_master_alerts_dataset(use_api_backfill=True)
print(f"Alert episodes: {len(df_alerts):,}  |  Gap episodes: {len(df_gaps):,}")
print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")
display(df_alerts['level'].value_counts().rename('episodes'))""")

    # 3. Identify regime boundary
    add_md("""## 3. Identify the Regime Boundary
Compute the monthly share of episodes that required a rollup (i.e. were
*not* native oblast-wide alerts) and pick the first month where that share
crosses 50% and stays there - that's where "oblast-wide" stopped being the
typical way an alert looked.""")

    add_code("""monthly = df_alerts.set_index('started_at').resample('ME')['level'].apply(
    lambda s: (s == 'oblast_rollup').mean() * 100
)

plt.figure(figsize=(13, 4))
monthly.plot(kind='bar', color='#7c3aed', edgecolor='black', alpha=0.85)
plt.axhline(50, color='red', linestyle='--', label='50% rollup share')
plt.title("Monthly Share of Episodes Requiring Rollup (raion/hromada -> oblast)")
plt.ylabel("% of episodes")
plt.xticks(rotation=90, fontsize=7)
plt.legend()
plt.tight_layout()
plt.show()

# First month where rollup share crosses 50% and never drops back below it
above_50 = monthly >= 50.0
regime_b_start_ts = above_50[above_50[::-1].cummin()[::-1]].index.min()
REGIME_CUTOVER = regime_b_start_ts.tz_convert('UTC').replace(day=1, hour=0, minute=0, second=0, microsecond=0)
print(f"Regime boundary (start of raion-dominant era): {REGIME_CUTOVER.strftime('%Y-%m-%d')}")""")

    # 4. Define training strategies + shared test window
    add_md("""## 4. Two Training Strategies, One Shared Test Window
Both strategies draw from the same trainable pool (everything before the
test window - no strategy gets to see data the other couldn't). The test
window is the most recent slice, held out from both: that's what the app
actually needs to be accurate on *right now*, regardless of how a model
was trained.

**Recency weight** (strategy 2): `0.5 ** (age_days / HALF_LIFE_DAYS)` - an
episode HALF_LIFE_DAYS old counts half as much as a fresh one. 60 days is
a starting point (roughly "the current regime's own age" at the time of
writing) - re-tune once more regime-B history accumulates.""")

    add_code("""TEST_WINDOW_DAYS = 21
HALF_LIFE_DAYS = 60

test_cutoff = df_alerts['started_at'].max() - pd.Timedelta(days=TEST_WINDOW_DAYS)
df_test_raw = df_alerts[df_alerts['started_at'] >= test_cutoff].copy()
df_trainable = df_alerts[df_alerts['started_at'] < test_cutoff].copy()

# Strategy 1: hard cutover
df_train_cutover = df_trainable[df_trainable['started_at'] >= REGIME_CUTOVER].copy()

# Strategy 2: full history, recency-weighted
df_train_weighted = df_trainable.copy()
age_days = (test_cutoff - df_train_weighted['started_at']).dt.total_seconds() / 86400.0
df_train_weighted['recency_weight'] = 0.5 ** (age_days / HALF_LIFE_DAYS)

print(f"Test window        : {test_cutoff.strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')} ({len(df_test_raw):,} episodes)")
print(f"Cutover training    : {len(df_train_cutover):,} episodes (from {REGIME_CUTOVER.strftime('%Y-%m-%d')})")
print(f"Weighted training    : {len(df_train_weighted):,} episodes (full history, weight range {df_train_weighted['recency_weight'].min():.4f}-{df_train_weighted['recency_weight'].max():.3f})")""")

    # 5. Build as-of samples for Task A
    add_md("""## 5. Task A As-Of Samples
Same point-in-time sampling as notebook 01 (`sample_as_of_alert_moments`).
The recency weight is attached back onto the weighted-strategy samples by
joining on `(region_id, start_ts)`, since the sampler doesn't carry
arbitrary extra columns through on its own.""")

    add_code("""test_a = sample_as_of_alert_moments(df_test_raw)
train_a_cutover = sample_as_of_alert_moments(df_train_cutover)

train_a_weighted = sample_as_of_alert_moments(df_train_weighted)
weight_lookup = df_train_weighted[['region_id', 'started_at', 'recency_weight']].rename(columns={'started_at': 'start_ts'})
train_a_weighted = train_a_weighted.merge(weight_lookup, on=['region_id', 'start_ts'], how='left')

print(f"Test samples: {len(test_a):,}")
print(f"Cutover train samples: {len(train_a_cutover):,}")
print(f"Weighted train samples: {len(train_a_weighted):,} (weight nulls: {train_a_weighted['recency_weight'].isna().sum()})")""")

    # 6. Baseline empirical comparison
    add_md("""## 6. Baseline Empirical Comparison (Task A)
The empirical conditional baseline (PLAN.md 5.1) is cheap enough to be a
fair, variance-free comparison of the two *data* strategies before
introducing a more complex model.""")

    add_code("""results = []

baseline_cutover = EmpiricalConditionalBaseline().fit(df_train_cutover)
pred_cutover = baseline_cutover.predict_batch(test_a, elapsed_col='elapsed_min')
m = evaluate_quantile_predictions(
    test_a['target_remaining_min'].values,
    pred_cutover['baseline_p10'].values, pred_cutover['baseline_p50'].values, pred_cutover['baseline_p90'].values,
)
results.append({'task': 'A', 'model': 'Empirical Baseline', 'strategy': 'Hard Cutover', **m})

baseline_weighted = EmpiricalConditionalBaseline().fit(df_train_weighted, weight_col='recency_weight')
pred_weighted = baseline_weighted.predict_batch(test_a, elapsed_col='elapsed_min')
m = evaluate_quantile_predictions(
    test_a['target_remaining_min'].values,
    pred_weighted['baseline_p10'].values, pred_weighted['baseline_p50'].values, pred_weighted['baseline_p90'].values,
)
results.append({'task': 'A', 'model': 'Empirical Baseline', 'strategy': 'Recency-Weighted', **m})

display(pd.DataFrame(results).set_index(['task', 'model', 'strategy']).round(2))""")

    # 7. LightGBM comparison
    add_md("""## 7. LightGBM Quantile Comparison (Task A)
Same two data strategies, now with a model that can actually use
`sample_weight` (vs. the baseline's weighted-percentile trick) and
region/time features beyond just elapsed time.""")

    add_code("""lgbm_feature_cols = ['region_id', 'elapsed_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month']

lgbm_cutover = LightGBMQuantileModel(feature_cols=lgbm_feature_cols)
lgbm_cutover.fit(train_a_cutover, target_col='target_remaining_min')
pred = lgbm_cutover.predict(test_a)
m = evaluate_quantile_predictions(test_a['target_remaining_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
results.append({'task': 'A', 'model': 'LightGBM Quantile', 'strategy': 'Hard Cutover', **m})

lgbm_weighted = LightGBMQuantileModel(feature_cols=lgbm_feature_cols)
lgbm_weighted.fit(train_a_weighted, target_col='target_remaining_min', weight_col='recency_weight')
pred = lgbm_weighted.predict(test_a)
m = evaluate_quantile_predictions(test_a['target_remaining_min'].values, pred['pred_p10'].values, pred['pred_p50'].values, pred['pred_p90'].values)
results.append({'task': 'A', 'model': 'LightGBM Quantile', 'strategy': 'Recency-Weighted', **m})

display(pd.DataFrame(results).set_index(['task', 'model', 'strategy']).round(2))""")

    add_code("""# Visual comparison: MAE and pinball loss, all four Task A variants
df_res_a = pd.DataFrame([r for r in results if r['task'] == 'A'])
df_res_a['label'] = df_res_a['model'] + ' / ' + df_res_a['strategy']

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, metric, title in zip(
    axes,
    ['mae_p50_min', 'mean_pinball_loss', 'coverage_p10_p90'],
    ['MAE of Median (min, lower better)', 'Mean Pinball Loss (lower better)', 'P10-P90 Coverage (ideal ~80%)'],
):
    sns.barplot(data=df_res_a, y='label', x=metric, hue='strategy', ax=ax, dodge=False, palette=['#f59e0b', '#0ea5e9'])
    ax.set_title(title)
    ax.set_ylabel('')
    if metric == 'coverage_p10_p90':
        ax.axvline(0.80, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()""")

    # 8. Task B (gaps) - baseline comparison
    add_md("""## 8. Task B (Time to Next Alert): Baseline Comparison
Same two-strategy comparison for the quiet-gap task. The LightGBM version
would follow an identical pattern to section 7 (target column
`target_remaining_gap_min`, `elapsed_quiet_min` in place of `elapsed_min`)
- omitted here for brevity since section 7 already demonstrates it.""")

    add_code("""gaps_test_raw = df_gaps[df_gaps['gap_start'] >= test_cutoff].copy()
gaps_trainable = df_gaps[df_gaps['gap_start'] < test_cutoff].copy()
gaps_train_cutover = gaps_trainable[gaps_trainable['gap_start'] >= REGIME_CUTOVER].copy()

gaps_train_weighted = gaps_trainable.copy()
gap_age_days = (test_cutoff - gaps_train_weighted['gap_start']).dt.total_seconds() / 86400.0
gaps_train_weighted['recency_weight'] = 0.5 ** (gap_age_days / HALF_LIFE_DAYS)

test_b = sample_as_of_gap_moments(gaps_test_raw)
train_b_cutover = sample_as_of_gap_moments(gaps_train_cutover)

train_b_weighted = sample_as_of_gap_moments(gaps_train_weighted)
weight_lookup_b = gaps_train_weighted[['region_id', 'gap_start', 'recency_weight']].rename(columns={'gap_start': 'gap_start_ts'})
train_b_weighted = train_b_weighted.merge(weight_lookup_b, on=['region_id', 'gap_start_ts'], how='left')

print(f"Test samples: {len(test_b):,}  |  Cutover train: {len(train_b_cutover):,}  |  Weighted train: {len(train_b_weighted):,}")""")

    add_code("""baseline_b_cutover = EmpiricalConditionalBaseline().fit(gaps_train_cutover, duration_col='gap_duration_min')
pred_b = baseline_b_cutover.predict_batch(test_b, elapsed_col='elapsed_quiet_min')
m = evaluate_quantile_predictions(test_b['target_remaining_gap_min'].values, pred_b['baseline_p10'].values, pred_b['baseline_p50'].values, pred_b['baseline_p90'].values)
results.append({'task': 'B', 'model': 'Empirical Baseline', 'strategy': 'Hard Cutover', **m})

baseline_b_weighted = EmpiricalConditionalBaseline().fit(gaps_train_weighted, duration_col='gap_duration_min', weight_col='recency_weight')
pred_b = baseline_b_weighted.predict_batch(test_b, elapsed_col='elapsed_quiet_min')
m = evaluate_quantile_predictions(test_b['target_remaining_gap_min'].values, pred_b['baseline_p10'].values, pred_b['baseline_p50'].values, pred_b['baseline_p90'].values)
results.append({'task': 'B', 'model': 'Empirical Baseline', 'strategy': 'Recency-Weighted', **m})

display(pd.DataFrame([r for r in results if r['task'] == 'B']).set_index(['task', 'model', 'strategy']).round(2))""")

    # 9. Verdict
    add_md("""## 9. Verdict
Data-driven summary across every (task, model) pair of which strategy won
on MAE and on pinball loss - a recommendation to weigh, not a decision
already made. This notebook changes nothing in `train.py` or the app by
itself; that's a deliberate next step once you've looked at the numbers
above and agree with (or override) the summary below.""")

    add_code("""df_all = pd.DataFrame(results)
wins = []
for (task, model), grp in df_all.groupby(['task', 'model']):
    mae_winner = grp.loc[grp['mae_p50_min'].idxmin(), 'strategy']
    pinball_winner = grp.loc[grp['mean_pinball_loss'].idxmin(), 'strategy']
    wins.append({'task': task, 'model': model, 'lower_mae': mae_winner, 'lower_pinball_loss': pinball_winner})

df_wins = pd.DataFrame(wins)
display(df_wins)

from collections import Counter
tally = Counter(df_wins['lower_mae']) + Counter(df_wins['lower_pinball_loss'])
print("\\nWin tally across all (task, model, metric) comparisons:")
for strategy, count in tally.most_common():
    print(f"  {strategy}: {count}")
print("\\nThis is a starting signal, not a final call - re-run this notebook as more")
print("regime-B history accumulates (via the app's own poller) and the comparison")
print("will become more statistically reliable.")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    out_file = project_root / "notebooks" / "02_regime_comparison.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")

if __name__ == "__main__":
    create_notebook()
