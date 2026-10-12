# Ukraine Air Raid Alerts: Real-Time Duration & Expiration Predictor

Statistical and machine learning system for predicting remaining alert durations and inter-alert quiet times across Ukrainian oblasts and raions, with a live Streamlit map and a historical dynamics dashboard. Aligned with [PLAN.md](PLAN.md).

> **Important Disclaimer:** This project is a statistical experiment and research benchmark. It is **not** a civil defense tool and must **never** be used in place of official alerts from the State Emergency Service of Ukraine or official air raid siren channels.

---

## 1. Problem Formulation

Standard regression directly on total duration is fundamentally flawed because real users query the system *while an alert is already in progress* (or while waiting in a quiet period). This is a conditional **time-to-event / survival analysis problem**:

| Task | Event | Observation Clock | Censoring & Conditional Formulation |
|---|---|---|---|
| **Task A: Remaining Alert Duration** | Alert ends | Siren starts ($t=0$) | At elapsed time $t$, predict $P(\text{ends in } \Delta \mid \text{duration} > t)$ |
| **Task B: Time to Next Alert** | Next alert starts | Previous alert ends ($t_{quiet}=0$) | At quiet elapsed time $t_{quiet}$, predict $P(\text{starts in } \Delta \mid \text{gap} > t_{quiet})$ |

---

## 2. Data Pipeline

The dataset is assembled from **three sources** with different coverage and reliability, merged into one canonical oblast-level episode table (`data.build_dataset.build_master_alerts_dataset`, walked through step by step in [notebook 01](notebooks/01_model_training_pipeline.ipynb)):

| Source | Role | Notes |
|---|---|---|
| [Official GitHub dataset (Vadimkin)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) | Primary, deep history (Mar 2022 → ~5 days ago) | Government-reported sirens, all granularities |
| [alerts.in.ua API](https://devs.alerts.in.ua/) | Recent backfill (last ~30 days) + live status | Token-gated, cached and rate-limited (see §2.3) |
| [Volunteer dataset (same repo)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) | Narrow gap-fill only | A *differently-measured* signal (eTryvoga channel monitoring) — see §2.2 |

### 2.1 The granularity-shift bias, and the rollup fix

Starting around Aug–Nov 2025, Ukraine's official reporting shifted from oblast-wide blanket alerts to localized raion/hromada alerts. A naive `level == 'oblast'` filter makes alert *counts* collapse to near zero after that point — not because alerts stopped, but because they started being reported at finer granularity. `data.rollup.merge_intervals_to_episodes` fixes this by merging overlapping/touching sub-oblast alerts into oblast-level episodes, tagging each as `oblast` (native) or `oblast_rollup` (derived) so the correction stays auditable rather than hidden. This also guards against a real bug found and fixed during development: a handful of corrupt, never-closed source rows (one spanning 604 days) used to chain-merge everything between their start and end into one giant episode that then got silently dropped — destroying up to a year and a half of otherwise-valid data for some regions. Individual input records longer than 24h are now filtered out *before* merging (regression-tested in `tests/test_rollup.py`).

### 2.2 Why the volunteer dataset is scoped narrowly

Cross-checked oblast-for-oblast against the official rollup, the volunteer feed reports far more, much shorter episodes that don't even align in time (e.g. Kharkiv, Aug 2026: ~490 episodes averaging ~19 min there vs. ~130 averaging ~290 min in the official rollup) — a different detection methodology, not just a different granularity of the same events. It's used *only* to fill the specific date gap left between the GitHub CSV's last record and the alerts.in.ua backfill's earliest one, never blended across the whole timeline.

### 2.3 Live API: caching and rate limiting

`ingestion.alerts_client` caches every alerts.in.ua response to `data/cache/alerts_in_ua/` and enforces a **persisted sliding-window call budget** per endpoint bucket (shared across separate process runs, not just spacing between consecutive calls — a plain per-call interval isn't enough to respect the server's actual rolling 30-calls/10-min limit). History responses are also served from disk with no network call at all when younger than 2 hours, since that endpoint's window barely covers one full 27-region pass.

### 2.4 Current dataset size

90,061 cleaned episodes across 24 oblasts (Crimea, Sevastopol, and Luhansk are excluded — see §6), spanning 2022-03-15 to today with no uncovered gap: 56,648 native oblast-level, 32,835 rollup-derived, 578 from the volunteer gap-fill.

---

## 3. Methodology & Modeling

Following **PLAN.md Section 5**, models are evaluated against strong non-parametric baselines, with two further choices validated empirically (not assumed) in dedicated comparison notebooks:

1. **Empirical Conditional Quantile Baseline** — non-parametric conditional distribution from historical alerts that exceeded elapsed time $t$, supporting optional sample weights.
2. **Kaplan–Meier Survival Baseline** — $S(t) = P(D > t)$, conditional survival $S(\Delta \mid t) = S(t+\Delta)/S(t)$, also weight-aware (via lifelines' native `weights=`).
3. **LightGBM Quantile Regressors** — $\alpha \in \{0.10, 0.50, 0.90\}$ for median remaining time and $[P10, P90]$ intervals.
4. **LightGBM Binary Probability Classifiers** — calibrated $P(\text{ends within } 15/30/60\text{ min})$ (Task A) and $P(\text{starts within } 1/3/6\text{ h})$ (Task B).

**Recency weighting** ([notebook 02](notebooks/02_regime_comparison.ipynb)): rather than hard-cutting training data at the oblast→raion reporting shift, every episode is weighted by `0.5 ** (age_days / 60)` — an episode 60 days old counts half as much as a fresh one. Compared head-to-head against a hard regime cutover (same models, same held-out recent test window), recency weighting won every metric comparison once the rollup bias fix restored the full history it draws on.

**Neighbor-region context** ([notebook 03](notebooks/03_neighbor_features.ipynb)): the quantile models (not the probability classifiers, which showed no benefit) also take point-in-time spatial features — how many of a region's direct neighbors, and how many regions nationally, were active at that exact historical moment (`features.builder.compute_neighbor_active_features`, a leakage-safe backward as-of join, see §5). This improved Task A and Task B quantile accuracy modestly (1–3% on MAE/pinball loss).

---

## 4. The App

```bash
PYTHONPATH=src streamlit run app.py
```

Two tabs:

- **🚨 Live Dashboard** — a **raion-level** choropleth (136 raions + Kyiv city + Sevastopol, with oblast borders overlaid) reflecting true current alert locations, since oblast-wide alerts are now rare (§2.1); a sidebar selector for oblast *or* a specific raion within it (raion predictions reuse the parent oblast's model, anchored to the raion's own live elapsed time — raion-level training needs more accumulated history first); live/demo mode with a "Simulate Custom State" override for demos.
- **📊 Statistics** — basic dynamics for the curious: a time-range picker (last day/week/month, any calendar year, or a custom range) and a region picker (all of Ukraine or one oblast), four summary numbers, and up to three charts (alerts-over-time, duration distribution, and — only for the national view — a most-active-regions breakdown). Deliberately kept small rather than a kitchen-sink dashboard.

Live status comes from alerts.in.ua's `/v1/alerts/active.json` when `ALERTS_IN_UA_TOKEN` is set (copy `.env.example` → `.env`), cascading both ways between granularities (a raion alert marks its parent oblast active; an oblast-wide alert marks all its raions active). Falls back to a realistic demo snapshot otherwise.

---

## 5. Leakage Prevention

Every feature is computed from an explicit `as_of_ts`, using only information that would have been available at that exact historical moment — enforced with a **shift check** test suite (PLAN.md §4.3): take a feature built at a fixed `as_of_ts`, move the same input event across that boundary, and confirm the feature only reacts once the event is in the past (`tests/test_leakage.py`). Covers the neighbor-context join, the point-in-time rolling-alert-count feature, as-of sampling never exceeding an episode's true known end, and the chronological train/test split itself.

```bash
pytest   # 45 tests: leakage, rollup merging (incl. the corrupt-record regression), cleaning, features, baselines
```

---

## 6. Known Limitations

- **Crimea and Luhansk** have no usable siren data in either source: Crimea has zero rows in 4.5 years of history (outside Ukrainian civil defense control), and Luhansk's last real record is from 2022-04-04 — both regions have had a "permanent siren" since 2022 (Luhansk from 2022-04-04 16:45 UTC, Crimea from 2022-12-10 22:22 UTC) that was never logged as a discrete event by either dataset. Not yet given separate handling in the model or map (tracked for a later iteration).
- **Raion-level training** doesn't exist yet — the live map and selector operate at raion granularity, but predictions still come from the parent oblast's model (clearly labeled in the UI).
- **Historical data has no uncovered gap as of this writing**, but the alerts.in.ua backfill only reaches ~30 days back; if the GitHub CSV's tail falls further behind than that *and* the volunteer dataset isn't refreshed, a gap could reopen — the notebook's validation step reports this explicitly if it happens.

---

## 7. Project Structure

```
.
├── PLAN.md                               # Architecture and implementation plan
├── README.md                             # Project overview and instructions
├── requirements.txt                      # Project dependencies
├── pytest.ini                            # Test configuration
├── app.py                                # Streamlit app (Live Dashboard + Statistics tabs)
├── .env.example                          # Copy to .env and set ALERTS_IN_UA_TOKEN
├── data/
│   ├── raw/                              # Immutable source dumps (official + volunteer CSVs)
│   ├── processed/                        # Parquet cleaned episodes (alerts & gaps)
│   ├── cache/alerts_in_ua/               # Cached/rate-limited API responses
│   └── geo/                              # Oblast + raion GeoJSON, SOURCES.md (provenance/license)
├── notebooks/
│   ├── 01_model_training_pipeline.ipynb  # Data pipeline walkthrough, EDA, baseline/LightGBM training
│   ├── 02_regime_comparison.ipynb        # Hard cutover vs. recency-weighted training comparison
│   ├── 03_neighbor_features.ipynb        # With vs. without neighbor-region context comparison
│   └── generate_notebook*.py             # Notebook generator scripts (source of truth; regenerate, don't hand-edit)
├── src/alertpred/
│   ├── config.py                         # Constants, region mappings, neighbor adjacency graph
│   ├── data/
│   │   ├── cleaning.py                   # Oblast-filtered cleaning, volunteer-data loader, time split
│   │   ├── rollup.py                     # Sub-oblast → oblast interval merging (the bias fix)
│   │   └── build_dataset.py              # Assembles the 3-source master dataset
│   ├── ingestion/
│   │   ├── alerts_client.py              # Cached, rate-limited alerts.in.ua HTTP client
│   │   ├── backfill.py                   # Per-region history fetch + rollup
│   │   ├── live.py                       # Live status (oblast + raion, two-way cascade)
│   │   └── alerts_regions.py             # alerts.in.ua UID / status-order mappings
│   ├── geo/
│   │   ├── oblasts.py                    # Oblast GeoJSON loader (winding-order fix)
│   │   └── raions.py                     # Raion registry: spatial join to parent oblast, name aliases
│   ├── features/
│   │   └── builder.py                    # As-of sampling, recency weights, neighbor-context features
│   ├── models/
│   │   ├── baseline.py                   # Empirical conditional quantiles & Kaplan-Meier (weight-aware)
│   │   └── quantile_lgbm.py              # LightGBM quantile/probability models + inference pipeline
│   ├── evaluation/
│   │   └── metrics.py                    # Pinball loss, coverage, MAE, Brier score, ROC-AUC
│   ├── app/
│   │   ├── map.py                        # Raion/oblast choropleth figures
│   │   ├── stats.py                      # Statistics tab
│   │   └── components.py                 # Probability chips, distribution curve
│   └── train.py                          # Standalone CLI training & evaluation script
├── tests/                                 # pytest suite (leakage, rollup, cleaning, features, baselines)
└── models/
    └── alertpred_pipeline.joblib         # Serialized production pipeline ready for inference
```

---

## 8. Quickstart

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Live Data (optional but recommended)

```bash
cp .env.example .env
# Request a token at https://devs.alerts.in.ua/ and set ALERTS_IN_UA_TOKEN in .env
```

Without a token, training falls back to the GitHub + volunteer sources only, and the app runs in demo mode.

### Running the Training Pipeline

```bash
PYTHONPATH=src python -m alertpred.train
# Flags: --no-api-backfill (skip the live fetch), --half-life-days, --train-ratio
```

### Launch the Interactive Web App

```bash
PYTHONPATH=src streamlit run app.py
```

### Run the Tests

```bash
pytest
```

### Interactive Notebooks

```bash
jupyter notebook notebooks/01_model_training_pipeline.ipynb
```

---

## 9. Sample Live Inference

```python
from alertpred.models.quantile_lgbm import AlertPredictionPipeline
from alertpred.ingestion.live import get_live_alerts_status

pipeline = AlertPredictionPipeline.load("models/alertpred_pipeline.joblib")

# Pass the current live_status snapshot so the model's neighbor-context
# features reflect what's actually happening right now, not a
# quiet-everywhere fallback.
live_status = get_live_alerts_status()

# Scenario 1: Active alert in Kyiv that has lasted 30 minutes
res_active = pipeline.predict_active_alert(region_id="kyiv_city", elapsed_min=30.0, live_status=live_status)
print(res_active)
# {'predicted_remaining_median_min': ..., 'predicted_range_p10_p90_min': [...],
#  'probabilities': {'prob_ends_within_15m': ..., 'prob_ends_within_30m': ..., 'prob_ends_within_60m': ...}}

# Scenario 2: Quiet region in Lviv, quiet for 3 hours
res_quiet = pipeline.predict_quiet_region(region_id="lviv", elapsed_quiet_min=180.0, live_status=live_status)
print(res_quiet)
# {'predicted_next_alert_gap_median_min': ..., 'predicted_range_p10_p90_min': [...],
#  'probabilities': {'prob_starts_within_1h': ..., 'prob_starts_within_3h': ..., 'prob_starts_within_6h': ...}}
```
