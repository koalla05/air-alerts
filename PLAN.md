# Ukraine Air Raid Alerts: Real-Time Duration & Expiration Predictor

> Implementation plan for a pet project (Python only). Phase 1 of a larger time series analysis of air raid alerts in Ukraine.

---

## 1. Goal and Scope

Build a small web app that:

1. Shows a **map of Ukraine** with oblasts colored by current alert status (active / inactive).
2. For any region the user selects, shows:
   - **If an alert is active:** predicted end time (with an uncertainty range) and the probability the alert is over within the next 15 / 30 / 60 minutes.
   - **If no alert is active:** predicted time until the next alert starts (with an uncertainty range) and the probability of an alert starting within the next 1 / 3 / 6 hours.
3. Looks good: light.dark theme, smooth map, clean charts, website-friendly.

**Out of scope for Phase 1:** raion/hromada-level granularity, attack-type classification (drones vs. missiles), real-time push notifications, any claim of safety-critical accuracy.

> **Important disclaimer (put it in the UI too):** this is a statistical experiment, not a safety tool. Predictions must never replace official alerts.

---

## 2. Problem Framing

The two prediction tasks are really **time-to-event problems**, not classic forecasting:

| Task | Event | Clock starts | Censoring |
|---|---|---|---|
| **A. Alert duration** | Alert ends | Alert start | Currently active alert is *right-censored* (we know it lasted at least X minutes) |
| **B. Gap to next alert** | Next alert starts | Previous alert end | Current quiet period is right-censored |

Key insight: when a user opens the app, the alert has *already* been running for some time `t`. The right question is the **conditional** distribution:

> P(alert ends in the next Δ | it has already lasted t minutes, region = R, current context)

This is why survival analysis (or a discrete-time hazard model) is a better fit than plain regression. It handles censoring and conditional remaining time natively.

---

## 3. Data

### 3.1 Sources

| Purpose | Source | Notes |
|---|---|---|
| Live status | **alerts.in.ua API** | Requires a token (request access). Check rate limits and terms of use (especially for non-commercial/pet projects and attribution). |
| Historical alerts | Community datasets on GitHub (search "ukrainian air raid sirens dataset", e.g. by Vadimkin) | Covers alerts since Feb 2022, usually oblast-level with start/end timestamps. Check license and last update date. |
| Own history | **Your own poller** | Start polling the live API on day 1 and store every snapshot. This gives you a clean, reproducible dataset that doesn't depend on third parties. |
| Map geometry | GeoJSON of Ukrainian oblasts (geoBoundaries, GADM, or a community GeoJSON) | Must match region names/IDs from the alert source. Plan a mapping table. |
| Optional context | Weather (Open-Meteo, free), calendar (holidays, weekday), moon phase | Weather may matter for drone activity. Treat as experiments, not assumptions. |

### 3.2 Regions to handle

- 24 oblasts + Kyiv city + Autonomous Republic of Crimea (Crimea and Luhansk have unusual alert histories; decide early whether to include or flag them).
- Decide on a consistent **region ID** (e.g., `uid` from the API) and keep a `regions.csv` mapping ID → English name → Ukrainian name → GeoJSON feature ID → neighbors.

### 3.3 Canonical data schema

**`alerts` table (one row per alert episode)**

| column | description |
|---|---|
| `region_id` | Region identifier |
| `start_ts` | Alert start (UTC) |
| `end_ts` | Alert end (UTC), NULL if still active |
| `duration_min` | Derived |
| `is_censored` | True if still active at observation time |

**`events` table (raw snapshots, optional but recommended)**: timestamp, region_id, status. Lets you rebuild episodes and audit data quality.

### 3.4 Data quality checklist

- Overlapping or duplicated episodes per region
- Zero/very short alerts (technical glitches, e.g. < 1 min)
- Missing `end_ts` for old alerts
- Timezone handling: store **UTC**, display **Europe/Kyiv** (DST changes!)
- Region renames and boundary changes over time (e.g. occupied territories)
- Gaps in your own poller (log downtime so it isn't mistaken for "no alerts")

### 3.5 Storage

- **SQLite** for the local DB, **Parquet** for processed feature tables.
- Keep raw data immutable; derive everything else through reproducible scripts.

---

## 4. Feature Engineering

### 4.1 Features for Task A (alert duration / remaining time)

**Region & time context**
- Region ID (categorical), hour of day (cyclical sin/cos), day of week, month, weekend/holiday flag
- Time of day relative to sunrise/sunset (optional)

**Current alert state**
- Elapsed minutes since alert start `t`
- Number of alerts in this region in the last 6 / 12 / 24 hours
- Duration of the previous alert in this region; time since it ended

**Regional history**
- Rolling median/mean/quantile of alert duration (last 7 / 30 days) for the region
- Region baseline duration (long-term)

**Spatial context (likely the most valuable group)**
- Number and fraction of **neighboring regions** currently under alert
- Total number of regions under alert nationally
- How long neighbors have been in alert
- Whether the alert started *after* a neighbor's alert (propagation pattern, e.g. alerts moving along a trajectory)

### 4.2 Features for Task B (time to next alert)

- Time since last alert ended
- Number of alerts in the last 24h / 7d, average gap in the last 7d / 30d
- Time-of-day / day-of-week patterns (e.g. typical night attack windows)
- Neighbor states: any neighbor currently or recently alerted
- National-level activity: count of active regions, number of alerts in the last 3 hours nationwide
- Recent "intensity regime" indicator (rolling alert rate, to capture periods of escalation)

### 4.3 Leakage warning

Every feature must be computed **using only data available at prediction time**. Build features with an explicit `as_of_ts` and test this with unit tests (e.g. shift check).

---

## 5. Modeling Strategy

Use a **ladder of models**: start simple, justify every increase in complexity with a metric.

### 5.1 Baselines (must-have)

1. **Empirical conditional median**: for a region (and optionally hour-of-day bucket), predicted remaining time = median of `duration − t` among historical alerts that lasted longer than `t`.
2. **Kaplan–Meier** survival curves per region (non-parametric, handles censoring).

These baselines are surprisingly strong; any fancy model has to beat them.

### 5.2 Primary models (recommended core)

| Model | Library | Use for | Why |
|---|---|---|---|
| **Gradient boosting with quantile loss** (LightGBM / XGBoost) | `lightgbm`, `xgboost` | A and B | Fast, handles mixed features, gives P10/P50/P90 directly. Great fit for the "predicted end with range" UI. |
| **Parametric AFT models** (Weibull, log-normal, log-logistic) | `lifelines` | A and B | Interpretable, smooth distribution, handles censoring, easy to explain in a write-up. |
| **Cox proportional hazards** | `lifelines`, `scikit-survival` | A and B | Standard survival baseline with covariates; interpretable hazard ratios. |
| **Discrete-time hazard model** (LightGBM binary classifier on "minute-level" or "5-minute-level" person-period data) | `lightgbm` | A and B | Flexible, handles time-varying features (e.g. neighbors changing during the alert) and yields the full survival curve. **Probably the best fit for real-time use.** |

### 5.3 Advanced / stretch models

| Model | Library | Notes |
|---|---|---|
| Random Survival Forest / Gradient Boosted Survival | `scikit-survival` | Non-linear survival with censoring support |
| **Hawkes process** (self-exciting point process) | `tick`, or custom with `scipy` | Natural fit for "alerts cluster in time"; good for Task B and for the later time-series phase |
| Deep survival / sequence models (DeepSurv, LSTM/Transformer on alert sequences) | `pycox`, `pytorch` | Only worth it if tree models plateau; likely overkill for this data volume |
| Spatio-temporal graph models (GNN over the region adjacency graph) | `pytorch-geometric` | Good future phase for modeling propagation between regions |

### 5.4 Recommended path

1. Baselines (empirical median + Kaplan–Meier)
2. **LightGBM quantile regression** for time-to-end and time-to-next-start → quick win with P10/P50/P90
3. **Discrete-time hazard model** to produce proper probability curves ("30% chance it ends within 15 min")
4. Weibull AFT as an interpretable benchmark and sanity check
5. Hawkes / GNN as stretch goals

### 5.5 Uncertainty and calibration

- Use **quantile outputs** or **survival curves** instead of a single point estimate. In the UI show a median plus a range (e.g. "likely ends between 18:40 and 19:55").
- Apply **conformal prediction** (e.g. `MAPIE`) to get calibrated intervals if using quantile models.
- Check calibration: if the model says "50% chance within 30 min", about 50% of such cases should actually end within 30 min.

### 5.6 Training and validation protocol

- **Time-based split only.** Never shuffle (e.g., train on 2022–2024, validate on early 2025, test on the most recent months).
- **Rolling-origin backtesting** (expanding window) to see stability over time.
- Watch for **regime shifts** (changes in the nature of attacks change alert lengths): consider sample weighting toward recent data or periodic retraining.
- Build the training set by sampling "as-of" moments **inside** each alert (e.g., at 0, 5, 10, 15 ... minutes after start) so the model learns the conditional remaining-time problem the app actually faces.

### 5.7 Metrics

| Task | Metrics |
|---|---|
| Point prediction (median) | MAE, median absolute error (in minutes), relative to baseline |
| Quantiles | Pinball loss, empirical coverage of the P10–P90 interval |
| Survival / hazard | Concordance index (C-index), integrated Brier score, time-dependent AUC |
| Probabilities | Calibration plots (reliability curves), Brier score, log loss |
| Practical | Accuracy of "ends within 15/30/60 min" and "starts within 1/3/6 h" classification |

Always report metrics **per region** and **per time period**, not only globally.

---

## 6. System Architecture

```
┌────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ Alerts API      │──▶│ Ingestion/Poller │──▶│ DuckDB / SQLite    │
│ (live)          │    │ (scheduled job)  │    │ + Parquet          │
└────────────────┘    └─────────────────┘    └─────────┬─────────┘
┌────────────────┐                                      │
│ Historical CSV  │──────────────────────────────────────┤
└────────────────┘                                      ▼
                                          ┌───────────────────────┐
                                          │ Feature pipeline        │
                                          └─────────┬─────────────┘
                                                    ▼
                                          ┌───────────────────────┐
                                          │ Training & evaluation   │
                                          │ (model registry/files)  │
                                          └─────────┬─────────────┘
                                                    ▼
                    ┌───────────────────────────────────────────────┐
                    │ Inference layer: current state + features → predictions │
                    └─────────┬─────────────────────────────────────┘
                              ▼
                    ┌───────────────────────┐
                    │ Streamlit app (map +   │
                    │ region panel + charts) │
                    └───────────────────────┘
```

**Design principles**
- **Separate** training (offline) from inference (online). The app only loads saved models.
- Cache the live API response for 15–30 s to respect rate limits and keep the UI snappy.
- Keep the app resilient: if the live API fails, show a "data stale" banner with the last known update time.

---

## 7. Tech Stack (all Python)

| Layer | Choice | Alternatives |
|---|---|---|
| Data wrangling | `pandas` / `polars`, `duckdb` | |
| Geo | `geopandas`, `shapely` | |
| Survival | `lifelines`, `scikit-survival` | |
| ML | `lightgbm`, `scikit-learn`, `MAPIE` (conformal) | `xgboost`, `catboost` |
| Experiment tracking | `mlflow` or just versioned metrics JSON | `wandb` |
| App framework | **Streamlit** (fastest to a nice result) | Dash (more control), NiceGUI, Panel |
| Visualization | **Plotly** (choropleth + charts) and/or **PyDeck** (WebGL, nicer dark maps) | Folium, Altair |
| Scheduling | `APScheduler`, cron, or GitHub Actions | Prefect |
| Optional API layer | `FastAPI` | Useful if you later separate backend and frontend |
| Quality | `pytest`, `ruff`, `pre-commit`, `pydantic` for schemas | |
| Packaging | `uv` or `poetry`, with a `pyproject.toml` | |

**Recommendation:** Streamlit + Plotly for the MVP. Move to PyDeck or Dash only if you need more polish or interactivity.

---

## 8. App Design and Visuals

### 8.1 Layout

- **Main area:** choropleth map of Ukraine
  - Red/orange = active alert (optionally shade by duration so far), muted dark gray/blue = no alert
  - Hover tooltip: region name, status, time since alert start / since last alert
  - Click (or sidebar selector) to select a region
- **Side/bottom panel for the selected region:**
  - Status badge ("ALERT · 47 min" / "No alert · 3 h 12 min")
  - **If active:** predicted end (median + P10–P90 range), probability gauge ("Ends within 30 min: 41%"), and a survival/ECDF curve of the remaining time
  - **If inactive:** predicted next start (median + range), probability of alert within 1 / 3 / 6 h, and a hazard curve
  - Small history strip: last 24–48 hours of alerts as a timeline (Gantt-style)
  - Short "how confident is this?" note (model error on recent backtest)
- **Header:** national summary: number of regions under alert, last data refresh time, data freshness indicator
- **Footer:** disclaimer, data attribution, link to the repo

### 8.2 Visual style

- Dark and light theme with a restrained palette (deep navy/charcoal background, one strong accent for alerts, one calm accent for safe)
- Custom Plotly template and fonts; consistent colors across map and charts
- Subtle pulsing / glow effect for active regions if using PyDeck or custom CSS
- Optional: animated "replay" slider showing alert spread across the last 24 hours (great for portfolio value and a natural lead-in to the time-series phase)
- Responsive layout (test on mobile width; Streamlit columns collapse reasonably)
- Use Ukrainian/English language toggle (region names are already bilingual in your mapping table)

### 8.3 Tone

Alerts relate to real danger for real people. Keep the design informative and calm; avoid sensational styling, alarm sounds, or gamification.

---

## 9. Repository Structure

```
air-alert-predictor/
├── README.md
├── docs/                    # manually populated folder
├── PLAN.md                  # this document
├── pyproject.toml
├── data/
│   ├── raw/                 # immutable source dumps (gitignored if large)
│   ├── interim/
│   ├── processed/
│   └── geo/                 # GeoJSON, regions.csv with neighbor graph
├── notebooks/               # EDA, model experiments
├── src/alertpred/
│   ├── config.py
│   ├── ingestion/           # API client, poller, historical loader
│   ├── data/                # episode building, cleaning, schemas
│   ├── features/            # feature pipelines (point-in-time safe)
│   ├── models/              # baselines, LightGBM, survival, hazard
│   ├── evaluation/          # backtesting, metrics, calibration
│   ├── inference/           # live state → predictions
│   └── app/                 # Streamlit pages, components, theme
├── models/                  # saved artifacts + metrics
├── tests/
└── .github/workflows/       # lint, tests, optional scheduled poller
```

---

## 10. Implementation Roadmap (1-Day Sprint)
 
**Time budget: ~12-14 focused hours (one long day).** Everything is cut down to a working, good-looking MVP. Items marked ⏩ are deliberately deferred to "later" (see the end of this section).
 
### Scope cuts for the sprint
 
| Full plan | Sprint version |
|---|---|
| Own poller collecting history | ⏩ Use the historical dataset only; live status via API (cached) |
| LightGBM quantile + discrete-time hazard + Weibull AFT | **Empirical conditional quantiles + Kaplan-Meier** as the core; **one LightGBM quantile model** only if time remains (Block 4 stretch) |
| Neighbor-region features, conformal intervals, calibration study | ⏩ Deferred (mention in README as future work) |
| Full backtesting framework | One time-based split (train on older data, test on the most recent months) |
| Deployment, retraining, drift monitoring | ⏩ Run locally with one command (optional: Streamlit Community Cloud if time is left) |
| Tests, CI, MLflow | ⏩ Skipped, except one sanity check against data leakage |
 
**Why this is still a legitimate approach:** the empirical conditional median and Kaplan-Meier curves are the baselines any fancier model would have to beat, and they already produce exactly what the UI needs: a median remaining time plus a P10-P90 range and "ends within X min" probabilities.
 
### Block 0: Setup (≈ 0.5 h)
- [ ] Create repo and environment (`uv` or `venv`), install: `pandas`, `duckdb` (optional), `geopandas`, `lifelines`, `lightgbm`, `streamlit`, `plotly`, `requests`
- [ ] Request/obtain alerts API token (do this **first**; approval may take time. Fallback: app runs in "demo mode" on a replayed historical snapshot)
- [ ] Create folders: `data/`, `src/`, `app/`, `models/`
### Block 1: Data & regions (≈ 2 h)
- [ ] Download historical alerts dataset; load into pandas
- [ ] Build clean `alerts` episodes table (region, start UTC, end UTC, duration); drop duplicates and sub-minute glitches
- [ ] Build `regions.csv`: region ID, English/Ukrainian name, GeoJSON feature ID
- [ ] Download oblast GeoJSON and verify every region joins to the map (the classic time sink; budget for it)
- [ ] Write a tiny live-status client (API call + 20-30 s cache + fallback to last known state)
### Block 2: Quick EDA (≈ 1 h)
- [ ] Duration distribution per region and by hour of day
- [ ] Gap-between-alerts distribution per region
- [ ] Decide on the training window (the last 6-12 months are usually more representative than all data since 2022)
- [ ] Save 2-3 plots; they go straight into the README
### Block 3: Baseline models (≈ 2.5 h)
- [ ] Time-based split (e.g. last 2-3 months as test)
- [ ] **Task A (remaining alert time):** for each region (optionally by day/night bucket), compute the conditional distribution of remaining time given elapsed time `t` from historical alerts that lasted longer than `t`; output P10/P50/P90 and P(ends within 15/30/60 min)
- [ ] **Task B (time to next alert):** per region, Kaplan-Meier (via `lifelines`) or empirical gap distribution given time already elapsed since the last alert; output P10/P50/P90 and P(alert within 1/3/6 h)
- [ ] Evaluate on the test split: MAE of the median, coverage of the P10-P90 interval, one reliability check
- [ ] Wrap the logic in a single function: `predict(region_id, current_state, now) -> dict`
- [ ] **Leakage sanity check:** confirm predictions for a test moment use only data before that moment
### Block 4: Optional model upgrade (≈ 1.5 h, only if Blocks 0-3 are done on time)
- [ ] LightGBM quantile regression (P10/P50/P90) with simple features: region, hour (sin/cos), weekday, elapsed time, number of alerts in the last 24 h, number of regions currently under alert nationally
- [ ] Compare to the baseline on the test split; keep it only if it clearly wins, otherwise keep the baseline and report the comparison honestly in the README
### Block 5: App MVP (≈ 3.5 h)
- [ ] Streamlit page with dark theme (`.streamlit/config.toml`)
- [ ] Plotly choropleth of Ukraine: active alerts in one accent color, quiet regions muted; hover tooltips
- [ ] Region selector (sidebar or map click)
- [ ] Region panel:
  - Active alert: elapsed time, predicted end (median + range), probability bars for 15/30/60 min, remaining-time curve
  - No alert: time since last alert, predicted next start (median + range), probability bars for 1/3/6 h
- [ ] Header with number of regions under alert and "last updated" time; "data stale" banner on API failure
- [ ] Disclaimer in the footer (statistical experiment, not a safety tool)
- [ ] Wrap the data loaders in `st.cache_data` so the app stays fast
### Block 6: Polish & documentation (≈ 1.5 h)
- [ ] Visual pass: consistent color palette, custom Plotly template, spacing, Kyiv-time display
- [ ] README: what it does, screenshot/GIF, method summary, test-split metrics, limitations, "future work" list (pointing to the rest of this plan)
- [ ] Final check: fresh clone → install → run app with a single command
### Time plan at a glance
 
| Block | Task | Hours |
|---|---|---|
| 0 | Setup & API token | 0.5 |
| 1 | Data & regions | 2.0 |
| 2 | Quick EDA | 1.0 |
| 3 | Baseline models | 2.5 |
| 4 | Optional LightGBM | 1.5 *(stretch)* |
| 5 | App MVP | 3.5 |
| 6 | Polish & README | 1.5 |
| | **Total** | **~11 h (+1.5 h stretch)** |
 
**If you fall behind, cut in this order:** (1) Block 4, (2) EDA plots beyond the essentials, (3) Task B (next-start prediction), keeping only Task A, (4) region click-to-select (use the sidebar selector instead).
 
**Priority rule:** a working map with live status and Task A predictions beats a half-finished advanced model. Get the end-to-end path (data → prediction → map) working by roughly hour 7, then spend the remaining time on polish.
 
### Deferred to later iterations (⏩)
- Own poller and continuously growing dataset
- Neighbor-region / spatial-propagation features
- Discrete-time hazard model, Weibull AFT, conformal intervals
- Rolling-origin backtesting, per-region calibration analysis
- Hawkes process, graph neural networks
- Scheduled retraining, monitoring, deployment, CI/tests
- Replay animation and the broader time series analysis phase
---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Low predictability.** Alerts depend on events that aren't in the data | Be honest in the UI; show wide intervals; always compare to baselines |
| **Non-stationarity** (patterns change over time) | Time-based validation, recency weighting, periodic retraining |
| **API access/rate limits/terms changes** | Cache aggressively, keep own history, abstract the data source behind an interface |
| **Data leakage** | Point-in-time features, explicit `as_of_ts`, unit tests |
| **Misuse of predictions as safety guidance** | Prominent disclaimer, no push notifications, no "safe to leave shelter" wording |
| **Region/geometry mismatches** | Single `regions.csv` as source of truth, tests for ID coverage |
| **Ethics / sensitivity** | Calm tone, no sensationalism, credit data sources |

---

## 12. Definition of Done for the MVP

- Map renders all regions with correct live status, refreshed automatically
- Selecting a region shows either predicted end (active) or predicted next start (inactive) with uncertainty ranges
- Models beat the simple baselines on a time-based backtest, with results documented
- Predictions use only point-in-time features (tested)
- Data staleness and API failures are handled gracefully
- App deployed publicly or runnable with a single command
- README contains screenshots, methodology, metrics, limitations, and a disclaimer

---

## 13. Key Decisions to Make Early

1. Include Crimea / temporarily occupied regions in the model, or only on the map?
2. Granularity of the time grid for the hazard model (1 min vs. 5 min)
3. Streamlit + Plotly vs. PyDeck for the map (try Plotly first)
4. How to host the poller (local machine, GitHub Actions cron, small VPS)
5. UI language(s): Ukrainian, English, or both