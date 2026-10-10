# Ukraine Air Raid Alerts: Real-Time Duration & Expiration Predictor

Statistical and machine learning system for predicting remaining alert durations and inter-alert quiet times across Ukrainian oblasts. Aligned with [PLAN.md](PLAN.md).

> **Important Disclaimer:** This project is a statistical experiment and research benchmark. It is **not** a civil defense tool and must **never** be used in place of official alerts from the State Emergency Service of Ukraine or official air raid siren channels.

---

## 1. Problem Formulation

Standard regression directly on total duration is fundamentally flawed because real users query the system *while an alert is already in progress* (or while waiting in a quiet period). This is a conditional **time-to-event / survival analysis problem**:

| Task | Event | Observation Clock | Censoring & Conditional Formulation |
|---|---|---|---|
| **Task A: Remaining Alert Duration** | Alert ends | Siren starts ($t=0$) | At elapsed time $t$, predict $P(\text{ends in } \Delta \mid \text{duration} > t)$ |
| **Task B: Time to Next Alert** | Next alert starts | Previous alert ends ($t_{quiet}=0$) | At quiet elapsed time $t_{quiet}$, predict $P(\text{starts in } \Delta \mid \text{gap} > t_{quiet})$ |

---

## 2. Methodology & Ladder of Models

Following **PLAN.md Section 5**, models are evaluated against strong non-parametric baselines:

1. **Empirical Conditional Quantile Baseline:** Non-parametric empirical conditional distribution derived from historical alerts that exceeded elapsed time $t$.
2. **Kaplan-Meier Survival Baseline:** Estimation of $S(t) = P(D > t)$ with conditional survival probability $S(\Delta \mid t) = \frac{S(t + \Delta)}{S(t)}$.
3. **LightGBM Quantile Regressors:** Gradient-boosted quantile regressions at $\alpha \in \{0.10, 0.50, 0.90\}$ to predict median remaining time and calibrated uncertainty intervals $[P10, P90]$.
4. **LightGBM Binary Probability Classifiers:** Predicting exact calibrated probabilities:
   - Task A: $P(\text{ends within } 15 / 30 / 60 \text{ min})$
   - Task B: $P(\text{starts within } 1 / 3 / 6 \text{ hours})$

---

## 3. Dataset

- Source: [Ukrainian Air Raid Sirens Dataset (Vadimkin)](https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv)
- Covers 299,000+ records from March 2022 to September 2026.
- Filtered to macroscopic oblast-level episodes ($65,172$ cleaned alert episodes across 24 oblasts).
- Sub-minute technical glitches ($< 1$ min) and unbounded outliers are removed.
- Validated with strict **point-in-time chronological train/test split** (training on older data, testing on held-out recent months) to prevent future data leakage.

---

## 4. Project Structure

```
.
├── PLAN.md                               # Architecture and implementation plan
├── README.md                             # Project overview and instructions
├── requirements.txt                      # Project dependencies
├── data/
│   ├── raw/                              # Immutable raw alerts dataset
│   │   └── official_data_uk.csv
│   └── processed/                        # Parquet cleaned episodes (alerts & gaps)
├── notebooks/
│   ├── 01_model_training_pipeline.ipynb  # End-to-end interactive training & evaluation pipeline
│   └── generate_notebook.py              # Notebook generator script
├── src/alertpred/
│   ├── __init__.py
│   ├── config.py                         # Constants, prediction horizons, region mappings
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py                   # Data ingestion, cleaning, UTC handling, episode builder
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py                    # "As-of" moment sampling and point-in-time features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                   # Empirical conditional quantiles & Kaplan-Meier estimators
│   │   └── quantile_lgbm.py              # LightGBM quantile regression & probability models pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                    # Pinball loss, coverage, MAE, Brier score, ROC-AUC
│   └── train.py                          # Standalone CLI training & evaluation script
└── models/
    └── alertpred_pipeline.joblib         # Serialized production pipeline ready for inference
```

---

## 5. Quickstart

### Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Training Pipeline

To train the models, run evaluation on the test set, and save model artifacts:

```bash
PYTHONPATH=src python -m alertpred.train --data data/raw/official_data_uk.csv
```

### Interactive Jupyter Notebook

Open and explore the step-by-step pipeline:

```bash
jupyter notebook notebooks/01_model_training_pipeline.ipynb
```

---

## 6. Sample Live Inference

```python
from alertpred.models.quantile_lgbm import AlertPredictionPipeline

# Load serialized pipeline
pipeline = AlertPredictionPipeline.load("models/alertpred_pipeline.joblib")

# Scenario 1: Active alert in Kyiv that has lasted 30 minutes
res_active = pipeline.predict_active_alert(region_id="kyiv_city", elapsed_min=30.0)
print(res_active)
# Output:
# Predicted remaining median: 41.6 min
# Uncertainty range [P10, P90]: [2.9, 80.8] min
# Probabilities:
#   Ends within 15 min: 26.2%
#   Ends within 30 min: 39.6%
#   Ends within 60 min: 65.2%

# Scenario 2: Quiet region in Lviv quiet for 3 hours
res_quiet = pipeline.predict_quiet_region(region_id="lviv", elapsed_quiet_min=180.0)
print(res_quiet)
# Output:
# Predicted next alert gap: 2494.3 min (~41.5 hrs)
# Probabilities:
#   Starts within 1h: 3.0%
#   Starts within 3h: 6.7%
#   Starts within 6h: 10.7%
```