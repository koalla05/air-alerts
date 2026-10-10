"""Standalone training and evaluation script for Ukraine Air Alert Duration & Gap models.

Usage:
    PYTHONPATH=src python -m alertpred.train --data data/raw/official_data_uk.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from alertpred.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
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
from alertpred.models.quantile_lgbm import AlertPredictionPipeline
from alertpred.evaluation.metrics import (
    evaluate_quantile_predictions,
    evaluate_classification_probabilities,
)


def run_training_pipeline(
    data_path: Path = RAW_DATA_PATH,
    output_model_path: Path = MODELS_DIR / "alertpred_pipeline.joblib",
    train_ratio: float = 0.80,
    save_processed: bool = True,
):
    print("=" * 70)
    print("UKRAINE AIR RAID ALERTS: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/7] Loading raw dataset from: {data_path}...")
    df_raw = load_raw_data(data_path)
    print(f"Loaded {len(df_raw):,} raw records.")

    # 2. Clean Alerts
    print("\n[2/7] Cleaning alerts (oblast level, dropping glitches <1m, parsing UTC)...")
    df_alerts = clean_alerts(df_raw, filter_level="oblast", min_duration_sec=60)
    print(f"Cleaned alert episodes: {len(df_alerts):,} rows across {df_alerts['region_id'].nunique()} oblasts.")
    print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")

    # Build Gap Episodes for Task B
    print("\n[3/7] Building inter-alert quiet gap episodes (Task B)...")
    df_gaps = build_gap_episodes(df_alerts)
    print(f"Constructed quiet gap episodes: {len(df_gaps):,} rows.")

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_alerts.to_parquet(PROCESSED_DATA_DIR / "cleaned_alerts.parquet", index=False)
        df_gaps.to_parquet(PROCESSED_DATA_DIR / "cleaned_gaps.parquet", index=False)
        print(f"Saved processed episodes to: {PROCESSED_DATA_DIR}")

    # 3. Train/Test Time Split
    print(f"\n[4/7] Splitting data chronologically (train ratio: {train_ratio:.0%})...")
    alerts_train, alerts_test = create_time_split(df_alerts, timestamp_col="started_at", train_ratio=train_ratio)
    gaps_train, gaps_test = create_time_split(df_gaps, timestamp_col="gap_start", train_ratio=train_ratio)
    
    split_date = alerts_test["started_at"].min()
    print(f"Time split boundary: {split_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Task A (Alerts) -> Train: {len(alerts_train):,}, Test: {len(alerts_test):,}")
    print(f"Task B (Gaps)   -> Train: {len(gaps_train):,}, Test: {len(gaps_test):,}")

    # 4. Generate As-Of Training Snapshots
    print("\n[5/7] Generating conditional 'as-of' moment snapshots...")
    # Sampling inside alerts and gaps to train conditional remaining time
    train_a_samples = sample_as_of_alert_moments(alerts_train)
    test_a_samples = sample_as_of_alert_moments(alerts_test)

    train_b_samples = sample_as_of_gap_moments(gaps_train)
    test_b_samples = sample_as_of_gap_moments(gaps_test)

    print(f"Task A Snapshots -> Train: {len(train_a_samples):,}, Test: {len(test_a_samples):,}")
    print(f"Task B Snapshots -> Train: {len(train_b_samples):,}, Test: {len(test_b_samples):,}")

    # 5. Fit Baselines
    print("\n[6/7] Evaluating Empirical Conditional Baseline...")
    baseline_a = EmpiricalConditionalBaseline().fit(alerts_train)
    baseline_preds = baseline_a.predict_batch(test_a_samples, elapsed_col="elapsed_min")
    
    baseline_eval = evaluate_quantile_predictions(
        y_true=test_a_samples["target_remaining_min"].values,
        p10=baseline_preds["baseline_p10"].values,
        p50=baseline_preds["baseline_p50"].values,
        p90=baseline_preds["baseline_p90"].values,
    )
    print("Baseline Metrics (Task A Test Set):")
    print(f"  MAE (Median P50) : {baseline_eval['mae_p50_min']:.2f} min")
    print(f"  Median AE (P50)  : {baseline_eval['median_ae_p50_min']:.2f} min")
    print(f"  P10-P90 Coverage : {baseline_eval['coverage_p10_p90']:.1%}")
    print(f"  Mean Pinball Loss: {baseline_eval['mean_pinball_loss']:.2f}")

    # 6. Fit LightGBM Models & Pipeline
    print("\n[7/7] Training LightGBM Quantile & Probability Models...")
    pipeline = AlertPredictionPipeline()
    pipeline.fit_task_a(train_a_samples)
    pipeline.fit_task_b(train_b_samples)

    # Evaluate Task A Model on Test
    lgb_preds_a = pipeline.task_a_quantile.predict(test_a_samples)
    lgb_eval_a = evaluate_quantile_predictions(
        y_true=test_a_samples["target_remaining_min"].values,
        p10=lgb_preds_a["pred_p10"].values,
        p50=lgb_preds_a["pred_p50"].values,
        p90=lgb_preds_a["pred_p90"].values,
    )
    print("\n" + "-" * 50)
    print("LIGHTGBM QUANTILE EVALUATION (Task A Test Set):")
    print(f"  MAE (Median P50) : {lgb_eval_a['mae_p50_min']:.2f} min (Baseline: {baseline_eval['mae_p50_min']:.2f} min)")
    print(f"  Median AE (P50)  : {lgb_eval_a['median_ae_p50_min']:.2f} min (Baseline: {baseline_eval['median_ae_p50_min']:.2f} min)")
    print(f"  P10-P90 Coverage : {lgb_eval_a['coverage_p10_p90']:.1%} (Ideal: ~80.0%)")
    print(f"  Mean Pinball Loss: {lgb_eval_a['mean_pinball_loss']:.2f} (Baseline: {baseline_eval['mean_pinball_loss']:.2f})")
    print("-" * 50)

    # Save Pipeline
    saved_path = pipeline.save(output_model_path)
    print(f"\nModel pipeline successfully saved to: {saved_path}")

    # Demo Inference
    print("\n" + "=" * 50)
    print("SAMPLE REAL-TIME INFERENCE:")
    demo_a = pipeline.predict_active_alert(region_id="kyiv_city", elapsed_min=30.0)
    print("Task A (Kyiv active alert running for 30m):")
    print(f"  Predicted remaining median: {demo_a['predicted_remaining_median_min']} min")
    print(f"  P10-P90 range: {demo_a['predicted_range_p10_p90_min']} min")
    print(f"  Probabilities: {demo_a['probabilities']}")

    demo_b = pipeline.predict_quiet_region(region_id="lviv", elapsed_quiet_min=180.0)
    print("\nTask B (Lviv quiet for 3 hours):")
    print(f"  Predicted next alert gap: {demo_b['predicted_next_alert_gap_median_min']} min")
    print(f"  P10-P90 range: {demo_b['predicted_range_p10_p90_min']} min")
    print(f"  Probabilities: {demo_b['probabilities']}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ukraine Air Raid Alert Models")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to raw official_data_uk.csv")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "alertpred_pipeline.joblib", help="Model output path")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train/test chronological split ratio")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        output_model_path=args.output,
        train_ratio=args.train_ratio,
    )

