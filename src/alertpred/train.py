"""Standalone training and evaluation script for Ukraine Air Alert Duration & Gap models.

Usage:
    PYTHONPATH=src python -m alertpred.train
    PYTHONPATH=src python -m alertpred.train --data data/raw/official_data_uk.csv --no-api-backfill
"""

import argparse
from pathlib import Path

from alertpred.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RECENCY_HALF_LIFE_DAYS,
)
from alertpred.data.build_dataset import build_master_alerts_dataset
from alertpred.data.cleaning import create_time_split
from alertpred.features.builder import (
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
    compute_recency_weight,
)
from alertpred.models.baseline import EmpiricalConditionalBaseline
from alertpred.models.quantile_lgbm import AlertPredictionPipeline
from alertpred.evaluation.metrics import evaluate_quantile_predictions


def _attach_recency_weight(samples_df, episodes_df, join_cols, weight_col="recency_weight"):
    """`sample_as_of_*_moments` doesn't carry arbitrary extra columns through
    its row-by-row construction, so the per-episode recency weight computed
    before as-of sampling is joined back on (region_id, start timestamp)."""
    lookup = episodes_df[["region_id", join_cols[1], weight_col]].rename(columns={join_cols[1]: join_cols[0]})
    return samples_df.merge(lookup, on=["region_id", join_cols[0]], how="left")


def run_training_pipeline(
    data_path: Path = RAW_DATA_PATH,
    output_model_path: Path = MODELS_DIR / "alertpred_pipeline.joblib",
    train_ratio: float = 0.80,
    use_api_backfill: bool = True,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    save_processed: bool = True,
):
    print("=" * 70)
    print("UKRAINE AIR RAID ALERTS: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Build the master dataset: bias-corrected oblast rollup (data.rollup)
    #    + alerts.in.ua backfill + volunteer gap-fill - see notebook 01 for
    #    the step-by-step walkthrough and validation of each source.
    print("\n[1/8] Building master alerts dataset (rollup + API backfill + gap-fill)...")
    df_alerts, df_gaps = build_master_alerts_dataset(
        raw_data_path=data_path, use_api_backfill=use_api_backfill, verbose=True,
    )
    print(f"\nAlert episodes: {len(df_alerts):,} across {df_alerts['region_id'].nunique()} oblasts")
    print(f"Gap episodes  : {len(df_gaps):,}")
    print(f"Date range: {df_alerts['started_at'].min().strftime('%Y-%m-%d')} to {df_alerts['started_at'].max().strftime('%Y-%m-%d')}")

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_alerts.to_parquet(PROCESSED_DATA_DIR / "cleaned_alerts.parquet", index=False)
        df_gaps.to_parquet(PROCESSED_DATA_DIR / "cleaned_gaps.parquet", index=False)
        print(f"Saved processed episodes to: {PROCESSED_DATA_DIR}")

    # 2. Recency weights (notebooks/02_regime_comparison.ipynb found this
    #    beats training only on data from after the oblast->raion reporting
    #    shift): an episode `half_life_days` old counts half as much as one
    #    dated "now" (the dataset's own latest timestamp).
    print(f"\n[2/8] Computing recency weights (half-life: {half_life_days:.0f} days)...")
    reference_ts = df_alerts["started_at"].max()
    df_alerts["recency_weight"] = compute_recency_weight(df_alerts["started_at"], reference_ts, half_life_days)
    df_gaps["recency_weight"] = compute_recency_weight(df_gaps["gap_start"], reference_ts, half_life_days)
    print(f"Weight range: {df_alerts['recency_weight'].min():.4f} (oldest) to {df_alerts['recency_weight'].max():.3f} (newest)")

    # 3. Train/Test Time Split
    print(f"\n[3/8] Splitting data chronologically (train ratio: {train_ratio:.0%})...")
    alerts_train, alerts_test = create_time_split(df_alerts, timestamp_col="started_at", train_ratio=train_ratio)
    gaps_train, gaps_test = create_time_split(df_gaps, timestamp_col="gap_start", train_ratio=train_ratio)

    split_date = alerts_test["started_at"].min()
    print(f"Time split boundary: {split_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Task A (Alerts) -> Train: {len(alerts_train):,}, Test: {len(alerts_test):,}")
    print(f"Task B (Gaps)   -> Train: {len(gaps_train):,}, Test: {len(gaps_test):,}")

    # 4. Generate As-Of Training Snapshots
    print("\n[4/8] Generating conditional 'as-of' moment snapshots...")
    train_a_samples = sample_as_of_alert_moments(alerts_train)
    test_a_samples = sample_as_of_alert_moments(alerts_test)
    train_b_samples = sample_as_of_gap_moments(gaps_train)
    test_b_samples = sample_as_of_gap_moments(gaps_test)

    train_a_samples = _attach_recency_weight(train_a_samples, alerts_train, ("start_ts", "started_at"))
    train_b_samples = _attach_recency_weight(train_b_samples, gaps_train, ("gap_start_ts", "gap_start"))

    print(f"Task A Snapshots -> Train: {len(train_a_samples):,}, Test: {len(test_a_samples):,}")
    print(f"Task B Snapshots -> Train: {len(train_b_samples):,}, Test: {len(test_b_samples):,}")

    # 5. Fit Baseline (recency-weighted, so the printed comparison matches
    #    what actually ships rather than an unweighted strawman)
    print("\n[5/8] Evaluating Empirical Conditional Baseline (recency-weighted)...")
    baseline_a = EmpiricalConditionalBaseline().fit(alerts_train, weight_col="recency_weight")
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

    # 6. Fit LightGBM Models & Pipeline (recency-weighted)
    print("\n[6/8] Training LightGBM Quantile & Probability Models (recency-weighted)...")
    pipeline = AlertPredictionPipeline()
    pipeline.fit_task_a(train_a_samples, weight_col="recency_weight")
    pipeline.fit_task_b(train_b_samples, weight_col="recency_weight")

    # 7. Evaluate Task A Model on Test
    print("\n[7/8] Evaluating on held-out test set...")
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

    # 8. Save Pipeline
    print("\n[8/8] Saving pipeline...")
    saved_path = pipeline.save(output_model_path)
    print(f"Model pipeline successfully saved to: {saved_path}")

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
    parser.add_argument("--no-api-backfill", action="store_true", help="Skip the alerts.in.ua live backfill (GitHub+volunteer data only)")
    parser.add_argument("--half-life-days", type=float, default=RECENCY_HALF_LIFE_DAYS, help="Recency weight half-life in days")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        output_model_path=args.output,
        train_ratio=args.train_ratio,
        use_api_backfill=not args.no_api_backfill,
        half_life_days=args.half_life_days,
    )
