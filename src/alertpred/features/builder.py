import pandas as pd
import numpy as np
from typing import List, Optional
from ..config import (
    ELAPSED_SAMPLE_STEPS,
    TASK_A_HORIZONS_MIN,
    TASK_B_HORIZONS_MIN,
    RECENCY_HALF_LIFE_DAYS,
)


def compute_recency_weight(
    timestamps: pd.Series,
    reference_ts: Optional[pd.Timestamp] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> pd.Series:
    """
    Exponential recency weight: `0.5 ** (age_days / half_life_days)`, so an
    episode `half_life_days` old counts half as much as one dated
    `reference_ts` itself (default: the latest timestamp in the series,
    i.e. "now" for a freshly-built training set). See
    notebooks/02_regime_comparison.ipynb for why training on the full
    history this way beats a hard cutoff at the oblast->raion reporting
    shift.
    """
    if reference_ts is None:
        reference_ts = timestamps.max()
    age_days = (reference_ts - timestamps).dt.total_seconds() / 86400.0
    return 0.5 ** (age_days / half_life_days)


def compute_cyclical_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """Derives cyclical hour and day-of-week sine/cosine representations."""
    dt = pd.to_datetime(timestamps, utc=True).dt
    hour = dt.hour + dt.minute / 60.0
    dow = dt.dayofweek

    features = pd.DataFrame(index=timestamps.index)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    features["is_weekend"] = (dow >= 5).astype(int)
    features["month"] = dt.month
    return features


def sample_as_of_alert_moments(
    df_alerts: pd.DataFrame,
    elapsed_steps: Optional[List[int]] = None,
    max_elapsed_sample: int = 180,
) -> pd.DataFrame:
    """
    Transforms alert episodes into point-in-time 'as-of' training snapshots (PLAN.md Section 5.6).
    At each step t (elapsed_min since start), if true duration D > t:
      - elapsed_min = t
      - target: remaining_min = D - t
      - target binary indicators: ends_within_15, ends_within_30, ends_within_60
    """
    if elapsed_steps is None:
        elapsed_steps = ELAPSED_SAMPLE_STEPS

    records = []
    for _, row in df_alerts.iterrows():
        duration = row["duration_min"]
        start_ts = row["started_at"]
        region_id = row["region_id"]
        region_en = row["region_en"]

        for t in elapsed_steps:
            if t < duration and t <= max_elapsed_sample:
                as_of_ts = start_ts + pd.Timedelta(minutes=t)
                remaining_min = duration - t
                
                rec = {
                    "region_id": region_id,
                    "region_en": region_en,
                    "as_of_ts": as_of_ts,
                    "start_ts": start_ts,
                    "elapsed_min": float(t),
                    "total_duration_min": duration,
                    "target_remaining_min": float(remaining_min),
                }
                for h in TASK_A_HORIZONS_MIN:
                    rec[f"target_ends_within_{h}m"] = int(remaining_min <= h)
                records.append(rec)

    df_samples = pd.DataFrame(records)
    if not df_samples.empty:
        df_samples["as_of_ts"] = pd.to_datetime(df_samples["as_of_ts"], utc=True)
        time_feats = compute_cyclical_time_features(df_samples["as_of_ts"])
        df_samples = pd.concat([df_samples, time_feats], axis=1)
    return df_samples


def sample_as_of_gap_moments(
    df_gaps: pd.DataFrame,
    elapsed_steps: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Transforms quiet gap episodes into 'as-of' training snapshots for Task B.
    At quiet elapsed time t_quiet since previous alert ended:
      - target: remaining_quiet_min = G - t_quiet
      - target binary indicators: starts_within_1h, 3h, 6h
    """
    if elapsed_steps is None:
        elapsed_steps = [0, 30, 60, 120, 240, 480, 720, 1440]

    records = []
    for _, row in df_gaps.iterrows():
        gap_total = row["gap_duration_min"]
        gap_start = row["gap_start"]
        region_id = row["region_id"]
        region_en = row["region_en"]
        prev_duration = row.get("prev_alert_duration_min", 45.0)

        for t in elapsed_steps:
            if t < gap_total:
                as_of_ts = gap_start + pd.Timedelta(minutes=t)
                remaining_gap = gap_total - t
                
                rec = {
                    "region_id": region_id,
                    "region_en": region_en,
                    "as_of_ts": as_of_ts,
                    "gap_start_ts": gap_start,
                    "elapsed_quiet_min": float(t),
                    "prev_alert_duration_min": float(prev_duration),
                    "total_gap_min": gap_total,
                    "target_remaining_gap_min": float(remaining_gap),
                }
                for h in TASK_B_HORIZONS_MIN:
                    rec[f"target_starts_within_{h // 60}h"] = int(remaining_gap <= h)
                records.append(rec)

    df_samples = pd.DataFrame(records)
    if not df_samples.empty:
        df_samples["as_of_ts"] = pd.to_datetime(df_samples["as_of_ts"], utc=True)
        time_feats = compute_cyclical_time_features(df_samples["as_of_ts"])
        df_samples = pd.concat([df_samples, time_feats], axis=1)
    return df_samples


def extract_point_in_time_features(
    df_samples: pd.DataFrame,
    df_alerts_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enriches samples with point-in-time safe historical and concurrent context:
    - Alerts in this region in the last 6, 12, 24 hours
    - National concurrent active regions at as_of_ts
    """
    df = df_samples.copy()
    if df.empty:
        return df

    # Pre-sort historical alerts for quick filtering
    alerts_hist = df_alerts_history.sort_values("started_at").copy()

    # Pre-compute rolling counts per region
    recent_6h_counts = []
    recent_24h_counts = []

    for _, row in df.iterrows():
        as_of = row["as_of_ts"]
        reg_id = row["region_id"]
        
        # Alerts in region ending or starting in the last 24h before as_of
        mask_reg = (alerts_hist["region_id"] == reg_id) & (alerts_hist["started_at"] < as_of)
        recent_reg = alerts_hist[mask_reg]
        
        count_6h = (recent_reg["started_at"] >= (as_of - pd.Timedelta(hours=6))).sum()
        count_24h = (recent_reg["started_at"] >= (as_of - pd.Timedelta(hours=24))).sum()
        
        recent_6h_counts.append(int(count_6h))
        recent_24h_counts.append(int(count_24h))

    df["alerts_last_6h"] = recent_6h_counts
    df["alerts_last_24h"] = recent_24h_counts

    return df


def _active_at_timestamps(as_of_ts: pd.Series, episodes_sorted: pd.DataFrame) -> pd.Series:
    """For each as_of_ts, was it inside some [started_at, finished_at) in
    `episodes_sorted` (columns started_at/finished_at, sorted by started_at,
    reset_index(drop=True))? A backward as-of join, so this only ever
    matches an episode that had already started by as_of_ts."""
    tmp = pd.DataFrame({"as_of_ts": as_of_ts}).sort_values("as_of_ts")
    merged = pd.merge_asof(tmp, episodes_sorted, left_on="as_of_ts", right_on="started_at", direction="backward")
    merged.index = tmp.index
    active = (merged["as_of_ts"] < merged["finished_at"]).fillna(False)
    return active.reindex(as_of_ts.index)


def compute_neighbor_active_features(
    samples_df: pd.DataFrame,
    df_alerts: pd.DataFrame,
    neighbors_map: dict,
) -> pd.DataFrame:
    """
    Point-in-time-safe spatial context (PLAN.md Section 4.1, "likely the
    most valuable feature group", deferred in the first pass - see
    notebooks/03_neighbor_features.ipynb for the comparison this enables).

    For each (region_id, as_of_ts) row in `samples_df`, computes how many of
    its direct neighbors (per `neighbors_map`, e.g. config.OBLAST_NEIGHBORS)
    - and how many regions nationally - had an active episode at exactly
    that historical moment, via a backward as-of join against each region's
    own sorted episode table in `df_alerts`. Only episodes that had already
    started by `as_of_ts` can match, so no future information leaks in.

    Adds `neighbors_active_count`, `neighbors_active_frac` (0 when a region
    has no neighbors, e.g. Sevastopol), and `national_active_count`.
    """
    df = samples_df.copy()
    episodes_by_region = {
        region_id: group.sort_values("started_at")[["started_at", "finished_at"]].reset_index(drop=True)
        for region_id, group in df_alerts.groupby("region_id")
    }

    # National count: one as-of join per region over the *unique* as_of_ts
    # values (shared across every sample at that moment), not one per row.
    unique_ts = pd.Series(df["as_of_ts"].unique())
    national_count = pd.Series(0, index=unique_ts.index, dtype=float)
    for ep in episodes_by_region.values():
        if ep.empty:
            continue
        national_count += _active_at_timestamps(unique_ts, ep).astype(int).values
    nat_lookup_df = pd.DataFrame({"as_of_ts": unique_ts, "national_active_count": national_count.values})
    df = df.merge(nat_lookup_df, on="as_of_ts", how="left")

    neighbors_active_count = pd.Series(0.0, index=df.index)
    neighbors_total = pd.Series(0.0, index=df.index)
    for region_id, group in df.groupby("region_id"):
        region_neighbors = neighbors_map.get(region_id, [])
        neighbors_total.loc[group.index] = len(region_neighbors)
        for neighbor_id in region_neighbors:
            ep = episodes_by_region.get(neighbor_id)
            if ep is None or ep.empty:
                continue
            active = _active_at_timestamps(group["as_of_ts"], ep)
            neighbors_active_count.loc[group.index] += active.astype(int)

    df["neighbors_active_count"] = neighbors_active_count
    df["neighbors_active_frac"] = (neighbors_active_count / neighbors_total.replace(0, np.nan)).fillna(0.0)
    return df

