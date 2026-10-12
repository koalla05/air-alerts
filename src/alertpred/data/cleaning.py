import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from ..config import RAW_DATA_PATH, REGION_MAPPING
from .rollup import build_rollup_episodes_df


def _map_oblast_name(oblast_name: str) -> Tuple[str, str]:
    """Ukrainian oblast display name -> (region_id, region_en)."""
    info = REGION_MAPPING.get(oblast_name)
    if info:
        return info["id"], info["en"]
    # Fallback slugification for any unmapped name
    clean_id = oblast_name.replace(" ", "_").replace("’", "").replace("'", "").lower()
    return clean_id, oblast_name


def load_raw_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load raw alerts CSV data."""
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    
    df = pd.read_csv(
        path,
        usecols=["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"],
        low_memory=False,
    )
    return df


def clean_alerts(
    df: pd.DataFrame,
    filter_level: str = "oblast",
    min_duration_sec: int = 60,
    max_duration_hours: int = 72,
) -> pd.DataFrame:
    """
    Clean and normalize raw alerts table according to PLAN.md specifications:
    - Filter level (oblast level for macro modeling)
    - Parse timestamps to UTC
    - Drop technical glitches (< 1 min) and unbounded anomalies (> 72 hours)
    - Deduplicate and resolve overlapping intervals per region
    - Map Ukrainian oblast names to standardized region IDs and English names
    """
    if filter_level:
        df = df[df["level"] == filter_level].copy()
    else:
        df = df.copy()

    # Drop missing timestamps or missing oblast
    df = df.dropna(subset=["oblast", "started_at", "finished_at"]).copy()

    # Parse timestamps to UTC
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # Compute duration in minutes
    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0

    # Filter invalid durations (glitches < 1 min, or anomalous alerts > max_duration_hours)
    min_duration_min = min_duration_sec / 60.0
    max_duration_min = max_duration_hours * 60.0
    df = df[(df["duration_min"] >= min_duration_min) & (df["duration_min"] <= max_duration_min)].copy()

    # Map oblast to standardized keys
    mapped = [_map_oblast_name(name) for name in df["oblast"]]
    df["region_id"] = [m[0] for m in mapped]
    df["region_en"] = [m[1] for m in mapped]

    # Sort sequentially per region
    df = df.sort_values(by=["region_id", "started_at"]).reset_index(drop=True)

    # Deduplicate exact duplicate or overlapping episodes per region
    # Keep the record with the maximum duration if multiple alerts start at the exact same minute
    df = df.drop_duplicates(subset=["region_id", "started_at"], keep="last").reset_index(drop=True)

    return df


def build_oblast_rollup_episodes(
    df_raw: pd.DataFrame,
    min_duration_sec: int = 60,
    max_duration_hours: int = 72,
) -> pd.DataFrame:
    """
    Oblast-level episodes derived from *all* rows (oblast, raion, hromada -
    unlike `clean_alerts`, which only keeps native `level == 'oblast'` rows).

    From roughly Aug-Nov 2025 onward, native oblast-level rows are rare (see
    the EDA notebook) because reporting shifted to raion/hromada
    granularity - a naive oblast filter makes it look like alerts nearly
    stopped. This rolls sub-oblast alerts up to "was any part of this
    oblast under alert", merging overlapping/touching intervals per region
    via `data.rollup.build_rollup_episodes_df`, giving a continuous,
    comparable oblast-level signal across the whole timeline.

    Output schema matches `clean_alerts()`, plus `level` is `"oblast"` for
    episodes built entirely from native oblast-level rows and
    `"oblast_rollup"` for ones that include at least one derived
    raion/hromada row.
    """
    df = df_raw.dropna(subset=["oblast", "started_at", "finished_at"]).copy()
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    mapped = [_map_oblast_name(name) for name in df["oblast"]]
    df["region_id"] = [m[0] for m in mapped]
    df["region_en"] = [m[1] for m in mapped]
    df["is_native_oblast"] = df["level"] == "oblast"

    return build_rollup_episodes_df(
        df,
        min_duration_sec=min_duration_sec,
        max_duration_hours=max_duration_hours,
    )


VOLUNTEER_DATA_PATH = RAW_DATA_PATH.parent / "volunteer_data_uk.csv"

# This source uses "Київ" where official/REGION_MAPPING uses "м. Київ"; Crimea
# and Sevastopol never appear at all (per the dataset's own README, both have
# had a continuous/"permanent" siren since 2022 that was never logged as a
# discrete event in either dataset - see PLAN.md's open question on how to
# handle those two regions).
_VOLUNTEER_NAME_FIXUPS = {"Київ": "м. Київ"}


def load_and_clean_volunteer_data(
    filepath: Optional[Union[str, Path]] = None,
    drop_naive: bool = True,
    min_duration_sec: int = 60,
    max_duration_hours: float = 72.0,
) -> pd.DataFrame:
    """
    Load and clean the volunteer dataset (eTryvoga-sourced, oblast-level
    only - see https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset).

    This is a *differently-measured* signal, not a drop-in replacement for
    the official data: compared oblast-for-oblast and month-for-month
    against the official rollup, it reports many more, much shorter
    episodes (e.g. Kharkiv, Aug 2026: ~490 episodes averaging ~19 min here
    vs. ~130 averaging ~290 min in the official rollup) and the events
    don't line up in time either - a different detection methodology, not
    just a different granularity of the same underlying events. Rows are
    tagged `level='oblast_volunteer'` (never `'oblast'`/`'oblast_rollup'`)
    so this is never silently conflated with the official data downstream.

    `naive=True` rows (per the dataset's docs) have no real end time - they
    are stamped `finished_at = started_at + 30min` as a placeholder, not a
    measurement, so `drop_naive=True` (default) excludes them rather than
    feeding a fabricated constant into any duration model.
    """
    path = Path(filepath) if filepath else VOLUNTEER_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Volunteer data file not found at: {path}")

    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    if drop_naive:
        df = df[~df["naive"]].copy()

    df["region"] = df["region"].replace(_VOLUNTEER_NAME_FIXUPS)
    mapped = [_map_oblast_name(name) for name in df["region"]]
    df["region_id"] = [m[0] for m in mapped]
    df["region_en"] = [m[1] for m in mapped]

    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
    min_duration_min = min_duration_sec / 60.0
    max_duration_min = max_duration_hours * 60.0
    df = df[(df["duration_min"] >= min_duration_min) & (df["duration_min"] <= max_duration_min)].copy()

    df["oblast"] = df["region"]
    df["raion"] = None
    df["hromada"] = None
    df["level"] = "oblast_volunteer"
    df["source"] = "volunteer"

    cols = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source",
            "duration_min", "region_id", "region_en"]
    df = df[cols].sort_values(["region_id", "started_at"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["region_id", "started_at"], keep="last").reset_index(drop=True)
    return df


def build_gap_episodes(df_alerts: pd.DataFrame, max_gap_days: int = 30) -> pd.DataFrame:
    """
    Construct Task B quiet gap episodes (inter-alert intervals) per region.
    Each row represents a quiet period from the end of alert i to the start of alert i+1.
    """
    gaps = []
    max_gap_min = max_gap_days * 24 * 60.0

    for region_id, group in df_alerts.groupby("region_id"):
        group = group.sort_values("started_at").reset_index(drop=True)
        if len(group) < 2:
            continue

        prev_ends = group["finished_at"].iloc[:-1].values
        curr_starts = group["started_at"].iloc[1:].values
        prev_durations = group["duration_min"].iloc[:-1].values
        region_en = group["region_en"].iloc[0]

        gap_durations = (curr_starts - prev_ends) / np.timedelta64(1, "m")

        for i in range(len(gap_durations)):
            gap_min = gap_durations[i]
            # Discard negative intervals (overlapping alerts) and extreme outliers (> 30 days)
            if 0 < gap_min <= max_gap_min:
                gaps.append({
                    "region_id": region_id,
                    "region_en": region_en,
                    "gap_start": prev_ends[i],
                    "gap_end": curr_starts[i],
                    "gap_duration_min": gap_min,
                    "prev_alert_duration_min": prev_durations[i],
                })

    df_gaps = pd.DataFrame(gaps)
    if not df_gaps.empty:
        df_gaps["gap_start"] = pd.to_datetime(df_gaps["gap_start"], utc=True)
        df_gaps["gap_end"] = pd.to_datetime(df_gaps["gap_end"], utc=True)
        df_gaps = df_gaps.sort_values(by=["gap_start"]).reset_index(drop=True)
    return df_gaps


def create_time_split(
    df: pd.DataFrame,
    timestamp_col: str = "started_at",
    split_date: Optional[Union[str, pd.Timestamp]] = None,
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based train/test split preventing future data leakage.
    If split_date is not provided, splits based on the quantile timestamp of train_ratio.
    """
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    if split_date is None:
        cutoff = df_sorted[timestamp_col].quantile(train_ratio)
    else:
        cutoff = pd.to_datetime(split_date, utc=True)

    train_df = df_sorted[df_sorted[timestamp_col] < cutoff].copy()
    test_df = df_sorted[df_sorted[timestamp_col] >= cutoff].copy()

    return train_df, test_df

