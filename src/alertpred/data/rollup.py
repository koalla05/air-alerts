"""Shared "roll sub-oblast alerts up to oblast-level episodes" logic, used by
both `data.cleaning` (for the raw GitHub CSV, which has raion/hromada-level
rows alongside oblast-level ones) and `ingestion.backfill` (for alerts.in.ua,
which is now almost entirely raion-level - see the EDA notebook).

Why this exists: from roughly Aug-Nov 2025 onward, Ukraine's official alert
reporting shifted from oblast-wide blanket alerts to localized raion/hromada
alerts. A naive `level == 'oblast'` filter (what `clean_alerts` did before)
makes oblast-level alert *counts* collapse to near zero after that point,
which looks like alerts stopped - they didn't, they just started being
reported at finer granularity. Rolling sub-oblast alerts up to "was any part
of this oblast under alert" restores a continuous, comparable oblast-level
signal across the whole timeline.
"""
from typing import Any, Dict, List

import pandas as pd


def merge_intervals_to_episodes(
    records: List[Dict[str, Any]],
    max_single_record_hours: float = 24.0,
) -> List[Dict[str, Any]]:
    """
    Merge overlapping/touching (started_at, finished_at) intervals for a
    single region into oblast-level episodes.

    `records`: list of {"started_at": Timestamp, "finished_at": Timestamp,
    "is_native_oblast": bool} - already filtered to one region, with any
    still-open (finished_at is NaT/None) records already excluded by the
    caller (a currently active alert isn't a completed historical episode).

    One corrupt record with an absurd duration (seen in practice: a
    hromada-level row the source never closed out, spanning *months*) would
    otherwise chain-merge everything between its start and end into one
    giant episode - silently destroying up to years of otherwise-valid data
    for that stretch, since the only other safety net is an upper bound on
    the *merged output*, by which point the damage is already done. So
    `max_single_record_hours` drops individual input records this long
    *before* merging - a real single siren essentially never runs unbroken
    that long (even the longest genuine merged episode seen in this data,
    built from many real back-to-back sub-oblast alerts, is ~43h).

    Returns episodes as {"started_at", "finished_at", "duration_min",
    "is_rollup"} sorted by start. `is_rollup` is True if the merged episode
    includes at least one sub-oblast (raion/hromada/city) record - i.e. it
    is a derived signal, not a native oblast-wide alert.
    """
    if not records:
        return []

    max_single_record_min = max_single_record_hours * 60.0
    records = [
        r for r in records
        if (r["finished_at"] - r["started_at"]).total_seconds() / 60.0 <= max_single_record_min
    ]
    if not records:
        return []

    ordered = sorted(records, key=lambda r: r["started_at"])
    merged: List[Dict[str, Any]] = []
    for rec in ordered:
        start, end = rec["started_at"], rec["finished_at"]
        is_native = rec["is_native_oblast"]
        if merged and start <= merged[-1]["finished_at"]:
            if end > merged[-1]["finished_at"]:
                merged[-1]["finished_at"] = end
            if not is_native:
                merged[-1]["is_rollup"] = True
        else:
            merged.append({"started_at": start, "finished_at": end, "is_rollup": not is_native})

    for ep in merged:
        ep["duration_min"] = (ep["finished_at"] - ep["started_at"]).total_seconds() / 60.0
    return merged


def build_rollup_episodes_df(
    df: pd.DataFrame,
    region_id_col: str = "region_id",
    oblast_col: str = "oblast",
    region_en_col: str = "region_en",
    source_col: str = "source",
    min_duration_sec: int = 60,
    max_duration_hours: int = 72,
    max_single_record_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Group already-parsed, already-region-mapped alert rows (any granularity)
    by region and merge them into oblast-level episodes via
    `merge_intervals_to_episodes` (see that function for why
    `max_single_record_hours` matters - it guards against one corrupt
    never-closed row silently swallowing a huge stretch of real data).

    Expects `df` to already have UTC `started_at`/`finished_at` (no nulls -
    still-open/censored rows must be dropped by the caller first), plus
    `region_id_col`, `oblast_col`, `region_en_col`, `source_col`, and a
    boolean `is_native_oblast` column.
    """
    rows: List[Dict[str, Any]] = []
    for region_id, group in df.groupby(region_id_col):
        records = [
            {"started_at": r.started_at, "finished_at": r.finished_at, "is_native_oblast": r.is_native_oblast}
            for r in group.itertuples()
        ]
        episodes = merge_intervals_to_episodes(records, max_single_record_hours=max_single_record_hours)
        oblast_name = group[oblast_col].iloc[0]
        region_en = group[region_en_col].iloc[0]
        sources = sorted(group[source_col].unique())
        source_tag = sources[0] if len(sources) == 1 else "+".join(sources)

        for ep in episodes:
            rows.append({
                "oblast": oblast_name,
                "raion": None,
                "hromada": None,
                "level": "oblast_rollup" if ep["is_rollup"] else "oblast",
                "started_at": ep["started_at"],
                "finished_at": ep["finished_at"],
                "source": source_tag,
                "duration_min": ep["duration_min"],
                "region_id": region_id,
                "region_en": region_en,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    min_duration_min = min_duration_sec / 60.0
    max_duration_min = max_duration_hours * 60.0
    out = out[(out["duration_min"] >= min_duration_min) & (out["duration_min"] <= max_duration_min)].copy()
    out = out.sort_values(["region_id", "started_at"]).reset_index(drop=True)
    return out
