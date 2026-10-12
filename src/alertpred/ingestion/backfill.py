"""Backfill recent alert history from alerts.in.ua to extend the (stale)
GitHub historical dataset, parsed into the same episode schema produced by
`data.cleaning.build_oblast_rollup_episodes` so it can be concatenated
directly.

Important limitation: the alerts.in.ua history endpoint only documents one
`period` value, `month_ago` (a rolling ~30-day lookback from today). It
cannot backfill further into the past than that - if the GitHub dataset's
tail is more than ~30 days old, there will still be an uncovered gap
between the two sources. The notebook EDA step reports this gap explicitly.

Like the raw GitHub CSV, alerts.in.ua is now almost entirely raion-level
(see the EDA notebook), so this rolls sub-oblast alerts up to oblast-level
episodes via `data.rollup`, the same way `data.cleaning` does for the CSV -
a naive `location_type == 'oblast'` filter would return close to nothing
for every oblast except Kyiv city.
"""
from typing import Dict, List, Optional

import pandas as pd

from ..config import REGION_MAPPING
from ..data.rollup import merge_intervals_to_episodes
from .alerts_client import AlertsApiError, fetch_region_history
from .alerts_regions import OBLAST_UID

# region_id -> (Ukrainian oblast display name, English name)
_ID_TO_META = {meta["id"]: (ukr_name, meta["en"]) for ukr_name, meta in REGION_MAPPING.items()}


def _parse_alert_objects(alert_objs: List[dict], region_id: str) -> List[dict]:
    """
    Roll up this region's alert objects (any location_type - the request is
    already scoped to this region's own uid, so no oblast-name matching is
    needed) into merged oblast-level episodes.
    """
    ukr_name, region_en = _ID_TO_META.get(region_id, (region_id, region_id))

    records = []
    for a in alert_objs:
        finished_at = a.get("finished_at")
        if not finished_at:
            continue  # still active / right-censored - not a completed episode
        started_at = pd.to_datetime(a["started_at"], utc=True)
        finished_dt = pd.to_datetime(finished_at, utc=True)
        if finished_dt <= started_at:
            continue  # defensive: drop malformed non-positive-duration records
        records.append({
            "started_at": started_at,
            "finished_at": finished_dt,
            "is_native_oblast": a.get("location_type") == "oblast",
        })

    rows = []
    for ep in merge_intervals_to_episodes(records):
        rows.append({
            "oblast": ukr_name,
            "raion": None,
            "hromada": None,
            "level": "oblast_rollup" if ep["is_rollup"] else "oblast",
            "started_at": ep["started_at"],
            "finished_at": ep["finished_at"],
            "source": "alerts_in_ua",
            "duration_min": ep["duration_min"],
            "region_id": region_id,
            "region_en": region_en,
        })
    return rows


def parse_oblast_episodes_from_payload(alert_objs: List[dict], region_id: str) -> List[dict]:
    """Public wrapper around `_parse_alert_objects`, for callers (e.g. the
    app's live-freshness check) that already hold a raw history payload
    (from `fetch_region_history`) and want the same oblast-level rollup
    `fetch_all_region_histories` produces, without re-fetching."""
    return _parse_alert_objects(alert_objs, region_id)


def fetch_all_region_histories(
    period: str = "month_ago",
    regions: Optional[Dict[str, int]] = None,
    token: Optional[str] = None,
    verbose: bool = True,
    max_duration_hours: float = 72.0,
    cache_ttl_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fetch alert history for every oblast via alerts.in.ua and return a
    DataFrame matching the schema of `data.cleaning.clean_alerts` output
    (region_id, region_en, started_at, finished_at, duration_min, ...), so
    it can be concatenated directly with the GitHub-derived episodes.

    Responses are cached on disk (see `alerts_client`), so re-running this
    is free except for regions not yet cached. The history endpoint is
    limited to 2 req/min, so a cold run over all 27 oblasts takes roughly
    13-14 minutes.

    `max_duration_hours` drops merged rollup episodes longer than this (same
    default and rationale as `data.cleaning.build_oblast_rollup_episodes`):
    a region with frequent-enough raion-level alerts and no true quiet gap
    between them can otherwise get chained by the interval merge into one
    absurd multi-week "episode" instead of being recognized as sustained
    high activity.

    `cache_ttl_sec`, if given, overrides `fetch_region_history`'s own
    default (2h) - e.g. the app uses a short TTL when freshening a single
    region's "last known alert end" for interactive display, since that
    default is tuned for bulk/training use where latency doesn't matter.
    """
    regions = regions or OBLAST_UID
    all_rows: List[dict] = []
    errors: Dict[str, str] = {}
    max_duration_min = max_duration_hours * 60.0
    history_kwargs = {"cache_ttl_sec": cache_ttl_sec} if cache_ttl_sec is not None else {}

    for i, (region_id, uid) in enumerate(regions.items(), start=1):
        try:
            payload = fetch_region_history(uid, period=period, token=token, **history_kwargs)
        except AlertsApiError as e:
            errors[region_id] = str(e)
            if verbose:
                print(f"  [{i}/{len(regions)}] {region_id} (uid={uid}): FAILED - {e}")
            continue

        rows = _parse_alert_objects(payload.get("alerts", []), region_id)
        n_dropped = sum(1 for r in rows if r["duration_min"] > max_duration_min)
        rows = [r for r in rows if r["duration_min"] <= max_duration_min]
        all_rows.extend(rows)
        if verbose:
            note = f" ({n_dropped} dropped, >{max_duration_hours:.0f}h merged chain)" if n_dropped else ""
            print(f"  [{i}/{len(regions)}] {region_id} (uid={uid}): {len(rows)} completed episodes{note}")

    if errors and verbose:
        print(f"\n{len(errors)} region(s) failed to fetch: {list(errors.keys())}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["region_id", "started_at"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["region_id", "started_at"], keep="last").reset_index(drop=True)
    return df


def merge_with_existing(df_existing: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate GitHub-derived episodes with the alerts.in.ua backfill,
    de-duplicating on (region_id, started_at) exactly like `clean_alerts`
    does, so an overlapping tail doesn't produce duplicate episodes."""
    if df_new.empty:
        return df_existing
    combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    combined = combined.sort_values(["region_id", "started_at"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["region_id", "started_at"], keep="last").reset_index(drop=True)
    return combined
