"""Backfill recent alert history from alerts.in.ua to extend the (stale)
GitHub historical dataset, parsed into the same episode schema produced by
`data.cleaning.clean_alerts` so it can be concatenated directly.

Important limitation: the alerts.in.ua history endpoint only documents one
`period` value, `month_ago` (a rolling ~30-day lookback from today). It
cannot backfill further into the past than that - if the GitHub dataset's
tail is more than ~30 days old, there will still be an uncovered gap
between the two sources. The notebook EDA step reports this gap explicitly.
"""
from typing import Dict, List, Optional

import pandas as pd

from ..config import REGION_MAPPING
from .alerts_client import AlertsApiError, fetch_region_history
from .alerts_regions import OBLAST_UID

# region_id -> (Ukrainian oblast display name, English name)
_ID_TO_META = {meta["id"]: (ukr_name, meta["en"]) for ukr_name, meta in REGION_MAPPING.items()}


def _parse_alert_objects(alert_objs: List[dict], region_id: str, uid: int) -> List[dict]:
    """Keep only completed, oblast-level episodes belonging to this region's own uid."""
    ukr_name, region_en = _ID_TO_META.get(region_id, (region_id, region_id))
    rows = []
    for a in alert_objs:
        if a.get("location_type") != "oblast":
            continue
        owner_uid = a.get("location_oblast_uid") or a.get("location_uid")
        if owner_uid is not None and str(owner_uid) != str(uid):
            continue
        finished_at = a.get("finished_at")
        if not finished_at:
            continue  # still active / right-censored - not a completed episode
        started_at = pd.to_datetime(a["started_at"], utc=True)
        finished_dt = pd.to_datetime(finished_at, utc=True)
        duration_min = (finished_dt - started_at).total_seconds() / 60.0
        rows.append({
            "oblast": ukr_name,
            "raion": None,
            "hromada": None,
            "level": "oblast",
            "started_at": started_at,
            "finished_at": finished_dt,
            "source": "alerts_in_ua",
            "duration_min": duration_min,
            "region_id": region_id,
            "region_en": region_en,
        })
    return rows


def fetch_all_region_histories(
    period: str = "month_ago",
    regions: Optional[Dict[str, int]] = None,
    token: Optional[str] = None,
    verbose: bool = True,
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
    """
    regions = regions or OBLAST_UID
    all_rows: List[dict] = []
    errors: Dict[str, str] = {}

    for i, (region_id, uid) in enumerate(regions.items(), start=1):
        try:
            payload = fetch_region_history(uid, period=period, token=token)
        except AlertsApiError as e:
            errors[region_id] = str(e)
            if verbose:
                print(f"  [{i}/{len(regions)}] {region_id} (uid={uid}): FAILED - {e}")
            continue

        rows = _parse_alert_objects(payload.get("alerts", []), region_id, uid)
        all_rows.extend(rows)
        if verbose:
            print(f"  [{i}/{len(regions)}] {region_id} (uid={uid}): {len(rows)} completed episodes")

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
