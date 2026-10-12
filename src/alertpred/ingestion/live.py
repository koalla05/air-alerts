import datetime
import time
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .alerts_client import fetch_active_alerts, get_api_token
from ..config import ISO_MAPPING, REGION_MAPPING
from ..geo.raions import API_NAME_TO_RAION_ID, RAION_BY_ID, RAIONS_BY_OBLAST

# Re-exported for backward compatibility (alertpred.ingestion.live.ISO_MAPPING)
__all__ = ["ISO_MAPPING", "get_live_alerts_status", "LIVE_CACHE_TTL_SEC"]

_UA_NAME_TO_REGION_ID: Dict[str, str] = {ukr_name: meta["id"] for ukr_name, meta in REGION_MAPPING.items()}

# PLAN.md guidance: cache the live API response for 15-30s to respect rate
# limits and keep the UI snappy. This is an in-process TTL cache of the raw
# "who is currently active, since when" snapshot; elapsed times are always
# recomputed against the current time on every call, cache hit or not.
LIVE_CACHE_TTL_SEC = 20
_active_cache: Dict[str, Any] = {"region_starts": None, "raion_starts": None, "fetched_at": 0.0}


def _kyiv_now() -> tuple:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return now_utc, now_utc.astimezone(datetime.timezone(datetime.timedelta(hours=3)))


def _parse_active(payload: Dict[str, Any]) -> tuple:
    """
    Extract {region_id: started_at} and {raion_id: started_at} for currently
    active alerts, from /v1/alerts/active.json.

    Matching is done on the `location_oblast`/`location_title` *name* fields,
    not the uid fields - inspecting real API responses showed
    `location_oblast_uid` does not reliably reference the parent oblast's
    uid (it appears to just echo the alert's own location uid).

    Granularity has shifted almost entirely from oblast-wide to raion-level
    alerts since late 2025 (see the EDA notebook), so both directions are
    cascaded: a raion-level alert also marks its parent oblast active, and an
    oblast-wide (blanket) alert marks every raion within it active. hromada-
    and city-level alerts are not yet mapped to a raion (no hromada->raion
    crosswalk is built) and are currently ignored here.
    """
    region_starts: Dict[str, datetime.datetime] = {}
    raion_starts: Dict[str, datetime.datetime] = {}

    def _mark(d: Dict[str, datetime.datetime], key: str, started_at: datetime.datetime) -> None:
        if key not in d or started_at < d[key]:
            d[key] = started_at

    for alert in payload.get("alerts", []):
        ltype = alert.get("location_type")
        if ltype not in ("oblast", "raion"):
            continue
        started_at = pd.to_datetime(alert["started_at"], utc=True).to_pydatetime()

        if ltype == "oblast":
            region_id = _UA_NAME_TO_REGION_ID.get(alert.get("location_oblast") or alert.get("location_title"))
            if region_id is None:
                continue
            _mark(region_starts, region_id, started_at)
            for raion_id in RAIONS_BY_OBLAST.get(region_id, []):
                _mark(raion_starts, raion_id, started_at)
        else:  # raion
            raion_id = API_NAME_TO_RAION_ID.get(alert.get("location_title"))
            if raion_id is None:
                continue
            _mark(raion_starts, raion_id, started_at)
            region_id = RAION_BY_ID[raion_id]["oblast_id"]
            _mark(region_starts, region_id, started_at)

    return region_starts, raion_starts


def _status_for(
    key: str,
    starts: Dict[str, datetime.datetime],
    now_utc: datetime.datetime,
    last_end: Optional[Mapping[str, Any]],
    default_quiet_min: float,
) -> Dict[str, float]:
    start = starts.get(key)
    if start is not None:
        elapsed_min = max(0.0, (now_utc - start).total_seconds() / 60.0)
        return {"is_active": True, "elapsed_min": elapsed_min, "elapsed_quiet_min": 0.0}

    quiet_min = default_quiet_min
    if last_end is not None:
        last = last_end.get(key)
        if last is not None and pd.notna(last):
            last_dt = pd.Timestamp(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.tz_localize("UTC")
            quiet_min = max(0.0, (now_utc - last_dt.to_pydatetime()).total_seconds() / 60.0)
    return {"is_active": False, "elapsed_min": 0.0, "elapsed_quiet_min": quiet_min}


def _build_statuses(
    region_starts: Dict[str, datetime.datetime],
    raion_starts: Dict[str, datetime.datetime],
    now_utc: datetime.datetime,
    last_alert_end_by_region: Optional[Mapping[str, Any]],
) -> tuple:
    regions = {
        region_id: _status_for(region_id, region_starts, now_utc, last_alert_end_by_region,
                                240.0 if "kyiv" in region_id else 480.0)
        for region_id in ISO_MAPPING.keys()
    }
    # No raion-level historical data yet (see PLAN.md discussion on raion
    # granularity), so quiet time for raions falls back to a flat placeholder.
    raions = {
        raion_id: _status_for(raion_id, raion_starts, now_utc, None, 480.0)
        for raion_id in RAION_BY_ID.keys()
    }
    return regions, raions


def _demo_snapshot(now_kyiv: datetime.datetime) -> Dict[str, Any]:
    """Realistic demo snapshot used when no API token is configured or the API call fails."""
    demo_active_regions = {
        "kharkiv": 42.0,       # 42 minutes into alert
        "donetsk": 75.0,       # 75 minutes into alert
        "zaporizhzhia": 30.0,  # 30 minutes into alert
        "sumy": 15.0,          # 15 minutes into alert
        "dnipropetrovsk": 20.0,  # 20 minutes into alert
    }

    regions = {}
    for reg_id in ISO_MAPPING.keys():
        if reg_id in demo_active_regions:
            regions[reg_id] = {"is_active": True, "elapsed_min": demo_active_regions[reg_id], "elapsed_quiet_min": 0.0}
        else:
            quiet_min = 240.0 if "kyiv" in reg_id else 480.0
            regions[reg_id] = {"is_active": False, "elapsed_min": 0.0, "elapsed_quiet_min": quiet_min}

    # Mirror the demo onto a representative raion or two per active oblast so
    # the raion map isn't empty in demo mode.
    raions = {}
    for raion_id, meta in RAION_BY_ID.items():
        if meta["oblast_id"] in demo_active_regions and not meta["is_pseudo_oblast"]:
            is_first = RAIONS_BY_OBLAST.get(meta["oblast_id"], [None])[0] == raion_id
            if is_first:
                raions[raion_id] = {
                    "is_active": True,
                    "elapsed_min": demo_active_regions[meta["oblast_id"]],
                    "elapsed_quiet_min": 0.0,
                }
                continue
        elif meta["oblast_id"] in demo_active_regions and meta["is_pseudo_oblast"]:
            raions[raion_id] = {"is_active": True, "elapsed_min": demo_active_regions[meta["oblast_id"]], "elapsed_quiet_min": 0.0}
            continue
        raions[raion_id] = {"is_active": False, "elapsed_min": 0.0, "elapsed_quiet_min": 480.0}

    return {
        "is_live_api": False,
        "last_updated": now_kyiv.strftime("%Y-%m-%d %H:%M:%S (Kyiv Time)"),
        "total_active_regions": len(demo_active_regions),
        "regions": regions,
        "raions": raions,
    }


def get_live_alerts_status(
    api_token: Optional[str] = None,
    last_alert_end_by_region: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetch current active alerts at both oblast (`regions`) and raion
    (`raions`) granularity.

    If a token is available (explicit arg, ALERTS_IN_UA_TOKEN env var, or
    .env file), queries alerts.in.ua's `/v1/alerts/active.json`. The "who is
    active since when" snapshot is cached in-process for `LIVE_CACHE_TTL_SEC`
    seconds so frequent callers (e.g. Streamlit reruns) don't spam the API;
    elapsed minutes are still recomputed against the current time on every
    call.

    Status is cascaded both ways between granularities: a raion-level alert
    also marks its parent oblast active (oblast-wide alerts are now rare -
    see the EDA notebook), and an oblast-wide alert marks every raion in it
    active.

    `last_alert_end_by_region` optionally maps region_id -> last known
    `finished_at` (from historical/backfilled episodes), used to compute a
    real elapsed-quiet time for oblasts with no active alert. No equivalent
    history exists yet at raion granularity, so raion quiet time is a rough
    placeholder.

    Falls back to a realistic demo snapshot if no token is configured or the
    API call fails.
    """
    now_utc, now_kyiv = _kyiv_now()

    cache_fresh = time.time() - _active_cache["fetched_at"] < LIVE_CACHE_TTL_SEC
    if cache_fresh and _active_cache["region_starts"] is not None:
        regions, raions = _build_statuses(
            _active_cache["region_starts"], _active_cache["raion_starts"], now_utc, last_alert_end_by_region
        )
        return {
            "is_live_api": True,
            "last_updated": now_kyiv.strftime("%Y-%m-%d %H:%M:%S (Kyiv Time)"),
            "total_active_regions": sum(1 for s in regions.values() if s["is_active"]),
            "regions": regions,
            "raions": raions,
        }

    token = get_api_token(api_token)
    if token:
        try:
            payload = fetch_active_alerts(token=token)
            region_starts, raion_starts = _parse_active(payload)
            _active_cache["region_starts"] = region_starts
            _active_cache["raion_starts"] = raion_starts
            _active_cache["fetched_at"] = time.time()
            regions, raions = _build_statuses(region_starts, raion_starts, now_utc, last_alert_end_by_region)
            return {
                "is_live_api": True,
                "last_updated": now_kyiv.strftime("%Y-%m-%d %H:%M:%S (Kyiv Time)"),
                "total_active_regions": sum(1 for s in regions.values() if s["is_active"]),
                "regions": regions,
                "raions": raions,
            }
        except Exception:
            pass  # network error, bad payload, or AlertsApiError - fall back to demo snapshot

    return _demo_snapshot(now_kyiv)
