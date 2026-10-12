"""Low-level HTTP client for the alerts.in.ua API with on-disk caching and
rate limiting, so repeated calls (e.g. Streamlit reruns, notebook
re-execution) never spam the API.

Docs: https://devs.alerts.in.ua/
Rate limits per the docs: ~8-10 req/min soft limit (12 req/min hard limit)
for most endpoints; the docs describe the /regions/{uid}/alerts/{period}
history endpoint as limited to 2 req/min and advise against polling it in
real time. The server's actual enforcement, confirmed from a live 429
response, is a rolling 30 calls / 10 minutes - a plain per-call minimum
spacing isn't enough to respect that across multiple separate process runs
in the same window, so the limiter tracks a persisted sliding-window call
count per bucket, not just the time of the last call.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..config import DATA_DIR, PROJECT_ROOT

API_BASE_URL = "https://api.alerts.in.ua"
CACHE_DIR = DATA_DIR / "cache" / "alerts_in_ua"
_RATE_STATE_PATH = CACHE_DIR / "_rate_state.json"

# Sliding-window call budget per rate-limit bucket: at most `max_calls`
# *actual* network requests within `window_sec`. A cache hit resolved via a
# 304 still counts here (it's still a request to the server); a cache hit
# served without even calling the API (a TTL hit here, or live.py's short
# TTL cache for live status) does not.
_RATE_LIMIT_WINDOW_SEC = {
    "default": 60.0,
    "history": 600.0,  # observed server enforcement: 30 calls / 10 min
}
_RATE_LIMIT_MAX_CALLS = {
    "default": 8,    # keeps us comfortably under the docs' 8-10/min soft limit
    "history": 25,   # margin below the observed 30/10min hard limit
}

_dotenv_loaded = False


class AlertsApiError(RuntimeError):
    """Raised for any non-2xx/304 response from the alerts.in.ua API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"alerts.in.ua API error {status_code}: {message}")


def get_api_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolve the API token from an explicit arg, the environment, or `.env`."""
    if explicit_token:
        return explicit_token
    token = os.environ.get("ALERTS_IN_UA_TOKEN")
    if token:
        return token

    global _dotenv_loaded
    if not _dotenv_loaded:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        _dotenv_loaded = True
    return os.environ.get("ALERTS_IN_UA_TOKEN")


def _cache_file_for(cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.json"


def _load_rate_state() -> Dict[str, list]:
    if _RATE_STATE_PATH.exists():
        try:
            return json.loads(_RATE_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_rate_state(state: Dict[str, list]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _RATE_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def _throttle(bucket: str) -> None:
    """Block until the sliding-window call budget for this bucket has room,
    then record this call. State is persisted to disk so the budget is
    shared across separate process runs (notebook kernels, Streamlit, ad-hoc
    scripts) hitting the same bucket in the same window."""
    window = _RATE_LIMIT_WINDOW_SEC.get(bucket, _RATE_LIMIT_WINDOW_SEC["default"])
    max_calls = _RATE_LIMIT_MAX_CALLS.get(bucket, _RATE_LIMIT_MAX_CALLS["default"])

    now = time.time()
    state = _load_rate_state()
    calls = [t for t in state.get(bucket, []) if now - t < window]

    if len(calls) >= max_calls:
        wait = window - (now - calls[0]) + 0.5
        if wait > 0:
            time.sleep(wait)
        now = time.time()
        calls = [t for t in calls if now - t < window]

    calls.append(now)
    state[bucket] = calls
    _save_rate_state(state)


def request_json(
    path: str,
    *,
    bucket: str = "default",
    params: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    use_cache: bool = True,
    cache_ttl_sec: Optional[float] = None,
) -> Any:
    """
    GET a JSON endpoint from alerts.in.ua with disk caching and rate limiting.

    Every response is cached on disk keyed by URL+params. On a cache hit we
    send `If-Modified-Since` (the caching approach the docs recommend for
    all endpoints); a 304 reuses the cached body. Set `use_cache=False` to
    always send a plain GET (still rate-limited, just without the
    conditional header / cache write).

    `cache_ttl_sec`, if set, skips the network call (and the rate-limit
    wait) entirely when the cached entry is younger than this - without it,
    even a guaranteed-304 response still pays the full per-request rate
    limit wait on every call, which makes re-running something like a full
    27-region history backfill shortly after a previous run slow for no
    reason (see notebook 01, section 2.2).
    """
    resolved_token = get_api_token(token)
    if not resolved_token:
        raise AlertsApiError(401, "No ALERTS_IN_UA_TOKEN found (env var or .env file)")

    url = f"{API_BASE_URL}{path}"
    cache_file = _cache_file_for(url + json.dumps(params or {}, sort_keys=True))

    cached: Optional[Dict[str, Any]] = None
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None

    if cached is not None and cache_ttl_sec is not None:
        age = time.time() - cached.get("fetched_at", 0)
        if age < cache_ttl_sec:
            return cached["body"]

    headers = {"Authorization": f"Bearer {resolved_token}"}
    if cached and cached.get("last_modified"):
        headers["If-Modified-Since"] = cached["last_modified"]

    _throttle(bucket)
    resp = requests.get(url, headers=headers, params=params, timeout=15)

    if resp.status_code == 304 and cached is not None:
        return cached["body"]

    if resp.status_code == 200:
        body = resp.json()
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    {
                        "url": url,
                        "last_modified": resp.headers.get("Last-Modified"),
                        "fetched_at": time.time(),
                        "body": body,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return body

    try:
        message = resp.json().get("message", resp.text)
    except ValueError:
        message = resp.text
    raise AlertsApiError(resp.status_code, message)


def fetch_active_alerts(token: Optional[str] = None) -> Dict[str, Any]:
    """GET /v1/alerts/active.json -> {"alerts": [...]}."""
    return request_json("/v1/alerts/active.json", bucket="default", token=token)


def fetch_region_history(
    uid: int,
    period: str = "month_ago",
    token: Optional[str] = None,
    cache_ttl_sec: float = 7200.0,
) -> Dict[str, Any]:
    """GET /v1/regions/{uid}/alerts/{period}.json -> {"alerts": [...]}.

    Only `month_ago` is documented (a rolling 30-day lookback from today),
    a slowly-changing window - the 30-calls/10-min server limit (see
    AlertsApiError 429 handling in the caller) barely covers one full
    27-region pass, so responses younger than `cache_ttl_sec` (default 2h)
    are served straight from disk with no network call at all. Delete the
    file under `data/cache/alerts_in_ua/` (or pass `cache_ttl_sec=0`) to
    force a refresh sooner.
    """
    return request_json(
        f"/v1/regions/{uid}/alerts/{period}.json", bucket="history", token=token, cache_ttl_sec=cache_ttl_sec
    )


def fetch_oblast_status_string(token: Optional[str] = None) -> str:
    """GET /v1/iot/active_air_raid_alerts_by_oblast.json -> 27-char status string."""
    return request_json("/v1/iot/active_air_raid_alerts_by_oblast.json", bucket="default", token=token)
