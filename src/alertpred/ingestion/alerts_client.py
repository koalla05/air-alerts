"""Low-level HTTP client for the alerts.in.ua API with on-disk caching and
rate limiting, so repeated calls (e.g. Streamlit reruns, notebook
re-execution) never spam the API.

Docs: https://devs.alerts.in.ua/
Rate limits per the docs: ~8-10 req/min soft limit (12 req/min hard limit)
for most endpoints; the /regions/{uid}/alerts/{period} history endpoint is
limited to 2 req/min and the docs advise against polling it in real time.
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

# Minimum seconds between two *actual* network requests, per rate-limit bucket.
# A cache hit resolved via a 304 still counts here (it's still a request),
# but a cache hit served without even calling the API (see live.py's short
# TTL cache) does not.
_RATE_LIMIT_INTERVAL_SEC = {
    "default": 7.0,   # keeps us comfortably under the 8-10/min soft limit
    "history": 31.0,  # history endpoint: 2 req/min
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


def _load_rate_state() -> Dict[str, float]:
    if _RATE_STATE_PATH.exists():
        try:
            return json.loads(_RATE_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_rate_state(state: Dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _RATE_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def _throttle(bucket: str) -> None:
    """Block until it is safe to send another request for this rate-limit bucket."""
    state = _load_rate_state()
    last_call = state.get(bucket, 0.0)
    interval = _RATE_LIMIT_INTERVAL_SEC.get(bucket, _RATE_LIMIT_INTERVAL_SEC["default"])
    wait = interval - (time.time() - last_call)
    if wait > 0:
        time.sleep(wait)
    state[bucket] = time.time()
    _save_rate_state(state)


def request_json(
    path: str,
    *,
    bucket: str = "default",
    params: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    use_cache: bool = True,
) -> Any:
    """
    GET a JSON endpoint from alerts.in.ua with disk caching and rate limiting.

    Every response is cached on disk keyed by URL+params. On a cache hit we
    send `If-Modified-Since` (the caching approach the docs recommend for
    all endpoints); a 304 reuses the cached body. Set `use_cache=False` to
    always send a plain GET (still rate-limited, just without the
    conditional header / cache write).
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


def fetch_region_history(uid: int, period: str = "month_ago", token: Optional[str] = None) -> Dict[str, Any]:
    """GET /v1/regions/{uid}/alerts/{period}.json -> {"alerts": [...]}.

    Only `month_ago` is documented (a rolling 30-day lookback from today).
    History responses are cached indefinitely per (uid, period) call - call
    `clear_history_cache` or delete the cache dir to force a refresh.
    """
    return request_json(f"/v1/regions/{uid}/alerts/{period}.json", bucket="history", token=token)


def fetch_oblast_status_string(token: Optional[str] = None) -> str:
    """GET /v1/iot/active_air_raid_alerts_by_oblast.json -> 27-char status string."""
    return request_json("/v1/iot/active_air_raid_alerts_by_oblast.json", bucket="default", token=token)
