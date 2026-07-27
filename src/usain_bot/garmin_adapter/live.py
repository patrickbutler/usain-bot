"""Live Garmin Connect adapter, backed by the unofficial `garminconnect`
Python library. Auth comes from environment variables only (see
config.GarminCredentials.from_env) — never hardcode credentials.

Session tokens are cached to disk (GARMINTOKENS) so most invocations
don't need to re-authenticate, which matters because Garmin rate-limits
and occasionally changes endpoints.

Scope: per product decision, only running activities are pulled —
activityType.typeKey in {running, trail_running, treadmill_running}
("treadmill" is also accepted in case Garmin's key varies). Everything
else (cycling, strength, ...) is excluded at this boundary.

Rate limiting: every fetch retries with exponential backoff. A response
that looks like HTTP 429 (rate limited) gets a much longer backoff
schedule than a generic transient failure, since Garmin's limiter needs
real cool-down time — this is what the initial full-history backfill
relies on. Any terminal failure is translated to GarminUnavailableError
so the agent can degrade gracefully instead of crashing mid-run.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

from ..config import GarminCredentials
from ..models import Activity, ActivityType
from .base import GarminAdapter, GarminUnavailableError

logger = logging.getLogger(__name__)

_METERS_PER_MILE = 1609.344

# Only these activityType.typeKey values are running activities we keep.
INCLUDED_TYPE_KEYS = {"running", "trail_running", "treadmill_running", "treadmill"}

# Backoff schedules (seconds). Rate-limit (429) responses need long
# cool-downs; other transient failures retry quickly.
RATE_LIMIT_BACKOFF_S = (30.0, 60.0, 120.0, 240.0)
TRANSIENT_BACKOFF_S = (2.0, 4.0, 8.0, 16.0)


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _parse_start_time(raw: dict) -> datetime | None:
    start = raw.get("startTimeLocal") or raw.get("startTimeGMT")
    if not start:
        return None
    try:
        return datetime.fromisoformat(start.replace("T", " ").strip())
    except ValueError:
        return None


def _normalize(raw: dict) -> Activity:
    distance_m = raw.get("distance") or 0.0
    duration_s = int(raw.get("duration") or 0)
    distance_mi = distance_m / _METERS_PER_MILE
    avg_pace = (duration_s / 60.0 / distance_mi) if distance_mi > 0 else None

    start_time = _parse_start_time(raw)
    activity_date = start_time.date() if start_time else date.today()

    return Activity(
        activity_id=str(raw.get("activityId")),
        date=activity_date,
        activity_type=ActivityType.RUNNING,
        distance_mi=distance_mi,
        duration_s=duration_s,
        avg_pace_min_per_mi=avg_pace,
        avg_hr=raw.get("averageHR"),
        max_hr=raw.get("maxHR"),
        elevation_gain_ft=(raw.get("elevationGain") or 0.0) * 3.28084 if raw.get("elevationGain") else None,
        name=raw.get("activityName"),
        start_time=start_time,
        raw=raw,
    )


class GarminConnectAdapter(GarminAdapter):
    def __init__(self, credentials: GarminCredentials, sleep_fn=time.sleep):
        self._credentials = credentials
        self._client = None
        self._sleep = sleep_fn  # injectable for tests

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        try:
            from garminconnect import Garmin
        except ImportError as exc:  # pragma: no cover - dependency wiring
            raise GarminUnavailableError(
                "garminconnect is not installed. `pip install garminconnect`."
            ) from exc

        client = Garmin(self._credentials.email, self._credentials.password)
        try:
            try:
                client.login(self._credentials.token_store)
            except Exception:  # noqa: BLE001 - cached token missing/expired, fall back
                client.login()
                if hasattr(client, "garth"):
                    client.garth.dump(self._credentials.token_store)
        except Exception as exc:  # noqa: BLE001
            raise GarminUnavailableError(f"Garmin login failed: {exc}") from exc

        self._client = client
        return client

    def _fetch_with_retries(self, start_date: date, end_date: date) -> list[dict]:
        client = self._ensure_client()
        last_exc: Exception | None = None
        attempt = 0
        max_attempts = max(len(RATE_LIMIT_BACKOFF_S), len(TRANSIENT_BACKOFF_S)) + 1
        while attempt < max_attempts:
            try:
                return client.get_activities_by_date(start_date.isoformat(), end_date.isoformat())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                schedule = RATE_LIMIT_BACKOFF_S if _is_rate_limited(exc) else TRANSIENT_BACKOFF_S
                if attempt >= len(schedule):
                    break
                wait = schedule[attempt]
                logger.warning(
                    "Garmin fetch failed (%s), attempt %d — backing off %.0fs: %s",
                    "rate-limited" if _is_rate_limited(exc) else "transient", attempt + 1, wait, exc,
                )
                self._sleep(wait)
                attempt += 1
        raise GarminUnavailableError(f"Garmin fetch failed after retries: {last_exc}") from last_exc

    def fetch_activities(self, start_date: date, end_date: date) -> list[Activity]:
        raw_activities = self._fetch_with_retries(start_date, end_date)

        activities = []
        for raw in raw_activities:
            type_key = ((raw.get("activityType") or {}).get("typeKey") or "").lower()
            if type_key not in INCLUDED_TYPE_KEYS:
                continue
            try:
                activities.append(_normalize(raw))
            except Exception:  # noqa: BLE001
                logger.warning("Skipping unparseable Garmin activity: %s", raw.get("activityId"))
        return activities
