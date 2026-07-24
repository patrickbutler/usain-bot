"""Live Garmin Connect adapter, backed by the unofficial `garminconnect`
Python library. Auth comes from environment variables only (see
config.GarminCredentials.from_env) — never hardcode credentials.

Session tokens are cached to disk (GARMINTOKENS) so most invocations
don't need to re-authenticate, which matters because Garmin rate-limits
and occasionally changes endpoints. Any failure here is translated to
GarminUnavailableError so the agent can degrade gracefully instead of
crashing mid-run.
"""

from __future__ import annotations

import logging
from datetime import date

from ..config import GarminCredentials
from ..models import Activity, ActivityType
from .base import GarminAdapter, GarminUnavailableError

logger = logging.getLogger(__name__)

_METERS_PER_MILE = 1609.344

_TYPE_KEY_MAP = {
    "running": ActivityType.RUNNING,
    "trail_running": ActivityType.RUNNING,
    "track_running": ActivityType.RUNNING,
    "treadmill_running": ActivityType.RUNNING,
    "street_running": ActivityType.RUNNING,
    "cycling": ActivityType.CYCLING,
    "road_biking": ActivityType.CYCLING,
    "mountain_biking": ActivityType.CYCLING,
    "indoor_cycling": ActivityType.CYCLING,
    "strength_training": ActivityType.STRENGTH_TRAINING,
}


def _map_activity_type(type_key: str | None) -> ActivityType:
    return _TYPE_KEY_MAP.get((type_key or "").lower(), ActivityType.OTHER)


def _normalize(raw: dict) -> Activity:
    distance_m = raw.get("distance") or 0.0
    duration_s = int(raw.get("duration") or 0)
    distance_mi = distance_m / _METERS_PER_MILE
    avg_pace = (duration_s / 60.0 / distance_mi) if distance_mi > 0 else None

    start = raw.get("startTimeLocal") or raw.get("startTimeGMT") or ""
    activity_date = date.fromisoformat(start[:10]) if start else date.today()

    type_key = (raw.get("activityType") or {}).get("typeKey")

    return Activity(
        activity_id=str(raw.get("activityId")),
        date=activity_date,
        activity_type=_map_activity_type(type_key),
        distance_mi=distance_mi,
        duration_s=duration_s,
        avg_pace_min_per_mi=avg_pace,
        avg_hr=raw.get("averageHR"),
        max_hr=raw.get("maxHR"),
        elevation_gain_ft=(raw.get("elevationGain") or 0.0) * 3.28084 if raw.get("elevationGain") else None,
        name=raw.get("activityName"),
        raw=raw,
    )


class GarminConnectAdapter(GarminAdapter):
    def __init__(self, credentials: GarminCredentials):
        self._credentials = credentials
        self._client = None

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

    def fetch_activities(self, start_date: date, end_date: date) -> list[Activity]:
        client = self._ensure_client()
        try:
            raw_activities = client.get_activities_by_date(
                start_date.isoformat(), end_date.isoformat()
            )
        except Exception as exc:  # noqa: BLE001
            raise GarminUnavailableError(f"Garmin fetch failed: {exc}") from exc

        activities = []
        for raw in raw_activities:
            try:
                activities.append(_normalize(raw))
            except Exception:  # noqa: BLE001
                logger.warning("Skipping unparseable Garmin activity: %s", raw.get("activityId"))
        return activities
