"""Mock adapter for tests and offline dry-runs. Reads normalized
activities from a JSON fixture instead of hitting the network, so the
rest of the system (classification, guardrails, planner, agent) is
fully testable without Garmin credentials.

Applies the same scope rule as the live adapter: only running
activities are returned (per product decision, cycling/strength/etc.
are excluded), so fixture files may contain non-running rows to prove
the filter works.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ..models import Activity, ActivityType
from .base import GarminAdapter


def _parse_start_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _activity_from_dict(d: dict) -> Activity:
    return Activity(
        activity_id=d["activity_id"],
        date=date.fromisoformat(d["date"]),
        activity_type=ActivityType(d["activity_type"]),
        distance_mi=d["distance_mi"],
        duration_s=d["duration_s"],
        avg_pace_min_per_mi=d.get("avg_pace_min_per_mi"),
        avg_hr=d.get("avg_hr"),
        max_hr=d.get("max_hr"),
        elevation_gain_ft=d.get("elevation_gain_ft"),
        name=d.get("name"),
        start_time=_parse_start_time(d.get("start_time")),
        raw=d.get("raw", {}),
    )


class MockGarminAdapter(GarminAdapter):
    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)
        with self.fixture_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._activities = [_activity_from_dict(d) for d in raw]

    @classmethod
    def from_activities(cls, activities: list[Activity]) -> "MockGarminAdapter":
        instance = object.__new__(cls)
        instance.fixture_path = None
        instance._activities = activities
        return instance

    def fetch_activities(self, start_date: date, end_date: date) -> list[Activity]:
        return [
            a for a in self._activities
            if start_date <= a.date <= end_date and a.activity_type == ActivityType.RUNNING
        ]
