"""Shared "next 7 days" projection, used by both the CLI's rolling forward
view and the web UI's Upcoming tab. Coarse by design (see planner.py's
module docstring) — actual distances for anything but today are decided
live on the day, not baked in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .models import Recommendation


@dataclass(frozen=True)
class DayProjection:
    date: date
    weekday: str
    run_type: str
    distance_mi: Optional[float]
    note: str

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "weekday": self.weekday, "run_type": self.run_type,
            "distance_mi": self.distance_mi, "note": self.note,
        }


def run_days_for_week(run_days_per_week: int) -> list[int]:
    """Evenly spaced weekday indices (0=Mon..6=Sun) for the given
    days/week, biased away from Monday so a 3-day/week pattern lands on
    something like Tue/Thu/Sat rather than Mon/Wed/Fri."""
    run_days_per_week = max(1, min(run_days_per_week, 7))
    return sorted({min(round(i * 7 / run_days_per_week) + 1, 6) for i in range(run_days_per_week)})


def project_next_7_days(
    as_of: date, run_days_per_week: int, quality_target: int, is_backoff: bool,
    recommendation: Recommendation,
) -> list[DayProjection]:
    run_days = run_days_for_week(run_days_per_week)
    long_day = max(run_days)
    quality_day = min(run_days) if quality_target > 0 and not is_backoff else None

    out = []
    for offset in range(7):
        d = as_of + timedelta(days=offset)
        weekday = d.weekday()
        if offset == 0:
            out.append(DayProjection(d, d.strftime("%A"), recommendation.run_type,
                                      recommendation.target_distance_mi, "today's recommendation"))
        elif weekday not in run_days:
            out.append(DayProjection(d, d.strftime("%A"), "rest_or_cross_training", None, ""))
        elif weekday == long_day:
            out.append(DayProjection(d, d.strftime("%A"), "long", None,
                                      "projected — actual distance decided live on that day"))
        elif weekday == quality_day:
            out.append(DayProjection(d, d.strftime("%A"), "quality", None, "projected"))
        else:
            out.append(DayProjection(d, d.strftime("%A"), "easy", None, "projected"))
    return out
