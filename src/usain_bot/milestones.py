"""Upcoming running milestones and whether each one has a set date.

This is the single place that answers "what am I training for, when is it,
and how far do I need to get before then". Both the planner and the coach
read it, so they can never disagree about a date.

The distinction that matters is **dated vs undated**, because it changes
the shape of the plan entirely (see `rules.py`):

* **Dated** — the calendar is fixed, so the ramp is smoothed to arrive on
  time and the weeks in between are spent maintaining rather than
  climbing. Peaking eight weeks early just means eight weeks of avoidable
  high-mileage exposure.
* **Undated** — nothing to smooth toward, so build conservatively to the
  milestone's peak long run, then taper into it. The date is an *output*
  of readiness rather than an input to the plan.

Peak long runs are the standard preparation distances, not the race
distances: you don't run a marathon in training. The 50K's 24 mi peak is
higher than the marathon's 22 because ultra prep leans on
back-to-back long runs and time on feet rather than a single longest run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .config import Config

# Preparation targets per milestone: the longest single training run that
# should be reached before tapering, and how long that taper runs.
MILESTONE_PREP = {
    "half_marathon": {"peak_long_run_mi": 12.0, "taper_weeks": 1},
    "marathon": {"peak_long_run_mi": 22.0, "taper_weeks": 2},
    "ultra_50k": {"peak_long_run_mi": 24.0, "taper_weeks": 2},
}

# Config goal names vary ("half_marathon_benchmark"); map them onto the
# prep table above by the distance they represent.
_DISTANCE_TO_KIND = [(13.1, "half_marathon"), (26.2, "marathon"), (31.1, "ultra_50k")]
_DISTANCE_TOLERANCE_MI = 1.5

DEFAULT_PEAK_LONG_RUN_MI = 22.0
DEFAULT_TAPER_WEEKS = 2


def milestone_kind(distance_mi: float) -> Optional[str]:
    """Which prep profile a goal distance corresponds to, if any."""
    for dist, kind in _DISTANCE_TO_KIND:
        if abs(distance_mi - dist) <= _DISTANCE_TOLERANCE_MI:
            return kind
    return None


@dataclass(frozen=True)
class Milestone:
    """One thing the athlete is training toward."""

    name: str
    distance_mi: float
    kind: Optional[str]              # half_marathon | marathon | ultra_50k | None
    goal_type: str                   # "race" | "milestone" (capability, not an event)
    target_date: Optional[date]      # None => no date set
    peak_long_run_mi: float
    taper_weeks: int

    @property
    def has_set_date(self) -> bool:
        return self.target_date is not None

    def weeks_away(self, as_of: date) -> Optional[int]:
        if self.target_date is None:
            return None
        return max((self.target_date - as_of).days // 7, 0)

    def is_upcoming(self, as_of: date) -> bool:
        """Undated milestones are always upcoming — that's the point of
        them. Dated ones stop being upcoming once the date has passed."""
        return self.target_date is None or self.target_date >= as_of

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "distance_mi": self.distance_mi,
            "kind": self.kind,
            "goal_type": self.goal_type,
            "has_set_date": self.has_set_date,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "peak_long_run_mi": self.peak_long_run_mi,
            "taper_weeks": self.taper_weeks,
        }

    def describe(self, as_of: Optional[date] = None) -> str:
        label = self.name.replace("_", " ")
        if self.target_date is None:
            return (f"{label} ({self.distance_mi:.1f} mi) — no date set; build conservatively to a "
                    f"{self.peak_long_run_mi:.0f} mi long run, then a {self.taper_weeks}-week taper")
        away = f", {self.weeks_away(as_of)} weeks away" if as_of else ""
        return (f"{label} ({self.distance_mi:.1f} mi) — {self.target_date.isoformat()}{away}; "
                f"smooth ramp to {self.peak_long_run_mi:.0f} mi, {self.taper_weeks}-week taper")


def _milestone_from_goal(goal) -> Milestone:
    kind = milestone_kind(goal.distance_mi)
    prep = MILESTONE_PREP.get(kind or "", {})
    target = date.fromisoformat(goal.date) if goal.date else None
    return Milestone(
        name=goal.name,
        distance_mi=goal.distance_mi,
        kind=kind,
        goal_type=goal.type,
        target_date=target,
        # A milestone can't require a longer training run than the event.
        peak_long_run_mi=min(prep.get("peak_long_run_mi", DEFAULT_PEAK_LONG_RUN_MI), goal.distance_mi),
        taper_weeks=prep.get("taper_weeks", DEFAULT_TAPER_WEEKS),
    )


def load_milestones(config: Config) -> list[Milestone]:
    """Every configured goal as a Milestone, dated ones first in date
    order, then undated (which are sequenced by readiness, not calendar)."""
    milestones = [_milestone_from_goal(g) for g in config.goals]
    return sorted(milestones, key=lambda m: (m.target_date is None, m.target_date or date.max, m.distance_mi))


def upcoming_milestones(config: Config, as_of: date) -> list[Milestone]:
    return [m for m in load_milestones(config) if m.is_upcoming(as_of)]


def next_milestone(config: Config, as_of: date) -> Optional[Milestone]:
    """The one the current training block is aimed at. Dated milestones win
    over undated ones — a fixed date is a commitment, an undated goal
    waits."""
    upcoming = upcoming_milestones(config, as_of)
    dated = [m for m in upcoming if m.has_set_date]
    if dated:
        return dated[0]
    return upcoming[0] if upcoming else None


def milestone_by_name(config: Config, name: str) -> Optional[Milestone]:
    return next((m for m in load_milestones(config) if m.name == name), None)


def taper_start(milestone: Milestone) -> Optional[date]:
    """Monday of the first taper week, for a dated milestone."""
    if milestone.target_date is None:
        return None
    race_week_start = milestone.target_date - timedelta(days=milestone.target_date.weekday())
    return race_week_start - timedelta(weeks=milestone.taper_weeks)


def milestones_payload(config: Config, as_of: date) -> dict:
    """What the coach and the UI read. Deliberately spells out the
    consequence of each date state so the answer doesn't have to be
    re-derived by whatever is consuming it."""
    all_ms = load_milestones(config)
    upcoming = [m for m in all_ms if m.is_upcoming(as_of)]
    nxt = next_milestone(config, as_of)
    return {
        "as_of": as_of.isoformat(),
        "milestones": [
            {**m.to_dict(), "weeks_away": m.weeks_away(as_of), "summary": m.describe(as_of)}
            for m in upcoming
        ],
        "dated": [m.name for m in upcoming if m.has_set_date],
        "undated": [m.name for m in upcoming if not m.has_set_date],
        "next_milestone": nxt.name if nxt else None,
        "completed_or_past": [m.name for m in all_ms if not m.is_upcoming(as_of)],
    }
