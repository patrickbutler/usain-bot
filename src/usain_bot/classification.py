"""Turn raw Garmin activities into classified runs and load anchors.

Per §4.1 of the spec: never blend long runs with easy runs into a single
average. Every activity is classified first; anchors are computed only
from the classified stream. Classification and anchor math here are pure
functions of a list of Activities plus a reference date — no I/O.

Classification heuristics (documented since Garmin's raw payload doesn't
give us a clean "this was a workout" flag in the common case):

- cross-training: activity_type is not RUNNING.
- long: the longest run in its rolling 7-day window (centered on the run's
  own date, so a long run early in the week is still compared against
  runs later that same week — not just what came before it), OR >= 1.4x
  the window's median run distance. Requires >= 2 runs in the window
  (otherwise "longest of one" is meaningless) and, for the "is the max"
  branch, requires the run to be strictly longer than the window median
  (so an exact tie between two same-distance runs doesn't spuriously
  crown one of them "the long run").
- quality: workout-ish naming (tempo/interval/threshold/track/...) or a
  sustained high avg_hr/max_hr ratio (>= 0.88), detected among the
  activities not already classified as long.
- recovery: short (<= 0.6x the recent median run distance) and low
  relative effort (avg_hr/max_hr < 0.72 when HR is available).
- everything else: easy/base.
"""

from __future__ import annotations

from datetime import date, timedelta
from statistics import median
from typing import Optional

from .models import (
    Activity,
    ActivityType,
    Anchors,
    ClassifiedActivity,
    GapInfo,
    GapSeverity,
    RunClass,
)

RUNNING_CLASSES = {RunClass.LONG, RunClass.EASY, RunClass.QUALITY, RunClass.RECOVERY}

_LONG_RUN_MEDIAN_MULTIPLE = 1.4
_QUALITY_HR_RATIO = 0.88
_RECOVERY_DISTANCE_MULTIPLE = 0.6
_RECOVERY_HR_RATIO = 0.72
_QUALITY_KEYWORDS = (
    "tempo", "interval", "threshold", "track", "speed", "fartlek", "race", "workout", "vo2",
)


def _looks_like_quality(activity: Activity) -> bool:
    name = (activity.name or "").lower()
    if any(k in name for k in _QUALITY_KEYWORDS):
        return True
    if activity.avg_hr and activity.max_hr and activity.max_hr > 0:
        if activity.avg_hr / activity.max_hr >= _QUALITY_HR_RATIO:
            return True
    return False


def _looks_like_recovery(activity: Activity, running_distances: list[float]) -> bool:
    if not running_distances or activity.distance_mi <= 0:
        return False
    med = median(running_distances)
    if med <= 0:
        return False
    is_short = activity.distance_mi <= _RECOVERY_DISTANCE_MULTIPLE * med
    if not is_short:
        return False
    if activity.avg_hr and activity.max_hr and activity.max_hr > 0:
        return (activity.avg_hr / activity.max_hr) < _RECOVERY_HR_RATIO
    return True


def classify_activities(activities: list[Activity]) -> list[ClassifiedActivity]:
    """Classify every activity per §4.1. Order matters: cross-training is
    determined by type; long is structural (rolling window); quality and
    recovery are heuristic; everything left over is easy/base.
    """
    sorted_acts = sorted(activities, key=lambda a: a.date)
    running = [a for a in sorted_acts if a.activity_type == ActivityType.RUNNING]

    classified: list[ClassifiedActivity] = []
    for activity in sorted_acts:
        if activity.activity_type != ActivityType.RUNNING:
            classified.append(ClassifiedActivity(activity, RunClass.CROSS_TRAINING))
            continue

        window = [
            r for r in running
            if activity.date - timedelta(days=3) <= r.date <= activity.date + timedelta(days=3)
        ]
        window_distances = [r.distance_mi for r in window if r.distance_mi > 0]

        is_long = False
        if len(window_distances) >= 2 and activity.distance_mi > 0:
            med = median(window_distances)
            is_long = (
                (activity.distance_mi == max(window_distances) and activity.distance_mi > med)
                or (med > 0 and activity.distance_mi >= _LONG_RUN_MEDIAN_MULTIPLE * med)
            )

        if is_long:
            classified.append(ClassifiedActivity(activity, RunClass.LONG))
            continue

        if _looks_like_quality(activity):
            classified.append(ClassifiedActivity(activity, RunClass.QUALITY))
            continue

        recent_distances = [
            r.distance_mi for r in running
            if activity.date - timedelta(days=27) <= r.date <= activity.date and r.distance_mi > 0
        ]
        if _looks_like_recovery(activity, recent_distances):
            classified.append(ClassifiedActivity(activity, RunClass.RECOVERY))
            continue

        classified.append(ClassifiedActivity(activity, RunClass.EASY))

    return classified


def prepare_classified(activities: list[Activity], merge_gap_hours: float = 3.0) -> list[ClassifiedActivity]:
    """The standard pipeline every reader should use: merge split-run
    recordings into sessions first (sessions.py), then classify. Anchors
    computed from unmerged activities would misread one split long run
    as several short runs."""
    from .sessions import merge_split_runs

    return classify_activities(merge_split_runs(activities, gap_hours=merge_gap_hours))


def _in_trailing_window(d: date, as_of: date, days: int) -> bool:
    return as_of - timedelta(days=days - 1) <= d <= as_of


def detect_gap(classified: list[ClassifiedActivity], as_of: date) -> GapInfo:
    """Days since the last completed run (any running class), bucketed per §5.4."""
    run_dates = sorted(
        c.activity.date for c in classified
        if c.run_class in RUNNING_CLASSES and c.activity.date <= as_of
    )
    if not run_dates:
        return GapInfo(gap_days=10_000, severity=GapSeverity.SEVERE, last_run_date=None)

    last_run = run_dates[-1]
    gap_days = (as_of - last_run).days

    if gap_days > 42:
        severity = GapSeverity.SEVERE
    elif gap_days > 14:
        severity = GapSeverity.LONG
    elif gap_days >= 7:
        severity = GapSeverity.MEDIUM
    else:
        severity = GapSeverity.SHORT

    return GapInfo(gap_days=gap_days, severity=severity, last_run_date=last_run)


def derive_run_days_per_week(classified: list[ClassifiedActivity], as_of: date) -> Optional[int]:
    """Actual running frequency: distinct days with at least one run in
    the trailing 28 days, divided by 4 and rounded. None when there's no
    recent running to derive from (cold start / long gap) — callers fall
    back to the configured value then. This is what makes the plan adapt
    when the athlete runs more or fewer days than config says."""
    run_days = {
        c.activity.date for c in classified
        if c.run_class in RUNNING_CLASSES and _in_trailing_window(c.activity.date, as_of, 28)
    }
    if not run_days:
        return None
    return max(1, min(7, round(len(run_days) / 4)))


def compute_anchors(
    classified: list[ClassifiedActivity],
    as_of: date,
    planned_runs_trailing_14d: Optional[int] = None,
) -> Anchors:
    """Load anchors per §4.2. Always derived from actuals."""
    acute_load = sum(
        c.activity.distance_mi for c in classified
        if c.run_class in RUNNING_CLASSES and _in_trailing_window(c.activity.date, as_of, 7)
    )
    chronic_total = sum(
        c.activity.distance_mi for c in classified
        if c.run_class in RUNNING_CLASSES and _in_trailing_window(c.activity.date, as_of, 28)
    )
    chronic_load = chronic_total / 4.0

    long_runs_21d = [
        c.activity.distance_mi for c in classified
        if c.run_class == RunClass.LONG and _in_trailing_window(c.activity.date, as_of, 21)
    ]
    long_run_anchor = max(long_runs_21d) if long_runs_21d else 0.0

    acwr = (acute_load / chronic_load) if chronic_load > 0 else None

    adherence_rate = None
    if planned_runs_trailing_14d:
        completed_14d = sum(
            1 for c in classified
            if c.run_class in RUNNING_CLASSES and _in_trailing_window(c.activity.date, as_of, 14)
        )
        adherence_rate = completed_14d / planned_runs_trailing_14d

    gap = detect_gap(classified, as_of)

    return Anchors(
        as_of=as_of,
        acute_load_mi=acute_load,
        chronic_load_mi=chronic_load,
        long_run_anchor_mi=long_run_anchor,
        adherence_rate=adherence_rate,
        acwr=acwr,
        gap=gap,
        runs_per_week=derive_run_days_per_week(classified, as_of),
    )
