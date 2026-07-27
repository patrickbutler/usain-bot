"""Split-run merging: a single run recorded as multiple Garmin activities
(watch stopped/restarted, battery swap, accidental save) must be analyzed
as one session, or classification and load anchors will misread one long
run as several short ones.

Rule (per athlete spec): if the gap between one run *ending* and the next
*beginning* is 3 hours or less, they are parts of the same run. More than
3 hours apart → separate runs.

This is a read-time analysis transform: stored activities stay raw and
append-only (each Garmin recording is its own immutable row); merging
happens here, on the way into classification/anchors. That's deliberate —
the merge rule can be tuned later (config sync.merge_gap_hours) and
re-applied to history without any data migration.

Only running activities merge, and only when both have a start_time —
without timestamps the gap is unknowable, so we conservatively keep them
separate rather than guess.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .models import Activity, ActivityType

DEFAULT_MERGE_GAP_HOURS = 3.0


def _combine(parts: list[Activity]) -> Activity:
    if len(parts) == 1:
        return parts[0]
    total_distance = sum(p.distance_mi for p in parts)
    total_duration = sum(p.duration_s for p in parts)

    hr_weighted = [
        (p.avg_hr, p.duration_s) for p in parts if p.avg_hr is not None and p.duration_s > 0
    ]
    avg_hr = (
        round(sum(hr * dur for hr, dur in hr_weighted) / sum(dur for _, dur in hr_weighted))
        if hr_weighted else None
    )
    max_hrs = [p.max_hr for p in parts if p.max_hr is not None]
    elevations = [p.elevation_gain_ft for p in parts if p.elevation_gain_ft is not None]

    first = parts[0]
    return Activity(
        activity_id="+".join(p.activity_id for p in parts),
        date=first.date,
        activity_type=first.activity_type,
        distance_mi=total_distance,
        duration_s=total_duration,
        avg_pace_min_per_mi=(total_duration / 60.0 / total_distance) if total_distance > 0 else None,
        avg_hr=avg_hr,
        max_hr=max(max_hrs) if max_hrs else None,
        elevation_gain_ft=sum(elevations) if elevations else None,
        name=(first.name or "Run") + f" (merged x{len(parts)})",
        start_time=first.start_time,
        raw={"merged_activity_ids": [p.activity_id for p in parts]},
    )


def merge_split_runs(activities: list[Activity], gap_hours: float = DEFAULT_MERGE_GAP_HOURS) -> list[Activity]:
    """Collapse split recordings into single sessions. Non-running
    activities and runs without start_time pass through untouched."""
    runs_with_time = sorted(
        (a for a in activities if a.activity_type == ActivityType.RUNNING and a.start_time is not None),
        key=lambda a: a.start_time,
    )
    passthrough = [
        a for a in activities
        if a.activity_type != ActivityType.RUNNING or a.start_time is None
    ]

    max_gap = timedelta(hours=gap_hours)
    merged: list[Activity] = []
    group: list[Activity] = []
    for run in runs_with_time:
        if not group:
            group = [run]
            continue
        prev_end = group[-1].end_time
        if prev_end is not None and (run.start_time - prev_end) <= max_gap:
            group.append(run)
        else:
            merged.append(_combine(group))
            group = [run]
    if group:
        merged.append(_combine(group))

    return sorted(merged + passthrough, key=lambda a: (a.date, a.start_time or datetime.min))
