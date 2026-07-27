"""Data-quality gate for everything entering storage.

Every write path (incremental sync and full backfill alike) runs
activities through `sanitize_activities` before saving and
`dedupe_stored_activities` after, so data validation and deduplication
are properties of ingestion rather than chores the athlete has to
remember to run. There is deliberately no `usain-bot dedupe` /
`usain-bot validate-data` command — if you ever need them by hand,
something upstream is broken.

Two distinct failure modes are handled differently:

- **Repairable**: a field is implausible but the activity is still a
  real run (e.g. a heart-rate spike of 400 bpm, or max_hr below avg_hr).
  The bad field is nulled and the run is kept — losing a whole run
  because one sensor glitched would corrupt mileage anchors far worse
  than a missing HR value.
- **Unusable**: the activity itself can't be trusted (negative distance,
  zero duration, a GPS teleport implying 900 mph, a date in the future).
  It's dropped, counted, and logged rather than silently admitted —
  a single bogus 300-mile "run" would wreck every load anchor and the
  entire plan built on them.

Thresholds are deliberately loose. The job here is catching corruption,
not judging training: a 25 min/mi hike-jog and a 4:30/mi track rep are
both legitimate and both pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from .models import Activity

logger = logging.getLogger("usain_bot.ingest")

# Corruption bounds, not training judgments.
MIN_PLAUSIBLE_PACE_MIN_PER_MI = 2.5    # ~24 mph sustained — GPS glitch, not a human
MAX_PLAUSIBLE_PACE_MIN_PER_MI = 60.0   # slower than 60 min/mi over a whole activity
MAX_PLAUSIBLE_DISTANCE_MI = 200.0      # beyond any single logged run
MAX_PLAUSIBLE_DURATION_S = 48 * 3600
MIN_PLAUSIBLE_HR = 20
MAX_PLAUSIBLE_HR = 250
FUTURE_DATE_TOLERANCE_DAYS = 2         # timezone/clock skew, not time travel

# Same physical run under two Garmin IDs.
DUPLICATE_START_TOLERANCE_S = 120
DUPLICATE_DISTANCE_TOLERANCE = 0.05


@dataclass
class IngestReport:
    accepted: list[Activity] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (activity_id, reason)
    repaired: list[tuple[str, str]] = field(default_factory=list)   # (activity_id, what)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def repaired_count(self) -> int:
        return len(self.repaired)

    def summary(self) -> str:
        parts = []
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected as unusable")
        if self.repaired:
            parts.append(f"{len(self.repaired)} field(s) repaired")
        return ", ".join(parts)


def _rejection_reason(a: Activity, today: date) -> str | None:
    if a.distance_mi is None or a.distance_mi < 0:
        return f"negative/missing distance ({a.distance_mi})"
    if a.duration_s is None or a.duration_s <= 0:
        return f"non-positive duration ({a.duration_s}s)"
    if a.distance_mi == 0:
        return "zero distance"
    if a.distance_mi > MAX_PLAUSIBLE_DISTANCE_MI:
        return f"implausible distance ({a.distance_mi:.1f} mi)"
    if a.duration_s > MAX_PLAUSIBLE_DURATION_S:
        return f"implausible duration ({a.duration_s / 3600:.1f} h)"
    if a.date > today + timedelta(days=FUTURE_DATE_TOLERANCE_DAYS):
        return f"date in the future ({a.date.isoformat()})"

    pace = (a.duration_s / 60.0) / a.distance_mi
    if pace < MIN_PLAUSIBLE_PACE_MIN_PER_MI:
        return f"implausible pace ({pace:.2f} min/mi — likely a GPS glitch)"
    if pace > MAX_PLAUSIBLE_PACE_MIN_PER_MI:
        return f"implausible pace ({pace:.1f} min/mi)"
    return None


def _repair(a: Activity, report: IngestReport) -> Activity:
    avg_hr, max_hr = a.avg_hr, a.max_hr

    if avg_hr is not None and not (MIN_PLAUSIBLE_HR <= avg_hr <= MAX_PLAUSIBLE_HR):
        report.repaired.append((a.activity_id, f"dropped implausible avg_hr={avg_hr}"))
        avg_hr = None
    if max_hr is not None and not (MIN_PLAUSIBLE_HR <= max_hr <= MAX_PLAUSIBLE_HR):
        report.repaired.append((a.activity_id, f"dropped implausible max_hr={max_hr}"))
        max_hr = None
    if avg_hr is not None and max_hr is not None and max_hr < avg_hr:
        report.repaired.append((a.activity_id, f"dropped max_hr={max_hr} below avg_hr={avg_hr}"))
        max_hr = None

    elevation = a.elevation_gain_ft
    if elevation is not None and elevation < 0:
        report.repaired.append((a.activity_id, f"dropped negative elevation_gain_ft={elevation}"))
        elevation = None

    # Recompute pace from distance/duration rather than trusting whatever
    # Garmin reported — the two can disagree after a manual edit.
    pace = (a.duration_s / 60.0) / a.distance_mi if a.distance_mi > 0 else None

    if (avg_hr, max_hr, elevation) == (a.avg_hr, a.max_hr, a.elevation_gain_ft) and \
            a.avg_pace_min_per_mi is not None and abs((a.avg_pace_min_per_mi or 0) - (pace or 0)) < 0.01:
        return a
    return replace(a, avg_hr=avg_hr, max_hr=max_hr, elevation_gain_ft=elevation, avg_pace_min_per_mi=pace)


def sanitize_activities(activities: list[Activity], today: date | None = None) -> IngestReport:
    """Validate and repair activities before they reach storage.

    Also drops within-batch duplicate activity_ids, which Garmin can
    return when a request window overlaps a paging boundary — saving
    them would be harmless (upsert) but the counts would lie."""
    today = today or date.today()
    report = IngestReport()
    seen_ids: set[str] = set()

    for a in activities:
        if not a.activity_id:
            report.rejected.append(("<missing id>", "activity has no id"))
            continue
        if a.activity_id in seen_ids:
            report.rejected.append((a.activity_id, "duplicate activity_id within the same batch"))
            continue

        reason = _rejection_reason(a, today)
        if reason:
            report.rejected.append((a.activity_id, reason))
            logger.warning("Rejecting activity %s: %s", a.activity_id, reason)
            continue

        seen_ids.add(a.activity_id)
        report.accepted.append(_repair(a, report))

    return report


def find_duplicate_groups(activities: list[Activity]) -> list[list[Activity]]:
    """Group activities that look like the same physical run stored under
    different Garmin IDs: same date and type, start times within two
    minutes (or both missing), distances within 5%.

    Distinct from split-run merging (sessions.py): a run deliberately
    recorded in two parts 25 minutes apart is two real activities that get
    merged at read time, not duplicates to delete. The tight start-time
    tolerance here is what keeps those cases apart.

    Each group is ordered with the record worth keeping first (richest raw
    payload, tie-broken by lowest id for determinism)."""
    import json

    groups: list[list[Activity]] = []
    used: set[str] = set()
    buckets: dict[tuple, list[Activity]] = {}
    for a in activities:
        buckets.setdefault((a.date, a.activity_type), []).append(a)

    for bucket in buckets.values():
        bucket = sorted(bucket, key=lambda x: x.activity_id)
        for i, a in enumerate(bucket):
            if a.activity_id in used:
                continue
            group = [a]
            for b in bucket[i + 1:]:
                if b.activity_id in used:
                    continue
                if a.start_time and b.start_time:
                    close = abs((a.start_time - b.start_time).total_seconds()) <= DUPLICATE_START_TOLERANCE_S
                else:
                    close = a.start_time is None and b.start_time is None
                largest = max(a.distance_mi, b.distance_mi, 0.01)
                similar = abs(a.distance_mi - b.distance_mi) / largest <= DUPLICATE_DISTANCE_TOLERANCE
                if close and similar:
                    group.append(b)
                    used.add(b.activity_id)
            if len(group) > 1:
                used.add(a.activity_id)
                group.sort(key=lambda x: (-len(json.dumps(x.raw, default=str)), x.activity_id))
                groups.append(group)
    return groups
