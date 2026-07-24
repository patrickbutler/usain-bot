"""Deterministic guardrail math from §5 of the spec.

Every function here is pure: no I/O, no config file reads, no clock
access. Callers (planner.py, agent.py) pass in already-computed anchors
and get back numbers plus the reasoning string that explains them. The
agent's reasoning sits *on top of* these — it never overrides them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# --- tunable constants (mirrored by config.yaml guardrails: section) -------

LONG_RUN_INCREMENT_ABS_CAP_MI = 1.0
LONG_RUN_INCREMENT_PCT = 0.10
WEEKLY_VOLUME_GROWTH_FACTOR = 1.10
LONG_RUN_PCT_OF_WEEKLY_VOLUME = 0.35
LONG_RUN_PCT_OF_WEEKLY_VOLUME_ULTRA_BLOCK = 0.40
BACKOFF_VOLUME_PCT_LOW = 0.70
BACKOFF_VOLUME_PCT_HIGH = 0.75
BACKOFF_LONG_RUN_PCT = 0.70
DEFAULT_BACKOFF_CADENCE = 3

ACWR_DETRAIN_MIN = 0.8
ACWR_GREEN_MAX = 1.3
ACWR_YELLOW_MAX = 1.5


class ACWRZone(str, Enum):
    DETRAINING = "detraining"   # < 0.8
    GREEN = "green"             # 0.8 - 1.3
    YELLOW = "yellow"           # 1.3 - 1.5, hold flat
    RED = "red"                 # > 1.5, force reduction
    UNDEFINED = "undefined"     # chronic load == 0 (cold start)


# --- 5.1 long-run increment -------------------------------------------------

def long_run_increment(long_run_anchor_mi: float) -> float:
    """min(1.0, 10% of anchor) — resolves the '1 mile vs 10%' tension."""
    if long_run_anchor_mi <= 0:
        return 0.0
    return min(LONG_RUN_INCREMENT_ABS_CAP_MI, LONG_RUN_INCREMENT_PCT * long_run_anchor_mi)


def next_long_run_distance(long_run_anchor_mi: float) -> float:
    return long_run_anchor_mi + long_run_increment(long_run_anchor_mi)


def max_long_run_by_weekly_volume_pct(
    planned_weekly_volume_mi: float, in_ultra_block: bool = False
) -> float:
    """Long run should not exceed ~30-35% of planned weekly volume
    (relaxed toward 40% only inside the ultra-specific block)."""
    pct = LONG_RUN_PCT_OF_WEEKLY_VOLUME_ULTRA_BLOCK if in_ultra_block else LONG_RUN_PCT_OF_WEEKLY_VOLUME
    return planned_weekly_volume_mi * pct


# --- 5.2 weekly volume -------------------------------------------------------

def max_weekly_volume(chronic_load_mi: float) -> float:
    """Hard ceiling: chronic load * 1.10. Applies even if the athlete feels great."""
    return chronic_load_mi * WEEKLY_VOLUME_GROWTH_FACTOR


# --- 5.3 ACWR -----------------------------------------------------------------

def acwr(acute_load_mi: float, chronic_load_mi: float) -> Optional[float]:
    """None when chronic load is zero (cold start / not enough history)."""
    if chronic_load_mi <= 0:
        return None
    return acute_load_mi / chronic_load_mi


def acwr_zone(acwr_value: Optional[float]) -> ACWRZone:
    if acwr_value is None:
        return ACWRZone.UNDEFINED
    if acwr_value < ACWR_DETRAIN_MIN:
        return ACWRZone.DETRAINING
    if acwr_value <= ACWR_GREEN_MAX:
        return ACWRZone.GREEN
    if acwr_value <= ACWR_YELLOW_MAX:
        return ACWRZone.YELLOW
    return ACWRZone.RED


# --- 5.4 return-from-gap protocol --------------------------------------------

@dataclass(frozen=True)
class GapAction:
    severity: str
    backtrack_weeks: int
    long_run_allowed: bool
    max_long_run_mi: Optional[float]
    easy_only: bool
    max_quality_sessions: int
    consecutive_easy_required: int
    regenerate_plan: bool
    description: str


def gap_action(gap_days: int, last_completed_long_run_mi: float) -> GapAction:
    """Table from §5.4. `last_completed_long_run_mi` should be the long-run
    anchor derived from *completed* runs only (never from the plan)."""
    if gap_days > 42:
        return GapAction(
            severity="severe",
            backtrack_weeks=0,
            long_run_allowed=False,
            max_long_run_mi=None,
            easy_only=True,
            max_quality_sessions=0,
            consecutive_easy_required=3,
            regenerate_plan=True,
            description=(
                f"Gap of {gap_days} days (>6 weeks): regenerate the plan from scratch "
                "using current actuals as the new baseline."
            ),
        )
    if gap_days > 14:
        weeks_missed = math.ceil(gap_days / 7)
        return GapAction(
            severity="long",
            backtrack_weeks=weeks_missed,
            long_run_allowed=False,
            max_long_run_mi=None,
            easy_only=True,
            max_quality_sessions=0,
            consecutive_easy_required=3,
            regenerate_plan=False,
            description=(
                f"Gap of {gap_days} days (>14 days): backtrack {weeks_missed} week(s) "
                "in the progression, one week per week missed. Rebuild from the current "
                "rolling average, not the plan. Easy running only until 3 consecutive "
                "completed runs."
            ),
        )
    if gap_days >= 7:
        return GapAction(
            severity="medium",
            backtrack_weeks=1,
            long_run_allowed=False,
            max_long_run_mi=None,
            easy_only=True,
            max_quality_sessions=0,
            consecutive_easy_required=2,
            regenerate_plan=False,
            description=(
                f"Gap of {gap_days} days (7-14 days): backtrack 1 week in the "
                "progression. First two runs back are easy/base only — no long run."
            ),
        )
    return GapAction(
        severity="short",
        backtrack_weeks=0,
        long_run_allowed=True,
        max_long_run_mi=last_completed_long_run_mi,
        easy_only=False,
        max_quality_sessions=2,
        consecutive_easy_required=0,
        regenerate_plan=False,
        description=(
            f"Gap of {gap_days} days (<7 days): resume plan, but cap long run at the "
            f"last completed long run ({last_completed_long_run_mi:.1f} mi). No increase."
        ),
    )


# --- 5.5 back-off weeks --------------------------------------------------------

@dataclass(frozen=True)
class BackoffWeekTargets:
    volume_mi: float
    long_run_mi: float
    quality_sessions: int = 0


def backoff_week_targets(
    prior_week_volume_mi: float,
    recent_peak_long_run_mi: float,
    volume_pct: float = BACKOFF_VOLUME_PCT_LOW,
) -> BackoffWeekTargets:
    """~70-75% of prior week's volume, long run capped at ~70% of recent peak,
    zero quality sessions."""
    clamped_pct = min(max(volume_pct, BACKOFF_VOLUME_PCT_LOW), BACKOFF_VOLUME_PCT_HIGH)
    return BackoffWeekTargets(
        volume_mi=prior_week_volume_mi * clamped_pct,
        long_run_mi=recent_peak_long_run_mi * BACKOFF_LONG_RUN_PCT,
        quality_sessions=0,
    )


def should_insert_backoff_week(weeks_since_last_backoff: int, cadence: int = DEFAULT_BACKOFF_CADENCE) -> bool:
    """Default cadence is 3 build weeks : 1 back-off week."""
    return weeks_since_last_backoff >= cadence


# --- 5.6 general safety rules --------------------------------------------------

def max_quality_sessions_per_week(is_return_from_gap_rebuild: bool) -> int:
    return 0 if is_return_from_gap_rebuild else 2


def is_valid_easy_hard_split(easy_mi: float, hard_mi: float, tolerance_pct: float = 0.10) -> bool:
    """Roughly 80/20 easy-to-hard distribution by volume, with tolerance."""
    total = easy_mi + hard_mi
    if total <= 0:
        return True
    hard_pct = hard_mi / total
    return hard_pct <= 0.20 + tolerance_pct


def taper_week_volume(peak_volume_mi: float, reduction_pct: float = 0.45) -> float:
    """Taper: reduce volume 40-50%, retain some intensity. Default 45% cut."""
    reduction_pct = min(max(reduction_pct, 0.40), 0.50)
    return peak_volume_mi * (1 - reduction_pct)
