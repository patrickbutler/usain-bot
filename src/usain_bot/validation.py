"""Deterministic training-plan validation. Pure functions, no I/O.

Encodes the athlete's milestone rules as checks that run against any
generated plan (planner.py aims to satisfy them; this module is the
independent referee that says whether it actually did):

- Half marathon: a 12 mi run must appear before its taper; taper is 1-2
  weeks immediately before the race week.
- Marathon: race week must sit on the FIXED configured date; a 20 mi run
  must appear before its taper; taper is 2-3 weeks.
- 50K: at least 3 consecutive weeks of >= 30 mi cumulative volume must
  appear before its taper; taper is 2-3 weeks.
- Build gradually: week-over-week long-run increases within the
  min(1 mi, 10%) guardrail; weekly volume growth within +10% (checked
  between consecutive build weeks only — tapers, races, back-offs and
  recovery transitions are expected discontinuities).
- Back-off cadence: no more than 4 consecutive build weeks without a
  back-off in the pre-marathon span.

Errors are rule violations; warnings are honest observations that don't
invalidate the plan (e.g. long-run share of weekly volume still high
while volume catches up to demonstrated long-run capability).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from . import guardrails as gr
from . import stages
from .models import PlanWeek

HM_PREREQ_LONG_RUN_MI = 12.0
HM_TAPER_RANGE = (1, 2)
MARATHON_PREREQ_LONG_RUN_MI = 20.0
MARATHON_TAPER_RANGE = (2, 3)
ULTRA_PREREQ_WEEKLY_MI = 30.0
ULTRA_PREREQ_CONSECUTIVE_WEEKS = 3
ULTRA_TAPER_RANGE = (2, 3)

_LR_INCREMENT_TOLERANCE_MI = 0.06
_VOL_GROWTH_TOLERANCE = 0.015
_HIGH_LONG_RUN_SHARE = 0.55
_MAX_CONSECUTIVE_BUILD_WEEKS = 4

_BUILD_BLOCKS = {"base_building", "marathon_block", "ultra_build"}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message}


def _find_race_index(weeks: list[PlanWeek], block: str) -> int | None:
    for i, w in enumerate(weeks):
        if w.block == block:
            return i
    return None


def _taper_length_before(weeks: list[PlanWeek], race_index: int, taper_block: str) -> int:
    count = 0
    i = race_index - 1
    while i >= 0 and weeks[i].block == taper_block:
        count += 1
        i -= 1
    return count


def _check_race(
    issues: list[ValidationIssue], weeks: list[PlanWeek], race_block: str, taper_block: str,
    taper_range: tuple[int, int], label: str,
) -> int | None:
    race_index = _find_race_index(weeks, race_block)
    if race_index is None:
        issues.append(ValidationIssue("warning", f"{race_block}_not_scheduled", f"{label} is not scheduled in this plan."))
        return None
    taper_len = _taper_length_before(weeks, race_index, taper_block)
    lo, hi = taper_range
    if not (lo <= taper_len <= hi):
        issues.append(ValidationIssue(
            "error", f"{race_block}_taper_length",
            f"{label} taper is {taper_len} week(s); required {lo}-{hi} weeks immediately before the race.",
        ))
    return race_index


def _contiguous_build_blocks(weeks: list[PlanWeek]) -> list[list[PlanWeek]]:
    """Build weeks grouped into runs of the same block, so each training
    block is judged against its own peak rather than the plan's."""
    groups: list[list[PlanWeek]] = []
    for w in weeks:
        if w.block not in _BUILD_BLOCKS:
            continue
        if groups and groups[-1][-1].block == w.block and w.week_number == groups[-1][-1].week_number + 1:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def validate_plan(weeks: list[PlanWeek], marathon_date: date) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not weeks:
        return [ValidationIssue("error", "empty_plan", "Plan has no weeks.")]

    # --- half marathon ---------------------------------------------------------
    hm_index = _check_race(issues, weeks, "half_marathon", "hm_taper", HM_TAPER_RANGE, "Half marathon")
    if hm_index is not None:
        taper_start = hm_index - _taper_length_before(weeks, hm_index, "hm_taper")
        if not any(w.long_run_mi >= HM_PREREQ_LONG_RUN_MI - 1e-6 for w in weeks[:taper_start]):
            issues.append(ValidationIssue(
                "error", "half_marathon_prereq",
                f"No {HM_PREREQ_LONG_RUN_MI:.0f} mi long run scheduled before the half-marathon taper.",
            ))

    # --- marathon (fixed date) -------------------------------------------------
    m_index = _check_race(issues, weeks, "marathon", "marathon_taper", MARATHON_TAPER_RANGE, "Marathon")
    if m_index is None:
        issues.append(ValidationIssue("error", "marathon_missing", "Marathon race week is missing from the plan."))
    else:
        expected_start = marathon_date - timedelta(days=marathon_date.weekday())
        if weeks[m_index].start_date != expected_start:
            issues.append(ValidationIssue(
                "error", "marathon_date",
                f"Marathon race week starts {weeks[m_index].start_date.isoformat()}, but the fixed race "
                f"date {marathon_date.isoformat()} falls in the week of {expected_start.isoformat()}.",
            ))
        taper_start = m_index - _taper_length_before(weeks, m_index, "marathon_taper")
        pre_taper = [w for w in weeks[:taper_start] if w.block != "half_marathon"]
        if not any(w.long_run_mi >= MARATHON_PREREQ_LONG_RUN_MI - 1e-6 for w in pre_taper):
            issues.append(ValidationIssue(
                "error", "marathon_prereq",
                f"No {MARATHON_PREREQ_LONG_RUN_MI:.0f} mi long run scheduled before the marathon taper.",
            ))

    # --- 50K -------------------------------------------------------------------
    u_index = _check_race(issues, weeks, "ultra_50k", "ultra_taper", ULTRA_TAPER_RANGE, "50K")
    if u_index is not None:
        taper_start = u_index - _taper_length_before(weeks, u_index, "ultra_taper")
        run = best = 0
        for w in weeks[:taper_start]:
            run = run + 1 if w.target_volume_mi >= ULTRA_PREREQ_WEEKLY_MI - 1e-6 else 0
            best = max(best, run)
        if best < ULTRA_PREREQ_CONSECUTIVE_WEEKS:
            issues.append(ValidationIssue(
                "error", "ultra_prereq",
                f"Only {best} consecutive week(s) at >={ULTRA_PREREQ_WEEKLY_MI:.0f} mi before the 50K "
                f"taper; {ULTRA_PREREQ_CONSECUTIVE_WEEKS} required.",
            ))

    # --- gradual build ---------------------------------------------------------
    # The increment guardrail governs *progression* — going further than
    # you have gone before. Inside the maintain stage the long run
    # oscillates by design, and a rebound toward a distance already run in
    # that stage is a return to demonstrated capability, not a new load. So
    # the check applies to anything above the stage's established peak, and
    # skips the rebounds below it.
    demonstrated_peak = 0.0
    for prev, cur in zip(weeks, weeks[1:]):
        demonstrated_peak = max(demonstrated_peak, prev.long_run_mi)
        if cur.week_number != prev.week_number + 1:
            continue
        if prev.block not in _BUILD_BLOCKS or cur.block not in _BUILD_BLOCKS:
            continue
        if prev.is_backoff or cur.is_backoff:
            continue
        maintaining = (prev.stage == stages.TrainingStage.MAINTAIN_MILEAGE.value
                       and cur.stage == stages.TrainingStage.MAINTAIN_MILEAGE.value)
        if maintaining and cur.long_run_mi <= demonstrated_peak + _LR_INCREMENT_TOLERANCE_MI:
            continue
        max_lr_step = gr.long_run_increment(prev.long_run_mi) + _LR_INCREMENT_TOLERANCE_MI
        if cur.long_run_mi - prev.long_run_mi > max_lr_step:
            issues.append(ValidationIssue(
                "error", "long_run_jump",
                f"Week {cur.week_number} long run jumps {prev.long_run_mi:.1f} -> {cur.long_run_mi:.1f} mi, "
                f"exceeding the min(1 mi, 10%) increment guardrail.",
            ))
        if prev.target_volume_mi > 0 and (
            cur.target_volume_mi / prev.target_volume_mi > gr.WEEKLY_VOLUME_GROWTH_FACTOR + _VOL_GROWTH_TOLERANCE
        ):
            issues.append(ValidationIssue(
                "error", "volume_jump",
                f"Week {cur.week_number} volume grows {prev.target_volume_mi:.1f} -> "
                f"{cur.target_volume_mi:.1f} mi, exceeding +10%/week.",
            ))

    # --- back-off cadence (pre-marathon span) ----------------------------------
    consecutive_builds = 0
    for w in weeks:
        if w.block in ("marathon_taper", "marathon"):
            break
        if w.block in _BUILD_BLOCKS and not w.is_backoff:
            consecutive_builds += 1
            if consecutive_builds > _MAX_CONSECUTIVE_BUILD_WEEKS:
                issues.append(ValidationIssue(
                    "warning", "backoff_cadence",
                    f"More than {_MAX_CONSECUTIVE_BUILD_WEEKS} consecutive build weeks around week "
                    f"{w.week_number} without a back-off.",
                ))
                consecutive_builds = 0
        else:
            consecutive_builds = 0

    # --- peak long runs are never back-to-back ---------------------------------
    # The single most load-bearing rule of the maintain stage: a max-mileage
    # week must be followed by a back-off (or the taper). Two 22-milers in a
    # row is the pattern that breaks people, and it is an error rather than
    # a warning because no amount of context makes it acceptable.
    # Checked per contiguous build block: the ultra block peaks lower than
    # the marathon block, and two peak weeks in a row there is just as bad
    # even though the number is smaller. A single plan-wide peak would let
    # the smaller block's plateau through.
    for block_weeks in _contiguous_build_blocks(weeks):
        block_peak = max(w.long_run_mi for w in block_weeks)
        if block_peak <= 0:
            continue
        for prev, cur in zip(block_weeks, block_weeks[1:]):
            if cur.week_number != prev.week_number + 1:
                continue
            if stages.is_repeated_peak(prev.long_run_mi, cur.long_run_mi, block_peak,
                                        _LR_INCREMENT_TOLERANCE_MI):
                issues.append(ValidationIssue(
                    "error", "back_to_back_peak",
                    f"Weeks {prev.week_number} and {cur.week_number} both run the peak long run "
                    f"({prev.long_run_mi:.1f} / {cur.long_run_mi:.1f} mi). A max-mileage week must be "
                    "followed by a back-off or the taper.",
                ))

    # --- long-run share (informational) ----------------------------------------
    high_share_weeks = [
        w.week_number for w in weeks
        if w.block in _BUILD_BLOCKS and w.target_volume_mi > 0
        and w.long_run_mi / w.target_volume_mi > _HIGH_LONG_RUN_SHARE
    ]
    if high_share_weeks:
        issues.append(ValidationIssue(
            "warning", "long_run_share",
            f"Long run exceeds {_HIGH_LONG_RUN_SHARE:.0%} of weekly volume in weeks "
            f"{high_share_weeks[:8]}{'...' if len(high_share_weeks) > 8 else ''} — expected while weekly "
            "volume catches up to demonstrated long-run capability; shrinks as the plan progresses.",
        ))

    return issues
