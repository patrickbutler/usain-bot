"""Training stages: what phase a plan week is in, and what that phase
means for its mileage.

Every week of a plan belongs to exactly one stage, and the stage is not a
label bolted on afterwards — it *decides* the week's long run and volume
(`stage_targets`). A week tagged "maintain" that quietly keeps climbing
would be a lie, so the tag and the numbers come from the same place.

The four stages:

* **increase_mileage** — climbing toward a milestone's peak long run,
  bounded by the §5.1 guardrail increment.
* **maintain_mileage** — the peak has been reached and the milestone is
  still weeks away. Hold weekly volume high while *oscillating* the long
  run, so the athlete keeps the aerobic base without repeating the most
  injurious session. This is where the "no back-to-back max weeks" rule
  lives.
* **taper** — strategic reduction into a milestone.
* **reduce_mileage** — deliberately below the high-mileage line with no
  milestone being chased: post-race recovery, or an off-season block.

Why maintain oscillates rather than plateaus
--------------------------------------------
Runs past `HIGH_MILEAGE_LONG_RUN_MI` carry disproportionate injury risk,
so they buy adaptation at a cost. Once you can run 22, running 22 every
week doesn't make you better at it — it just accumulates the cost. The
cycle below spends one peak week, then backs off and rebuilds toward it,
so the athlete touches the peak repeatedly without stacking it:

    peak 22.0  ->  16.1  ->  18.7  ->  back-off 15.4  ->  peak 22.0 ...

The cycle length follows the athlete's configured back-off cadence
(3 build : 1 back-off by default), so the back-off week the rest of the
system already expects *is* the last week of each maintain cycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class TrainingStage(str, Enum):
    INCREASE_MILEAGE = "increase_mileage"
    MAINTAIN_MILEAGE = "maintain_mileage"
    TAPER = "taper"
    REDUCE_MILEAGE = "reduce_mileage"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


ALL_STAGES = tuple(s.value for s in TrainingStage)

# Long runs past this distance are where injury risk concentrates, so
# reduce_mileage is defined as sitting below it.
HIGH_MILEAGE_LONG_RUN_MI = 15.0

# A week counts as "at peak" within this tolerance. Tight on purpose: this
# decides whether the no-back-to-back-max rule has been broken, and 21.9
# after 22.0 is a repeated peak in everything but arithmetic.
PEAK_TOLERANCE_MI = 0.5

# Maintain-cycle long run as a fraction of the peak. The first week of a
# cycle is the peak itself; the middle weeks climb from FLOOR to CEILING;
# the last week is the back-off. Calibrated so a 22 mi peak on a 3:1
# cadence yields 22.0 / 16.1 / 18.7 / 15.4.
MAINTAIN_FLOOR_FRACTION = 0.73
MAINTAIN_CEILING_FRACTION = 0.85
MAINTAIN_BACKOFF_FRACTION = 0.70

# Volume behaviour. The point of maintain is high mileage at lower strain,
# so volume holds while the long run dips — the missing long-run miles are
# redistributed across the week rather than dropped.
MAINTAIN_VOLUME_FRACTION = 1.00
MAINTAIN_BACKOFF_VOLUME_FRACTION = 0.75

# Taper: fraction of peak volume / peak long run, by week from the race.
# Index 0 is the week furthest from the race.
TAPER_VOLUME_FRACTIONS = (0.60, 0.45, 0.35)
TAPER_LONG_RUN_FRACTIONS = (0.55, 0.36, 0.25)

REDUCE_LONG_RUN_CAP_MI = HIGH_MILEAGE_LONG_RUN_MI - 1.0


def maintain_cycle_length(backoff_cadence: int) -> int:
    """Weeks in one maintain cycle: N build weeks plus the back-off."""
    return max(int(backoff_cadence), 1) + 1


def maintain_long_run_fraction(week_in_cycle: int, backoff_cadence: int) -> float:
    """Fraction of the peak long run for a given position in the cycle.

    Position 0 is the peak; the final position is the back-off; the middle
    positions climb from FLOOR toward CEILING so the athlete rebuilds
    toward the peak instead of sitting just under it.
    """
    cycle = maintain_cycle_length(backoff_cadence)
    pos = week_in_cycle % cycle
    if pos == 0:
        return 1.0
    if pos == cycle - 1:
        return MAINTAIN_BACKOFF_FRACTION
    middle_count = cycle - 2          # positions 1 .. cycle-2
    if middle_count <= 1:
        return MAINTAIN_FLOOR_FRACTION
    step = (MAINTAIN_CEILING_FRACTION - MAINTAIN_FLOOR_FRACTION) / (middle_count - 1)
    return MAINTAIN_FLOOR_FRACTION + step * (pos - 1)


def is_maintain_backoff_week(week_in_cycle: int, backoff_cadence: int) -> bool:
    cycle = maintain_cycle_length(backoff_cadence)
    return week_in_cycle % cycle == cycle - 1


def is_maintain_peak_week(week_in_cycle: int, backoff_cadence: int) -> bool:
    return week_in_cycle % maintain_cycle_length(backoff_cadence) == 0


def stage_targets(
    stage: TrainingStage,
    peak_long_run_mi: float,
    peak_volume_mi: float,
    week_in_stage: int,
    backoff_cadence: int = 3,
    weeks_remaining_in_stage: Optional[int] = None,
    current_long_run_mi: Optional[float] = None,
) -> tuple[float, float, bool]:
    """The (long_run_mi, volume_mi, is_backoff) a stage dictates for one week.

    This is the function that makes stages real: the generator asks the
    stage what the week looks like rather than deciding for itself and
    labelling afterwards.

    `current_long_run_mi` matters only for INCREASE_MILEAGE, where the
    caller owns the guardrail-limited climb and this function just reports
    the ceiling it may not exceed.
    """
    if stage is TrainingStage.MAINTAIN_MILEAGE:
        fraction = maintain_long_run_fraction(week_in_stage, backoff_cadence)
        backoff = is_maintain_backoff_week(week_in_stage, backoff_cadence)
        volume_fraction = MAINTAIN_BACKOFF_VOLUME_FRACTION if backoff else MAINTAIN_VOLUME_FRACTION
        return peak_long_run_mi * fraction, peak_volume_mi * volume_fraction, backoff

    if stage is TrainingStage.TAPER:
        total = len(TAPER_VOLUME_FRACTIONS)
        # Index from the *end*: the last taper week is always the sharpest
        # cut regardless of how many taper weeks there are.
        remaining = weeks_remaining_in_stage if weeks_remaining_in_stage is not None else 1
        idx = max(0, min(total - remaining, total - 1))
        return (peak_long_run_mi * TAPER_LONG_RUN_FRACTIONS[idx],
                peak_volume_mi * TAPER_VOLUME_FRACTIONS[idx], False)

    if stage is TrainingStage.REDUCE_MILEAGE:
        return min(REDUCE_LONG_RUN_CAP_MI, peak_long_run_mi * 0.45), peak_volume_mi * 0.45, False

    # INCREASE_MILEAGE: the peak is the ceiling, the climb rate belongs to
    # the guardrails, so report the ceiling and let the caller clamp.
    backoff = is_maintain_backoff_week(week_in_stage, backoff_cadence)
    return peak_long_run_mi, peak_volume_mi, backoff


def is_repeated_peak(prev_long_run_mi: float, cur_long_run_mi: float, block_peak_mi: float,
                     climb_tolerance_mi: float = 0.1) -> bool:
    """Whether two consecutive weeks both run the block's peak long run.

    The single definition of the hard rule, shared by the validator and
    the milestone rules engine so the two can never drift apart.

    A *rising* pair is excluded: under a gentle ramp two climbing weeks
    can both land inside the peak band, and that's progression toward the
    peak, not a repeat of it.
    """
    if block_peak_mi <= 0:
        return False
    both_at_peak = (prev_long_run_mi >= block_peak_mi - PEAK_TOLERANCE_MI
                    and cur_long_run_mi >= block_peak_mi - PEAK_TOLERANCE_MI)
    still_climbing = cur_long_run_mi > prev_long_run_mi + climb_tolerance_mi
    return both_at_peak and not still_climbing


# Blocks that are inherently a taper or a wind-down, regardless of numbers.
_TAPER_BLOCKS = {"hm_taper", "marathon_taper", "ultra_taper"}
_RACE_BLOCKS = {"half_marathon", "marathon", "ultra_50k"}
_REDUCE_BLOCKS = {"recovery", "return_to_running"}


def classify_plan(weeks: list, backoff_cadence: int = 3) -> list[str]:
    """Stage for every week of an existing plan, inferred from its shape.

    A race week takes the stage of its taper — it is the end of that
    block, not a stage of its own.
    """
    if not weeks:
        return []

    build_weeks = [w for w in weeks if w.block not in _TAPER_BLOCKS | _RACE_BLOCKS | _REDUCE_BLOCKS]
    peak = max((w.long_run_mi for w in build_weeks), default=0.0)

    stages: list[str] = []
    running_peak = 0.0
    for week in weeks:
        if week.block in _TAPER_BLOCKS or week.block in _RACE_BLOCKS:
            stages.append(TrainingStage.TAPER.value)
            continue
        if week.block in _REDUCE_BLOCKS:
            stages.append(TrainingStage.REDUCE_MILEAGE.value)
            continue
        running_peak = max(running_peak, week.long_run_mi)
        # Once the plan's peak has been touched, later build weeks are
        # holding it rather than climbing to it — even the low weeks of
        # the oscillation.
        reached_peak = running_peak >= peak - PEAK_TOLERANCE_MI and peak > 0
        stages.append(
            TrainingStage.MAINTAIN_MILEAGE.value if reached_peak else TrainingStage.INCREASE_MILEAGE.value
        )
    return stages


def annotate_stages(weeks: list, backoff_cadence: int = 3) -> list:
    """Fill in `stage` on any week that doesn't already carry one."""
    inferred = classify_plan(weeks, backoff_cadence)
    for week, stage in zip(weeks, inferred):
        if not week.stage:
            week.stage = stage
    return weeks


def stage_summary(weeks: list) -> list[dict]:
    """Contiguous runs of the same stage — how the plan reads as a
    sequence of phases rather than 40-odd rows."""
    out: list[dict] = []
    for week in weeks:
        stage = week.stage or ""
        if out and out[-1]["stage"] == stage:
            out[-1]["weeks"] += 1
            out[-1]["end_week"] = week.week_number
            out[-1]["end_date"] = week.start_date.isoformat()
            out[-1]["peak_long_run_mi"] = round(max(out[-1]["peak_long_run_mi"], week.long_run_mi), 1)
        else:
            out.append({
                "stage": stage,
                "label": stage.replace("_", " "),
                "start_week": week.week_number,
                "end_week": week.week_number,
                "start_date": week.start_date.isoformat(),
                "end_date": week.start_date.isoformat(),
                "weeks": 1,
                "peak_long_run_mi": round(week.long_run_mi, 1),
            })
    return out
