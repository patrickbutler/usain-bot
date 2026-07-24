"""Macro plan generation, versioning, and conversational overrides.

Scope note: this module produces the *rolling forward projection* shown
in output section B (volume + long run per week). It is intentionally
week-granular, not session-granular — fine-grained same-week rules like
"never increase volume and intensity together" or "never bump the long
run the same week as a new quality session" (§5.6) are enforced by
agent.py's live decision procedure for *today*, which is what actually
binds. The projection here is regenerated from current anchors on every
invocation, so it never goes stale — it's a forecast, not a commitment.

Macro arc: base_building -> half_marathon_capability -> marathon_block
-> taper -> marathon -> recovery -> ultra_specific_block -> ultra_50k.
Per the spec, the 50K is never scheduled with a real date until the
marathon is done and recovery data confirms readiness, so the last two
phases are always emitted as explicit TBD placeholders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from . import guardrails as gr
from .config import Config
from .models import Anchors, PlanVersion, PlanWeek

TAPER_WEEKS = 3
MARATHON_PEAK_WEEKS_OUT = 3
MARATHON_LONG_RUN_MIN_MI = 20.0
MARATHON_LONG_RUN_MAX_MI = 22.0
RECOVERY_VOLUME_PCT_OF_PEAK = 0.40
HALF_MARATHON_BENCHMARK_MI = 13.1
OVERRIDE_EASIER_WEEK_FACTOR = 0.75


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def generate_macro_plan(
    config: Config,
    anchors: Anchors,
    start_date: date,
    version: int,
    trigger: str,
    rationale: str,
    gap_hold_weeks: int = 0,
    gap_easy_only: bool = False,
    marathon_date_override: Optional[date] = None,
) -> PlanVersion:
    """Generate the full macro arc from current actuals. `gap_hold_weeks`
    implements the §5.4 backtrack: for that many build weeks, hold
    volume/long-run flat (or easy-only) before resuming normal growth.
    """
    baseline_long_run = (
        anchors.long_run_anchor_mi if anchors.long_run_anchor_mi > 0
        else config.athlete.baseline_long_run_mi
    )
    if anchors.chronic_load_mi > 0:
        baseline_volume = anchors.chronic_load_mi
    else:
        # Cold start: no chronic load yet, so size volume so the athlete's
        # configured baseline long run sits right at (not above) the
        # pct-of-weekly-volume cap, rather than an arbitrary multiplier
        # that could clip a long run they've already been safely running.
        baseline_volume = baseline_long_run / config.guardrails.long_run_pct_of_weekly_volume

    marathon_goal = config.goal("marathon")
    if marathon_date_override is not None:
        marathon_date = marathon_date_override
    elif marathon_goal and marathon_goal.date:
        marathon_date = date.fromisoformat(marathon_goal.date)
    else:
        raise ValueError("marathon goal must have a date to generate the macro plan")

    taper_start = marathon_date - timedelta(weeks=TAPER_WEEKS)

    weeks: list[PlanWeek] = []
    week_num = 1
    cur_date = _week_start(start_date)
    cur_volume = baseline_volume
    cur_long_run = baseline_long_run
    weeks_since_backoff = 0
    milestone_hit = cur_long_run >= HALF_MARATHON_BENCHMARK_MI
    milestone_week: Optional[int] = week_num if milestone_hit else None
    peak_volume = cur_volume
    peak_long_run = cur_long_run
    holds_remaining = gap_hold_weeks
    # Tracks the volume/long-run at the start of the current build cycle
    # (i.e. right after the last back-off week). Three weeks of +10% growth
    # compounds to only ~1.33x, which a 70-75% cutback nearly (or fully)
    # erases -- taken completely literally over many cycles that produces
    # a net decline instead of the progressive overload the macro arc
    # requires. The floor below guarantees each cycle can't regress past
    # where the previous one started, without ever exceeding the injury
    # guardrails (pct-of-volume cap always wins if the two conflict).
    cycle_start_volume = cur_volume
    cycle_start_long_run = cur_long_run

    while cur_date < taper_start:
        is_backoff = gr.should_insert_backoff_week(weeks_since_backoff, config.guardrails.backoff_cadence)
        holding = holds_remaining > 0

        if holding:
            block = "return_to_running" if gap_easy_only else "base_building"
            volume = cur_volume  # flat, no increase
            long_run = 0.0 if gap_easy_only else cur_long_run
            quality = 0
            notes = "Holding post-gap: no volume/long-run increase this week." if not gap_easy_only \
                else "Post-gap rebuild: easy running only, no long run yet."
            holds_remaining -= 1
            weeks_since_backoff += 1
        elif is_backoff:
            targets = gr.backoff_week_targets(cur_volume, peak_long_run)
            block = _current_block_label(cur_long_run, milestone_hit)
            volume = max(targets.volume_mi, cycle_start_volume)
            quality = 0
            pct_cap = gr.max_long_run_by_weekly_volume_pct(volume, in_ultra_block=False)
            # §5.1/§5.5: never increase the long run in a back-off week, and
            # it still must not exceed the pct-of-weekly-volume ceiling —
            # 70% of peak can violate both once volume has shrunk more than
            # the long run has, so take the min of all three (the ratchet
            # floor is applied first, then clamped back down by the cap).
            long_run = min(max(targets.long_run_mi, cycle_start_long_run), cur_long_run, pct_cap)
            notes = "Scheduled back-off week (3:1 cadence): no long-run increase, no quality."
            weeks_since_backoff = 0
            cycle_start_volume, cycle_start_long_run = volume, long_run
        else:
            proposed_volume = cur_volume * gr.WEEKLY_VOLUME_GROWTH_FACTOR
            proposed_long_run = gr.next_long_run_distance(cur_long_run)
            pct_cap = gr.max_long_run_by_weekly_volume_pct(proposed_volume, in_ultra_block=False)
            long_run = min(proposed_long_run, pct_cap, MARATHON_LONG_RUN_MAX_MI)
            volume = proposed_volume
            quality = 1 if milestone_hit else 0
            block = _current_block_label(long_run, milestone_hit)
            notes = ""
            weeks_since_backoff += 1

        weeks.append(PlanWeek(week_num, cur_date, block, volume, long_run, quality, is_backoff and not holding, notes))

        cur_volume, cur_long_run = volume, (long_run if long_run > 0 else cur_long_run)
        peak_volume = max(peak_volume, volume)
        peak_long_run = max(peak_long_run, long_run)
        if not milestone_hit and long_run >= HALF_MARATHON_BENCHMARK_MI:
            milestone_hit = True
            milestone_week = week_num

        week_num += 1
        cur_date += timedelta(days=7)

    # Taper block (final TAPER_WEEKS weeks up to, but not including, race week).
    taper_long_run = peak_long_run
    for i in range(TAPER_WEEKS - 1):
        taper_volume = gr.taper_week_volume(peak_volume, reduction_pct=0.40 + 0.03 * i)
        taper_long_run = peak_long_run * (0.6 - 0.15 * i)
        weeks.append(PlanWeek(
            week_num, cur_date, "taper", taper_volume, max(taper_long_run, 0), 1,
            False, "Taper: volume down, retain some intensity.",
        ))
        week_num += 1
        cur_date += timedelta(days=7)

    # Race week.
    race_week_start = _week_start(marathon_date)
    marathon_note = f"Marathon race week — target date {marathon_date.isoformat()}."
    if peak_long_run < MARATHON_LONG_RUN_MIN_MI:
        required_volume = peak_long_run / config.guardrails.long_run_pct_of_weekly_volume
        marathon_note += (
            f" [!] Projected peak long run ({peak_long_run:.1f} mi) falls short of the "
            f"{MARATHON_LONG_RUN_MIN_MI:.0f}-{MARATHON_LONG_RUN_MAX_MI:.0f} mi marathon target under "
            f"current constraints ({config.athlete.available_run_days_per_week} running days/week, "
            f"long run capped at {config.guardrails.long_run_pct_of_weekly_volume:.0%} of weekly volume). "
            f"Reaching {MARATHON_LONG_RUN_MIN_MI:.0f} mi at that cap needs ~{required_volume:.0f} mi/wk "
            "total, which is hard to sustain on this many running days without violating the pct-of-volume "
            "guardrail. This is a static forecast from today's anchors — actual chronic load should rise "
            "as real training accumulates and gets re-derived on every invocation, which will likely close "
            "part of this gap; the rest would need an added running day, a later marathon date, or an "
            "explicit override accepting a higher long-run share."
        )
    weeks.append(PlanWeek(
        week_num, race_week_start, "marathon", peak_volume * 0.35, 26.2, 0, False,
        marathon_note,
    ))
    week_num += 1
    cur_date = race_week_start + timedelta(days=7)

    # Recovery block (non-negotiable reverse taper).
    recovery_weeks = config.sequencing.post_marathon_recovery_weeks
    for i in range(recovery_weeks):
        recovery_volume = peak_volume * RECOVERY_VOLUME_PCT_OF_PEAK * (0.8 + 0.2 * i / max(recovery_weeks - 1, 1))
        recovery_long_run = min(
            6.0, peak_long_run * 0.4,
            gr.max_long_run_by_weekly_volume_pct(recovery_volume, in_ultra_block=False),
        )
        weeks.append(PlanWeek(
            week_num, cur_date, "recovery", recovery_volume, recovery_long_run, 0, False,
            "Post-marathon recovery block: easy only, no long-run growth.",
        ))
        week_num += 1
        cur_date += timedelta(days=7)

    # Ultra-specific block + 50K: deliberately unscheduled per spec §2/§4 —
    # never scheduled until the marathon is complete and recovery data
    # confirms readiness.
    weeks.append(PlanWeek(
        week_num, cur_date, "ultra_specific_block_TBD", 0.0, 0.0, 0, False,
        (
            f"Placeholder: ~{config.sequencing.ultra_specific_block_weeks} weeks, begins after the "
            "recovery block. NOT scheduled — will be generated from post-marathon recovery actuals."
        ),
    ))
    week_num += 1
    weeks.append(PlanWeek(
        week_num, cur_date, "ultra_50k_TBD", 0.0, 0.0, 0, False,
        "Placeholder: NOT scheduled. 50K date is null until the marathon is complete and recovery confirms readiness.",
    ))

    return PlanVersion(
        version=version,
        created_at=datetime.utcnow(),
        trigger=trigger,
        rationale=rationale,
        weeks=weeks,
    )


def _current_block_label(long_run_mi: float, milestone_hit: bool) -> str:
    if not milestone_hit and long_run_mi < HALF_MARATHON_BENCHMARK_MI:
        return "base_building"
    if long_run_mi < MARATHON_LONG_RUN_MIN_MI:
        return "half_marathon_capability" if long_run_mi < 16 else "marathon_block"
    return "marathon_block"


_DIFF_COMPARE_TOLERANCE_MI = 0.1


@dataclass(frozen=True)
class WeekDiff:
    """One week's before/after across two plan versions. `old`/`new` are
    None for added/removed weeks. This is the structured form the web
    UI renders as a table; `diff_plan_versions` below is a text renderer
    over the same data, kept for the CLI/plan-version rationale field."""

    week_number: int
    change_type: str  # "added" | "removed" | "changed" | "unchanged"
    old: Optional[PlanWeek]
    new: Optional[PlanWeek]
    changed_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "week_number": self.week_number,
            "change_type": self.change_type,
            "old": self.old.to_dict() if self.old else None,
            "new": self.new.to_dict() if self.new else None,
            "changed_fields": list(self.changed_fields),
        }


def diff_plan_weeks(old: Optional[PlanVersion], new: PlanVersion) -> list[WeekDiff]:
    """Structured per-week lineage between two plan versions. `old=None`
    (first-ever plan) reports every week as added."""
    if old is None:
        return [WeekDiff(w.week_number, "added", None, w) for w in new.weeks]

    old_by_week = {w.week_number: w for w in old.weeks}
    new_by_week = {w.week_number: w for w in new.weeks}
    diffs: list[WeekDiff] = []
    for wn in sorted(set(old_by_week) | set(new_by_week)):
        ow, nw = old_by_week.get(wn), new_by_week.get(wn)
        if ow is None:
            diffs.append(WeekDiff(wn, "added", None, nw))
            continue
        if nw is None:
            diffs.append(WeekDiff(wn, "removed", ow, None))
            continue

        changed_fields = []
        if abs(ow.target_volume_mi - nw.target_volume_mi) >= _DIFF_COMPARE_TOLERANCE_MI:
            changed_fields.append("target_volume_mi")
        if abs(ow.long_run_mi - nw.long_run_mi) >= _DIFF_COMPARE_TOLERANCE_MI:
            changed_fields.append("long_run_mi")
        if ow.block != nw.block:
            changed_fields.append("block")
        if ow.is_backoff != nw.is_backoff:
            changed_fields.append("is_backoff")

        diffs.append(WeekDiff(
            wn, "changed" if changed_fields else "unchanged", ow, nw, tuple(changed_fields),
        ))
    return diffs


def diff_plan_versions(old: Optional[PlanVersion], new: PlanVersion) -> str:
    if old is None:
        return f"Initial plan (v{new.version}): {len(new.weeks)} weeks generated."

    lines = [f"Plan v{old.version} -> v{new.version}:"]
    changed = 0
    for d in diff_plan_weeks(old, new):
        if d.change_type == "added":
            lines.append(
                f"  + week {d.week_number}: added ({d.new.block}, {d.new.target_volume_mi:.1f} mi / "
                f"LR {d.new.long_run_mi:.1f} mi)"
            )
            changed += 1
        elif d.change_type == "removed":
            lines.append(f"  - week {d.week_number}: removed (was {d.old.block})")
            changed += 1
        elif d.change_type == "changed":
            lines.append(
                f"  ~ week {d.week_number}: {d.old.block}/{d.old.target_volume_mi:.1f}mi/LR{d.old.long_run_mi:.1f} "
                f"-> {d.new.block}/{d.new.target_volume_mi:.1f}mi/LR{d.new.long_run_mi:.1f}"
            )
            changed += 1
    if changed == 0:
        lines.append("  (no material changes)")
    return "\n".join(lines)


# --- conversational overrides -------------------------------------------------

@dataclass
class OverrideResult:
    plan: Optional[PlanVersion]
    applied: bool
    rationale: str
    warnings: list[str]


def _first_future_week(plan: PlanVersion, as_of: date) -> Optional[PlanWeek]:
    upcoming = [w for w in plan.weeks if w.start_date > _week_start(as_of)]
    return min(upcoming, key=lambda w: w.start_date) if upcoming else None


# These three functions are the actual mutations behind every override,
# whether it arrives via the CLI's regex-based `apply_override` (below) or
# the chat UI's LLM tool-calls (usain_bot/chat/tools.py) — the LLM never
# computes a new volume/long-run/date itself, it only ever picks which of
# these to call and with what arguments; the arithmetic still lives here.

def ease_week(
    plan: PlanVersion, week_number: Optional[int], as_of: date, reason: str,
    factor: float = OVERRIDE_EASIER_WEEK_FACTOR,
) -> OverrideResult:
    """Reduce a specific (or the next upcoming) week's volume/long-run by
    `factor` and mark it as a manual back-off."""
    if week_number is None:
        target = _first_future_week(plan, as_of)
        if target is None:
            return OverrideResult(None, False, "No upcoming week found to ease.", ["No future week in the current plan to modify."])
        week_number = target.week_number
    else:
        target = next((w for w in plan.weeks if w.week_number == week_number), None)
        if target is None:
            return OverrideResult(None, False, "", [f"No week {week_number} in the current plan."])

    factor = min(max(factor, 0.3), 0.95)
    warnings: list[str] = []
    new_weeks = []
    for w in plan.weeks:
        if w.week_number == week_number:
            new_volume = w.target_volume_mi * factor
            new_long_run = w.long_run_mi * factor
            if new_volume < w.target_volume_mi * gr.BACKOFF_VOLUME_PCT_LOW * 0.8:
                warnings.append(
                    f"Week {w.week_number} volume cut to {new_volume:.1f} mi is below typical "
                    "back-off floor — this is more conservative than the standard guardrail, which is fine."
                )
            new_weeks.append(PlanWeek(
                w.week_number, w.start_date, w.block, new_volume, new_long_run, 0,
                True, f"User override: eased per request ('{reason}').",
            ))
        else:
            new_weeks.append(w)
    new_plan = PlanVersion(
        version=plan.version + 1, created_at=datetime.utcnow(), trigger="user_override",
        rationale=f"User requested an easier week {week_number}: \"{reason}\".",
        weeks=new_weeks,
    )
    return OverrideResult(new_plan, True, new_plan.rationale, warnings)


def shift_marathon(
    plan: PlanVersion, config: Config, anchors: Anchors, as_of: date, weeks: int, reason: str,
) -> OverrideResult:
    """Shift the marathon date by `weeks` (positive = later, negative =
    earlier) and regenerate the full downstream arc from it."""
    marathon_goal = config.goal("marathon")
    if not marathon_goal or not marathon_goal.date:
        return OverrideResult(None, False, "", ["No marathon date configured to shift."])
    cur_date = date.fromisoformat(marathon_goal.date)
    new_date = cur_date + timedelta(weeks=weeks)
    direction = "later" if weeks > 0 else "earlier"
    warnings = [
        f"Marathon date shifted from {cur_date.isoformat()} to {new_date.isoformat()} ({abs(weeks)} week(s) {direction}). "
        "This changes the whole downstream arc (taper, recovery, ultra block start) — "
        "update config.yaml goals.marathon.date to persist this beyond one session."
    ]
    new_plan = generate_macro_plan(
        config, anchors, as_of, plan.version + 1, "user_override",
        f"User requested shifting the marathon {abs(weeks)} week(s) {direction}: \"{reason}\".",
        marathon_date_override=new_date,
    )
    return OverrideResult(new_plan, True, new_plan.rationale, warnings)


def note_long_run_day(plan: PlanVersion, day: str, reason: str) -> OverrideResult:
    """Record a long-run day-of-week preference as a note on every future
    week. Not a per-day schedule structure in this version (see warning)."""
    new_weeks = [
        PlanWeek(
            w.week_number, w.start_date, w.block, w.target_volume_mi, w.long_run_mi,
            w.quality_sessions, w.is_backoff,
            (w.notes + f" | Long run day preference: {day}.").strip(" |"),
        )
        for w in plan.weeks
    ]
    new_plan = PlanVersion(
        version=plan.version + 1, created_at=datetime.utcnow(), trigger="user_override",
        rationale=f"User requested moving the long run to {day}: \"{reason}\".",
        weeks=new_weeks,
    )
    warnings = [
        "Day-of-week scheduling is tracked as a preference note, not a per-day plan structure "
        "in this version — the 7-day rolling view will honor it, but it isn't guardrail-checked "
        "for spacing (e.g. back-to-back hard days)."
    ]
    return OverrideResult(new_plan, True, new_plan.rationale, warnings)


def apply_override(
    plan: PlanVersion,
    text: str,
    config: Config,
    anchors: Anchors,
    as_of: date,
) -> OverrideResult:
    """Regex-based override parser used by the CLI. Recognizes a small,
    fixed set of phrasings and dispatches to the same mutation functions
    the chat UI's LLM tools call. Anything else is returned unapplied
    with a warning rather than silently guessing at intent — for freer
    phrasing, use the chat UI instead.
    """
    lowered = text.lower().strip()

    if re.search(r"(make|keep).{0,20}next week.{0,20}eas(y|ier)", lowered):
        return ease_week(plan, None, as_of, text)

    m = re.search(r"shift.{0,20}marathon.{0,20}back (\w+) weeks?", lowered) or \
        re.search(r"push.{0,20}marathon.{0,20}back (\w+) weeks?", lowered)
    if m:
        n = _word_to_int(m.group(1))
        if n is None:
            return OverrideResult(None, False, "", [f"Could not parse a week count from: '{text}'."])
        return shift_marathon(plan, config, anchors, as_of, n, text)

    day_match = re.search(r"move.{0,20}long run.{0,20}to\s+(\w+)", lowered)
    if day_match:
        return note_long_run_day(plan, day_match.group(1), text)

    return OverrideResult(
        None, False, "",
        [f"Override not recognized: \"{text}\". Supported: easing next week, shifting the marathon "
         "date, moving the long run to a given day. Rephrase, edit config.yaml directly, or use the "
         "chat UI (`usain-bot serve`) for freer phrasing."],
    )


_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _word_to_int(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return _WORD_NUMS.get(token)
