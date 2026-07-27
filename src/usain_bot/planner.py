"""Macro plan generation, versioning, and conversational overrides.

The plan is generated FROM ACTUALS: the long-run progression starts at
the athlete's demonstrated long-run anchor (longest completed run,
trailing 21 days) and weekly volume starts at actual chronic load. The
plan never schedules a long run below what the athlete has already
proven they can run — an earlier version clipped the long run to a
percent-of-volume cap and produced a beginner plan for an athlete with
an 8-mile long run on the books; the share cap is now a *reported
warning* (see validation.py) while actual volume catches up, never a
reason to un-earn demonstrated capability.

Milestone rules (deterministic, enforced by validation.py):
- Half marathon (flexible date): requires a 12 mi run completed before a
  1-2 week taper. Scheduled automatically at the earliest build week
  where the long run reaches 12 mi (plus any athlete-requested delay).
- Marathon (FIXED date, config goals.marathon.date): requires a 20 mi
  run before a 2-3 week taper.
- 50K (flexible date): requires >= 30 mi/week cumulative for at least 3
  consecutive weeks before a 2-3 week taper. Scheduled after marathon
  recovery once the volume prerequisite is met (plus requested delay).

Athlete-requested milestone pushes ("move my half marathon back 2
weeks") persist as preferences (storage) and re-enter generation as
hm_delay_weeks / ultra_delay_weeks — the plan fills the extra time with
normal build/back-off weeks so the base is maintained, rather than
going flat.

Scope note: week-granular (volume + long run per week). Fine-grained
same-week rules (§5.6) are enforced live by agent.py for "today". The
projection regenerates from current anchors on every invocation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from . import guardrails as gr
from .config import Config
from .models import Anchors, PlanVersion, PlanWeek

# Milestone prerequisites (athlete-specified, deterministic).
HM_DISTANCE_MI = 13.1
HM_PREREQ_LONG_RUN_MI = 12.0
HM_TAPER_WEEKS = 1              # valid range 1-2 (validation.py)

MARATHON_DISTANCE_MI = 26.2
MARATHON_PREREQ_LONG_RUN_MI = 20.0
MARATHON_TAPER_WEEKS = 2        # valid range 2-3
MARATHON_LONG_RUN_MAX_MI = 22.0

ULTRA_DISTANCE_MI = 31.1
ULTRA_PREREQ_WEEKLY_MI = 30.0
ULTRA_PREREQ_CONSECUTIVE_WEEKS = 3
ULTRA_TAPER_WEEKS = 2           # valid range 2-3
ULTRA_LONG_RUN_MAX_MI = 24.0
ULTRA_VOLUME_CAP_MI = 38.0

# Long-run share of weekly volume the generator *steers toward* as volume
# grows. Not a clip: demonstrated capability always wins (see module
# docstring); validation.py reports weeks where the share is still high.
TARGET_LONG_RUN_SHARE = 0.45

BACKOFF_VOLUME_FACTOR = 0.72
BACKOFF_LONG_RUN_FACTOR = 0.70
RECOVERY_WEEKS_VOLUME = (0.30, 0.35, 0.40)   # of marathon-block peak
OVERRIDE_EASIER_WEEK_FACTOR = 0.75

_MAX_PLAN_WEEKS = 130


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


@dataclass
class _BuildState:
    lr: float
    vol: float
    peak_lr: float
    peak_vol: float
    weeks_since_backoff: int = 0

    def build_week(self, lr_cap: float, vol_cap: float) -> tuple[float, float]:
        """One build week: long run += min(1mi, 10%) up to lr_cap; volume
        grows toward lr/TARGET_SHARE, at most +10%/week, never shrinking."""
        next_lr = min(self.lr + gr.long_run_increment(self.lr), lr_cap)
        target_vol = next_lr / TARGET_LONG_RUN_SHARE
        next_vol = min(max(self.vol, min(target_vol, self.vol * gr.WEEKLY_VOLUME_GROWTH_FACTOR)), vol_cap)
        self.lr, self.vol = next_lr, next_vol
        self.peak_lr = max(self.peak_lr, next_lr)
        self.peak_vol = max(self.peak_vol, next_vol)
        self.weeks_since_backoff += 1
        return next_lr, next_vol

    def backoff_week(self) -> tuple[float, float]:
        lr = min(self.peak_lr * BACKOFF_LONG_RUN_FACTOR, self.lr)
        vol = self.vol * BACKOFF_VOLUME_FACTOR
        self.weeks_since_backoff = 0
        # lr/vol state intentionally NOT reduced: next build resumes from
        # the pre-backoff level (the ratchet), so back-off weeks recover
        # without surrendering progression.
        return lr, vol


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
    hm_delay_weeks: int = 0,
    ultra_delay_weeks: int = 0,
    run_days_per_week: Optional[int] = None,
) -> PlanVersion:
    """Generate the full macro arc from current actuals. See module
    docstring for the milestone rules this encodes."""
    run_days = run_days_per_week or anchors.runs_per_week or config.athlete.available_run_days_per_week

    # Baselines from actuals. At a true cold start (no data), fall back to
    # configured baseline; otherwise the athlete's real numbers win.
    baseline_lr = anchors.long_run_anchor_mi if anchors.long_run_anchor_mi > 0 else config.athlete.baseline_long_run_mi
    baseline_vol = anchors.chronic_load_mi if anchors.chronic_load_mi > 0 else baseline_lr * 1.6

    marathon_goal = config.goal("marathon")
    if marathon_date_override is not None:
        marathon_date = marathon_date_override
    elif marathon_goal and marathon_goal.date:
        marathon_date = date.fromisoformat(marathon_goal.date)
    else:
        raise ValueError("marathon goal must have a date to generate the macro plan")

    race_week_start = _week_start(marathon_date)
    marathon_taper_start = race_week_start - timedelta(weeks=MARATHON_TAPER_WEEKS)

    weeks: list[PlanWeek] = []
    week_num = 1
    cur_date = _week_start(start_date)
    state = _BuildState(lr=baseline_lr, vol=baseline_vol, peak_lr=baseline_lr, peak_vol=baseline_vol)
    holds_remaining = gap_hold_weeks
    hm_done = False
    hm_delay_remaining = max(0, hm_delay_weeks)
    cadence = config.guardrails.backoff_cadence

    def add(block: str, vol: float, lr: float, quality: int, is_backoff: bool, notes: str = "") -> None:
        nonlocal week_num, cur_date
        weeks.append(PlanWeek(week_num, cur_date, block, round(vol, 2), round(lr, 2), quality, is_backoff, notes))
        week_num += 1
        cur_date += timedelta(days=7)

    # ---- pre-marathon build (base -> HM -> marathon block) -------------------
    while cur_date < marathon_taper_start and week_num < _MAX_PLAN_WEEKS:
        weeks_remaining = (marathon_taper_start - cur_date).days // 7

        if holds_remaining > 0:
            block = "return_to_running" if gap_easy_only else "base_building"
            lr = 0.0 if gap_easy_only else state.lr
            notes = ("Post-gap rebuild: easy running only, no long run yet." if gap_easy_only
                     else "Holding post-gap: no volume/long-run increase this week.")
            add(block, state.vol, lr, 0, False, notes)
            holds_remaining -= 1
            continue

        hm_ready = (not hm_done) and state.lr >= HM_PREREQ_LONG_RUN_MI
        if hm_ready and hm_delay_remaining == 0 and weeks_remaining >= HM_TAPER_WEEKS + 3:
            add("hm_taper", state.vol * 0.65, state.peak_lr * 0.6, 0, False,
                f"Half-marathon taper: 12 mi prerequisite met (long run at {state.lr:.1f} mi).")
            add("half_marathon", state.vol * 0.6 + HM_DISTANCE_MI * 0.4, HM_DISTANCE_MI, 0, False,
                "Half-marathon race week (flexible date — scheduled at earliest readiness).")
            state.lr = max(state.lr, HM_DISTANCE_MI)
            state.peak_lr = max(state.peak_lr, HM_DISTANCE_MI)
            hm_done = True
            continue
        if hm_ready and hm_delay_remaining > 0:
            hm_delay_remaining -= 1  # keep building; base is maintained, race waits

        if gr.should_insert_backoff_week(state.weeks_since_backoff, cadence):
            lr, vol = state.backoff_week()
            add(_block_label(hm_done), vol, lr, 0, True,
                "Scheduled back-off week (3:1 cadence): no long-run increase, no quality.")
            continue

        lr, vol = state.build_week(lr_cap=MARATHON_LONG_RUN_MAX_MI, vol_cap=MARATHON_PREREQ_LONG_RUN_MI / TARGET_LONG_RUN_SHARE)
        quality = 1 if hm_done else 0
        add(_block_label(hm_done), vol, lr, quality, False)

    # ---- marathon taper + race -----------------------------------------------
    peak_lr, peak_vol = state.peak_lr, state.peak_vol
    taper_notes = ""
    if peak_lr + 1e-6 < MARATHON_PREREQ_LONG_RUN_MI:
        taper_notes = (
            f"[!] Peak long run projected at {peak_lr:.1f} mi — the 20 mi marathon prerequisite is NOT "
            "met before this taper. See plan validation for details."
        )
    add("marathon_taper", peak_vol * 0.60, min(12.0, peak_lr), 1, False,
        ("Marathon taper (2 weeks): volume down, retain some intensity. " + taper_notes).strip())
    add("marathon_taper", peak_vol * 0.45, min(8.0, peak_lr), 0, False,
        "Marathon taper: final sharpening week.")
    # Race week pinned to the fixed date.
    cur_date = race_week_start
    add("marathon", peak_vol * 0.35, MARATHON_DISTANCE_MI, 0, False,
        f"Marathon race week — fixed date {marathon_date.isoformat()}.")

    # ---- recovery -------------------------------------------------------------
    for pct in RECOVERY_WEEKS_VOLUME[: config.sequencing.post_marathon_recovery_weeks]:
        add("recovery", peak_vol * pct, min(6.0, peak_lr * 0.4), 0, False,
            "Post-marathon recovery block: easy only, no long-run growth.")

    # ---- ultra build -> >=30 mi/wk x3 -> taper -> 50K -------------------------
    ustate = _BuildState(
        lr=min(12.0, peak_lr * 0.6), vol=peak_vol * 0.45,
        peak_lr=min(12.0, peak_lr * 0.6), peak_vol=peak_vol * 0.45,
    )
    consecutive_30 = 0
    ultra_hold_remaining = max(0, ultra_delay_weeks)
    while week_num < _MAX_PLAN_WEEKS:
        if ustate.vol >= ULTRA_PREREQ_WEEKLY_MI:
            consecutive_30 += 1
            notes = f"Ultra prerequisite week {consecutive_30}/{ULTRA_PREREQ_CONSECUTIVE_WEEKS}: >=30 mi cumulative."
            if consecutive_30 > ULTRA_PREREQ_CONSECUTIVE_WEEKS:
                notes = "Holding >=30 mi/wk base (athlete-requested delay)."
            lr = min(ustate.lr, ULTRA_LONG_RUN_MAX_MI)
            add("ultra_build", ustate.vol, lr, 0, False,
                notes + " Back-to-back long runs allowed in this block only.")
            if consecutive_30 >= ULTRA_PREREQ_CONSECUTIVE_WEEKS:
                if ultra_hold_remaining > 0:
                    ultra_hold_remaining -= 1
                else:
                    break
            continue
        if gr.should_insert_backoff_week(ustate.weeks_since_backoff, cadence):
            lr, vol = ustate.backoff_week()
            consecutive_30 = 0 if vol < ULTRA_PREREQ_WEEKLY_MI else consecutive_30
            add("ultra_build", vol, lr, 0, True, "Back-off week inside ultra build.")
            continue
        lr, vol = ustate.build_week(lr_cap=ULTRA_LONG_RUN_MAX_MI, vol_cap=ULTRA_VOLUME_CAP_MI)
        add("ultra_build", vol, lr, 0, False)

    upeak_vol = max(ustate.peak_vol, ustate.vol)
    add("ultra_taper", upeak_vol * 0.60, 10.0, 0, False, "Ultra taper (2 weeks).")
    add("ultra_taper", upeak_vol * 0.45, 8.0, 0, False, "Ultra taper: final week.")
    ultra_race_date = cur_date
    add("ultra_50k", upeak_vol * 0.40, ULTRA_DISTANCE_MI, 0, False,
        f"50K race week — flexible date, projected {ultra_race_date.isoformat()} from the volume "
        "prerequisite (>=30 mi/wk x3) plus taper. Confirmed after marathon recovery data.")

    return PlanVersion(
        version=version,
        created_at=datetime.utcnow(),
        trigger=trigger,
        rationale=rationale,
        weeks=weeks,
    )


def _block_label(hm_done: bool) -> str:
    return "marathon_block" if hm_done else "base_building"


# --- plan diffs (unchanged behavior) -----------------------------------------

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


# These functions are the actual mutations behind every override, whether
# it arrives via the CLI's regex-based `apply_override` (below) or the
# chat UI's LLM tool-calls (usain_bot/chat/tools.py) — the LLM never
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
