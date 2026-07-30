"""Milestone-shape rules, and the repair loop that enforces them.

`validation.py` is the referee for the *training* rules — build rate,
back-off cadence, taper length, prerequisites. This module is the referee
for the **shape a plan must take given its milestones**, and unlike the
validator it does not merely report: it repairs.

The two shapes, from the milestone's date state (`milestones.py`):

* **Dated milestone** — the calendar is fixed, so the plan must ramp
  smoothly and then *maintain* until the taper. Arriving at peak months
  early and sitting there is the failure mode; so is arriving late and
  never reaching the prerequisite distance.
* **Undated milestone** — no calendar to hit, so run a conservative
  increase stage until the milestone's peak long run is reached, then
  taper straight into it. The date falls out of readiness rather than
  being an input.

Why repair rather than just report
----------------------------------
A plan that breaks these rules is not a plan the athlete should ever see.
Reporting "this plan peaks too early" and rendering it anyway pushes the
judgement onto the person least equipped to make it mid-training. So
`generate_valid_plan` re-generates with corrected inputs until the rules
pass, and if it genuinely cannot satisfy them it says so explicitly
rather than returning a quietly broken plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from . import milestones as ms
from . import stages
from .config import Config
from .models import Anchors, PlanVersion, PlanWeek
from .validation import validate_plan

# How many repair passes before giving up. Each pass makes the plan
# strictly more conservative, so this converges quickly; the cap exists to
# make failure loud rather than infinite.
MAX_REPAIR_PASSES = 4

# An undated milestone taper should start as soon as the peak is reached,
# with at most this much maintain padding.
MAX_UNDATED_MAINTAIN_WEEKS = 2


@dataclass
class RuleViolation:
    code: str
    message: str
    # How the next generation pass should differ to fix this. Empty means
    # the violation is not automatically repairable.
    repair: dict = field(default_factory=dict)
    # "error" blocks; "warning" is a consequence the athlete should see but
    # that the system must not quietly overrule (see `locked` below).
    severity: str = "error"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "severity": self.severity, "repair": self.repair}


@dataclass
class RuleReport:
    violations: list[RuleViolation]
    passes_used: int = 0
    repaired: bool = False

    @property
    def errors(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def ok(self) -> bool:
        """No errors. Warnings are accepted trade-offs, not failures."""
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "passes_used": self.passes_used,
            "repaired": self.repaired,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [v.message for v in self.warnings],
        }


def _build_weeks(weeks: list[PlanWeek], blocks: tuple[str, ...]) -> list[PlanWeek]:
    return [w for w in weeks if w.block in blocks]


def _first_peak_index(weeks: list[PlanWeek], peak: float) -> Optional[int]:
    for i, w in enumerate(weeks):
        if w.long_run_mi >= peak - stages.PEAK_TOLERANCE_MI:
            return i
    return None


def check_milestone_shape(
    weeks: list[PlanWeek],
    milestone: ms.Milestone,
    build_blocks: tuple[str, ...],
    taper_blocks: tuple[str, ...],
) -> list[RuleViolation]:
    """Whether the plan's shape matches what this milestone's date state
    demands."""
    violations: list[RuleViolation] = []
    build = _build_weeks(weeks, build_blocks)
    if not build:
        return [RuleViolation("no_build_weeks", f"No build weeks lead into {milestone.name}.")]

    achieved_peak = max(w.long_run_mi for w in build)
    target_peak = milestone.peak_long_run_mi

    # --- reaching the prerequisite --------------------------------------------
    if achieved_peak + 1e-6 < target_peak:
        shortfall = target_peak - achieved_peak
        violations.append(RuleViolation(
            "peak_not_reached",
            f"{milestone.name} needs a {target_peak:.0f} mi long run before its taper; the plan only "
            f"reaches {achieved_peak:.1f} mi ({shortfall:.1f} mi short).",
            # Climbing sooner is the only lever that adds distance without
            # breaking the increment guardrail.
            {"pacing_mode": "ramp_asap", "ramp_rate": 1.0},
        ))

    # --- the no-back-to-back-peak rule ----------------------------------------
    for prev, cur in zip(build, build[1:]):
        if cur.week_number != prev.week_number + 1:
            continue
        if stages.is_repeated_peak(prev.long_run_mi, cur.long_run_mi, achieved_peak):
            violations.append(RuleViolation(
                "back_to_back_peak",
                f"Weeks {prev.week_number} and {cur.week_number} both run the peak long run "
                f"({prev.long_run_mi:.1f} mi). A max-mileage week must be followed by a back-off.",
                {"force_maintain_oscillation": True},
            ))
            break

    # --- shape per date state --------------------------------------------------
    peak_index = _first_peak_index(build, achieved_peak)
    weeks_at_peak_onward = (len(build) - peak_index) if peak_index is not None else 0

    if milestone.has_set_date:
        # A dated milestone is *supposed* to ramp and then maintain until
        # the date, so a long maintain phase is the goal, not a defect —
        # what matters is that those weeks oscillate (checked above) rather
        # than sitting at peak. The one real failure is the plan still
        # climbing when the taper arrives, which means the ramp was paced
        # to overshoot the calendar.
        if not any(w.stage == stages.TrainingStage.MAINTAIN_MILEAGE.value for w in build):
            violations.append(RuleViolation(
                "no_maintain_stage",
                f"{milestone.name} has a set date, so the plan should reach peak and then maintain "
                "until the taper; the ramp is still climbing when the taper starts.",
                # The fix is to climb sooner. If the athlete deliberately
                # slowed the ramp, that fix is theirs to decline and this
                # becomes a reported trade-off instead.
                {"ramp_rate": 1.0, "pacing_mode": "ramp_asap"},
            ))
    else:
        # Undated: taper should follow the peak promptly.
        if peak_index is not None and weeks_at_peak_onward - 1 > MAX_UNDATED_MAINTAIN_WEEKS:
            violations.append(RuleViolation(
                "undated_holds_too_long",
                f"{milestone.name} has no set date, so the taper should start once the "
                f"{target_peak:.0f} mi long run is reached; the plan holds for "
                f"{weeks_at_peak_onward - 1} extra weeks.",
                # Usually the block is stuck below the milestone's real peak
                # because a cap is holding it there, so it never triggers
                # the taper. Lifting the cap is the fix — and if the athlete
                # set that cap, this becomes their trade-off to accept.
                {"peak_long_run_cap": None},
            ))

    # --- taper present and the right length ------------------------------------
    taper = [w for w in weeks if w.block in taper_blocks]
    if not taper:
        violations.append(RuleViolation(
            "missing_taper", f"{milestone.name} has no taper block before it."))
    elif len(taper) != milestone.taper_weeks:
        violations.append(RuleViolation(
            "taper_length",
            f"{milestone.name} taper is {len(taper)} week(s); {milestone.taper_weeks} expected.",
        ))

    return violations


# Which blocks lead into, and taper for, each milestone kind.
_MILESTONE_BLOCKS = {
    "half_marathon": (("base_building",), ("hm_taper",)),
    "marathon": (("base_building", "marathon_block"), ("marathon_taper",)),
    "ultra_50k": (("ultra_build",), ("ultra_taper",)),
}


def check_plan_rules(config: Config, plan: PlanVersion, as_of: date) -> list[RuleViolation]:
    """Every milestone's shape rules, plus the training rules from
    `validation.py` promoted to violations so one report covers both."""
    violations: list[RuleViolation] = []
    for milestone in ms.upcoming_milestones(config, as_of):
        blocks = _MILESTONE_BLOCKS.get(milestone.kind or "")
        if not blocks:
            continue
        build_blocks, taper_blocks = blocks
        violations.extend(check_milestone_shape(plan.weeks, milestone, build_blocks, taper_blocks))

    marathon = ms.milestone_by_name(config, "marathon")
    if marathon and marathon.target_date:
        for issue in validate_plan(plan.weeks, marathon.target_date):
            if issue.severity != "error":
                continue
            violations.append(RuleViolation(issue.code, issue.message, _REPAIR_FOR_CODE.get(issue.code, {})))
    return violations


# How each validator error can be repaired by regenerating. A prerequisite
# that isn't met is fixed by climbing sooner — which is also exactly the
# thing an athlete who asked to slow down has declined, so tagging it here
# is what lets the loop tell "the system got it wrong" apart from "this is
# the cost of what you asked for".
_REPAIR_FOR_CODE = {
    "back_to_back_peak": {"force_maintain_oscillation": True},
    "marathon_prereq": {"ramp_rate": 1.0, "pacing_mode": "ramp_asap"},
    "hm_prereq": {"ramp_rate": 1.0, "pacing_mode": "ramp_asap"},
    "ultra_prereq": {"ramp_rate": 1.0, "pacing_mode": "ramp_asap"},
    "long_run_jump": {"ramp_rate": 1.0},
}


def generate_valid_plan(
    generate: Callable[..., PlanVersion],
    config: Config,
    anchors: Anchors,
    as_of: date,
    base_kwargs: Optional[dict] = None,
    locked: Optional[set[str]] = None,
) -> tuple[PlanVersion, RuleReport]:
    """Generate a plan and repair it until the milestone rules pass.

    `locked` names the kwargs the athlete set explicitly. The loop will
    not overwrite those to satisfy a rule; instead the resulting violation
    is downgraded to a warning and reported, so a request like "ramp up
    slower" is honoured even when it costs half a mile off the marathon
    prerequisite — and the athlete is told that it did.

    `generate` takes the same keyword arguments as
    `planner.generate_macro_plan`; each repair pass merges in the `repair`
    hints from the violations it saw and regenerates from scratch rather
    than patching week rows — a patched week is inconsistent with the
    weeks around it, which is how plans drift out of shape in the first
    place.

    Returns the best plan found together with a report. When the loop
    cannot satisfy the rules, the report carries the surviving violations
    so the caller can surface them instead of pretending the plan is fine.
    """
    kwargs = dict(base_kwargs or {})
    locked_keys = set(locked or ())
    best_plan: Optional[PlanVersion] = None
    best_violations: list[RuleViolation] = []

    def score(violations: list[RuleViolation]) -> tuple[int, int]:
        errors = sum(1 for v in violations if v.severity == "error")
        return errors, len(violations)

    passes = 0
    for attempt in range(1, MAX_REPAIR_PASSES + 1):
        passes = attempt
        plan = generate(config, anchors, as_of, **kwargs)
        stages.annotate_stages(plan.weeks, config.guardrails.backoff_cadence)
        violations = check_plan_rules(config, plan, as_of)

        if best_plan is None or score(violations) < score(best_violations):
            best_plan, best_violations = plan, violations

        if not violations:
            return plan, RuleReport([], passes_used=attempt, repaired=attempt > 1)

        # Only repairs that leave the athlete's own choices intact.
        repairs: dict = {}
        for v in violations:
            repairs.update({k: val for k, val in v.repair.items() if k not in locked_keys})
        if not repairs:
            break
        # A repair that doesn't change the inputs would regenerate the same
        # plan and spin the loop for nothing.
        if all(kwargs.get(k) == val for k, val in repairs.items()):
            break
        kwargs.update(repairs)

    assert best_plan is not None
    # The loop has now tried everything it is allowed to. Whatever survives
    # and would have needed a locked input to fix is the cost of an
    # explicit athlete choice: honour the choice, report the cost. Silently
    # overruling them is how a coach loses trust — and silently shipping
    # the consequence unmentioned is how an athlete gets hurt.
    for v in best_violations:
        if locked_keys and set(v.repair) & locked_keys:
            v.severity = "warning"
            v.message += " (accepted: this follows from what you asked for)"
            v.repair = {}
    return best_plan, RuleReport(best_violations, passes_used=passes, repaired=False)
