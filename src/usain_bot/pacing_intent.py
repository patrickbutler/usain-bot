"""Turning pacing language into plan constraints, deterministically.

"Ramp up slower", "let's cool down", "ease off a bit" all mean something
precise about the plan — the long run should climb more gradually — but
they arrive as prose. The risk is that the LLM hears one of these,
replies sympathetically, and either changes nothing or invents its own
numbers. Neither is acceptable: the athlete asked for a different plan
and should get one, computed the same way every other plan is computed.

So the mapping from phrase to constraint lives here, in Python, and is
unit-tested. `interpret_pacing_request` is used two ways:

1. As a **safety net** — the chat layer runs it on every incoming message,
   so a pacing request is recognised even if the model doesn't call the
   tool.
2. As **governance on the tool call** — when the model does call
   `propose_plan_revision` with a free-text intent, the intent is resolved
   through the same table, so the model chooses *which* knob to turn but
   never *how far*.

`ramp_rate` is a multiplier on the climb rate: 0.6 means the long run
takes roughly 1/0.6 as many weeks to reach the peak. It can slow the
climb freely, but it can never push past the §5.1 guardrail increment —
"build me up faster" gets the guardrail maximum, not more than it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .planner import PacingMode, PlanConstraints

# Ramp multipliers, from "barely slower" to "practically flat".
RAMP_GENTLE = 0.75
RAMP_SLOW = 0.6
RAMP_MUCH_SLOWER = 0.4
RAMP_FASTER = 1.4

# Long-run cap applied when the athlete asks to back right off.
COOL_DOWN_LONG_RUN_CAP_MI = 15.0


@dataclass(frozen=True)
class PacingIntent:
    """A recognised pacing request and what it does to the plan."""

    matched_phrase: str
    intent: str                     # slow_down | cool_down | speed_up | smooth | hold
    constraints: PlanConstraints
    explanation: str

    def to_dict(self) -> dict:
        return {
            "matched_phrase": self.matched_phrase,
            "intent": self.intent,
            "explanation": self.explanation,
            "changes": self.constraints.describe(),
        }


# Ordered most-specific first: "cool down" implies a bigger change than a
# plain "slower", and "much slower" beats "slower".
_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, intent, human explanation)
    (r"\b(a lot|much|way|significantly)\s+(slower|more gradual|gentler)\b", "slow_down",
     "Ramping the long run up much more gradually."),
    (r"\bcool[\s-]?down\b|\bback right off\b|\bdial (it|things) (right )?back\b", "cool_down",
     "Cooling down: long run capped below the high-mileage line and the ramp slowed while you recover."),
    (r"\b(ramp|build|increase|scale)[\w\s]{0,12}\b(slower|more slowly|more gradual\w*|gentler|gently)\b",
     "slow_down", "Ramping the long run up more gradually."),
    (r"\b(slower|more gradual\w*|gentler)\s+(ramp|build|progression|increase)\b", "slow_down",
     "Ramping the long run up more gradually."),
    (r"\bease (off|up|back)\b|\btake it easy\b|\bgo easier\b|\bless aggressive\b", "slow_down",
     "Easing the progression off — smaller week-over-week increases."),
    (r"\btoo (fast|aggressive|much too soon)\b|\bramping up too\b", "slow_down",
     "Slowing the ramp down — the current progression is too aggressive."),
    (r"\b(ramp|build|increase)[\w\s]{0,12}\b(faster|quicker|more aggressive\w*)\b", "speed_up",
     "Building as fast as the guardrails safely allow."),
    (r"\bspeed (up|things up)\b|\bpush harder\b|\bmore aggressive\b", "speed_up",
     "Building as fast as the guardrails safely allow."),
    (r"\bsmooth(\s+out)?\b|\bmore even\b|\bspread (it|the ramp) out\b", "smooth",
     "Smoothing the ramp across the weeks available before the milestone."),
    (r"\bhold (steady|where i am)\b|\bmaintain\b|\bstay (here|flat)\b|\bstop increasing\b", "hold",
     "Holding current mileage: maintain stage rather than continuing to climb."),
]

_INTENT_CONSTRAINTS = {
    "slow_down": PlanConstraints(ramp_rate=RAMP_SLOW, pacing_mode=PacingMode.MILESTONE_SMOOTHED),
    "cool_down": PlanConstraints(ramp_rate=RAMP_MUCH_SLOWER, pacing_mode=PacingMode.MILESTONE_SMOOTHED,
                                  peak_long_run_cap=COOL_DOWN_LONG_RUN_CAP_MI),
    "speed_up": PlanConstraints(ramp_rate=RAMP_FASTER, pacing_mode=PacingMode.RAMP_ASAP),
    "smooth": PlanConstraints(ramp_rate=RAMP_GENTLE, pacing_mode=PacingMode.MILESTONE_SMOOTHED),
    "hold": PlanConstraints(ramp_rate=RAMP_MUCH_SLOWER, pacing_mode=PacingMode.MILESTONE_SMOOTHED),
}

VALID_INTENTS = tuple(_INTENT_CONSTRAINTS)


def interpret_pacing_request(message: str) -> Optional[PacingIntent]:
    """The pacing intent in a message, or None if there isn't one.

    Deliberately conservative: an unrecognised phrase returns None so the
    coach asks rather than guessing at a number.
    """
    if not message:
        return None
    text = message.lower()
    for pattern, intent, explanation in _PATTERNS:
        match = re.search(pattern, text)
        if match:
            # "much slower" is caught by the first pattern; downgrade the
            # generic slow_down to the stronger multiplier.
            constraints = _INTENT_CONSTRAINTS[intent]
            if intent == "slow_down" and re.search(_PATTERNS[0][0], text):
                constraints = PlanConstraints(ramp_rate=RAMP_MUCH_SLOWER,
                                               pacing_mode=PacingMode.MILESTONE_SMOOTHED)
            return PacingIntent(match.group(0).strip(), intent, constraints, explanation)
    return None


def constraints_for_intent(intent: str) -> Optional[PlanConstraints]:
    """The constraints a named intent maps to. This is the governance
    path: the model picks the intent, the numbers come from here."""
    return _INTENT_CONSTRAINTS.get(intent)


def merge_constraints(base: PlanConstraints, extra: PlanConstraints) -> PlanConstraints:
    """Overlay explicitly-set fields of `extra` onto `base`.

    Explicit numbers from the athlete ("cap me at 18") outrank whatever a
    phrase implied, so the caller passes the phrase-derived constraints as
    `base` and the explicit ones as `extra`.
    """
    return PlanConstraints(
        pacing_mode=extra.pacing_mode or base.pacing_mode,
        max_weeks_at_peak=extra.max_weeks_at_peak if extra.max_weeks_at_peak is not None else base.max_weeks_at_peak,
        peak_long_run_cap=(extra.peak_long_run_cap if extra.peak_long_run_cap is not None
                           else base.peak_long_run_cap),
        hm_delay_weeks=extra.hm_delay_weeks if extra.hm_delay_weeks is not None else base.hm_delay_weeks,
        ultra_delay_weeks=(extra.ultra_delay_weeks if extra.ultra_delay_weeks is not None
                           else base.ultra_delay_weeks),
        run_days_per_week=(extra.run_days_per_week if extra.run_days_per_week is not None
                           else base.run_days_per_week),
        ramp_rate=extra.ramp_rate if abs(extra.ramp_rate - 1.0) > 1e-6 else base.ramp_rate,
    )
