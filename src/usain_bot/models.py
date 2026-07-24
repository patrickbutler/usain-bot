"""Core data types shared across every module.

These are plain dataclasses/enums with no I/O and no framework
dependencies, so they can be imported by pure guardrail functions,
storage backends, and the CLI alike without pulling in the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class ActivityType(str, Enum):
    RUNNING = "running"
    CYCLING = "cycling"
    STRENGTH_TRAINING = "strength_training"
    OTHER = "other"


class RunClass(str, Enum):
    LONG = "long"
    EASY = "easy"
    QUALITY = "quality"
    RECOVERY = "recovery"
    CROSS_TRAINING = "cross_training"


@dataclass(frozen=True)
class Activity:
    """A single Garmin activity, normalized to the fields the agent reasons over."""

    activity_id: str
    date: date
    activity_type: ActivityType
    distance_mi: float
    duration_s: int
    avg_pace_min_per_mi: Optional[float] = None
    avg_hr: Optional[int] = None
    max_hr: Optional[int] = None
    elevation_gain_ft: Optional[float] = None
    name: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassifiedActivity:
    activity: Activity
    run_class: RunClass


class GapSeverity(str, Enum):
    SHORT = "short"      # < 7 days
    MEDIUM = "medium"    # 7-14 days
    LONG = "long"         # > 14 days, <= 6 weeks
    SEVERE = "severe"    # > 6 weeks


@dataclass(frozen=True)
class GapInfo:
    gap_days: int
    severity: GapSeverity
    last_run_date: Optional[date]


@dataclass(frozen=True)
class Anchors:
    """Load anchors per §4.2 — always derived from actuals, never from the plan."""

    as_of: date
    acute_load_mi: float               # trailing 7-day total running mileage
    chronic_load_mi: float             # trailing 28-day mileage / 4
    long_run_anchor_mi: float          # longest completed run in trailing 21 days
    adherence_rate: Optional[float]    # completed / planned runs, trailing 14 days
    acwr: Optional[float]              # None when chronic_load_mi == 0 (cold start)
    gap: GapInfo


@dataclass(frozen=True)
class GuardrailResult:
    """One guardrail's verdict for 'today'. The agent takes the min across all of these."""

    name: str
    max_value: Optional[float]
    reason: str
    zone: Optional[str] = None


@dataclass
class PlanWeek:
    week_number: int
    start_date: date
    block: str
    target_volume_mi: float
    long_run_mi: float
    quality_sessions: int
    is_backoff: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_number": self.week_number,
            "start_date": self.start_date.isoformat(),
            "block": self.block,
            "target_volume_mi": round(self.target_volume_mi, 2),
            "long_run_mi": round(self.long_run_mi, 2),
            "quality_sessions": self.quality_sessions,
            "is_backoff": self.is_backoff,
            "notes": self.notes,
        }


@dataclass
class PlanVersion:
    version: int
    created_at: datetime
    trigger: str
    rationale: str
    weeks: list[PlanWeek]
    diff_from_prior: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "trigger": self.trigger,
            "rationale": self.rationale,
            "diff_from_prior": self.diff_from_prior,
            "weeks": [w.to_dict() for w in self.weeks],
        }


@dataclass(frozen=True)
class ConversationEntry:
    timestamp: datetime
    role: str  # "user" | "agent"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceDoc:
    doc_id: str
    title: str
    source: str
    added_at: datetime
    content: str


@dataclass(frozen=True)
class ReferenceChunk:
    doc_id: str
    chunk_index: int
    text: str
    title: str
    score: float = 0.0


@dataclass(frozen=True)
class HealthFlag:
    timestamp: datetime
    flag: str  # "hip" | "back" | "fatigue" | ...
    note: Optional[str] = None


@dataclass
class Recommendation:
    date: date
    run_type: str
    target_distance_mi: Optional[float]
    time_on_feet_min: Optional[float]
    effort_guidance: str
    binding_constraint: str
    reasoning: list[str]
    unlock_next_time: str
    guardrail_results: list[GuardrailResult]
    conflicts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "run_type": self.run_type,
            "target_distance_mi": self.target_distance_mi,
            "time_on_feet_min": self.time_on_feet_min,
            "effort_guidance": self.effort_guidance,
            "binding_constraint": self.binding_constraint,
            "reasoning": self.reasoning,
            "unlock_next_time": self.unlock_next_time,
            "guardrail_results": [
                {"name": g.name, "max_value": g.max_value, "reason": g.reason, "zone": g.zone}
                for g in self.guardrail_results
            ],
            "conflicts": self.conflicts,
        }
