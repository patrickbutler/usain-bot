"""Shared state for the web server: an in-memory cache of "today's"
computed recommendation, so viewing the Upcoming tab (or asking the
chat about it) doesn't create a new plan version and conversation
entry on every page load — only `agent.run_invocation` does that, and
we only want to call it once per day unless the user explicitly asks
for a refresh (or a health flag changes the picture).

Both the REST endpoints (web/app.py) and the chat tool dispatch
(chat/tools.py) hold a reference to the same CoachService instance, so
they never disagree about what "today" currently looks like.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Optional

from . import agent
from .classification import classify_activities
from .config import Config
from .garmin_adapter.base import GarminAdapter
from .models import ClassifiedActivity, HealthFlag, PlanVersion
from .projection import project_next_7_days
from .storage.base import StorageBackend


class CoachService:
    def __init__(self, config: Config, storage: StorageBackend, adapter: GarminAdapter):
        self.config = config
        self.storage = storage
        self.adapter = adapter
        self._lock = threading.RLock()
        self._cached: Optional[agent.InvocationResult] = None
        self._cached_flag: Optional[str] = None

    def get_today(self, as_of: Optional[date] = None) -> agent.InvocationResult:
        as_of = as_of or date.today()
        with self._lock:
            if self._cached is not None and self._cached.recommendation.date == as_of:
                return self._cached
        return self.refresh_today(as_of=as_of)

    def refresh_today(self, as_of: Optional[date] = None, health_flag: Optional[str] = None) -> agent.InvocationResult:
        """Actually runs the §4.3 decision procedure (sync, classify,
        guardrails, plan re-projection) and persists a new plan version —
        same as `usain-bot run`. Called once per day automatically via
        get_today(), or on demand for an explicit refresh / after a
        health flag changes.

        The whole call is serialized on the service lock (not just the
        cache write): two concurrent requests both computing "next plan
        version" from the same prior version before either has saved
        would otherwise race and collide on the version primary key.
        """
        as_of = as_of or date.today()
        with self._lock:
            flag = health_flag if health_flag is not None else self._cached_flag
            result = agent.run_invocation(self.config, self.storage, self.adapter, as_of=as_of, health_flag=flag)
            self._cached = result
            if health_flag is not None:
                self._cached_flag = health_flag
        return result

    def apply_plan_update(self, new_plan: PlanVersion) -> None:
        """Point the cached "today" view at a newly-saved plan version
        without re-running the full decision procedure.

        This matters: `refresh_today()` calls `agent.run_invocation`,
        which *always* regenerates the plan fresh from current anchors
        (per planner.py's "rolling projection" design) — calling it right
        after saving a conversational override would silently regenerate
        over the override in the same breath, discarding the very change
        that was just applied. Today's already-computed recommendation
        doesn't need to change just because a future week was edited, so
        this only swaps the plan on the cached result, nothing else.
        """
        with self._lock:
            if self._cached is not None:
                self._cached = replace(self._cached, plan=new_plan)

    def set_health_flag(self, flag: str, note: Optional[str] = None) -> agent.InvocationResult:
        self.storage.save_health_flag(HealthFlag(timestamp=datetime.utcnow(), flag=flag, note=note))
        return self.refresh_today(health_flag=flag)

    def clear_health_flag(self) -> None:
        with self._lock:
            self._cached_flag = None

    def get_today_payload(self, as_of: Optional[date] = None) -> dict:
        """Single source of truth for "today" as JSON — used by both the
        REST API (GET /api/today) and the get_today_recommendation chat
        tool, so they can never disagree on shape or content."""
        result = self.get_today(as_of)
        rec = result.recommendation
        plan = result.plan
        next7 = project_next_7_days(
            rec.date, self.config.athlete.available_run_days_per_week,
            plan.weeks[0].quality_sessions, plan.weeks[0].is_backoff, rec,
        )
        return {
            "recommendation": rec.to_dict(),
            "next_7_days": [d.to_dict() for d in next7],
            "anchors": {
                "acute_load_mi": round(result.anchors.acute_load_mi, 1),
                "chronic_load_mi": round(result.anchors.chronic_load_mi, 1),
                "long_run_anchor_mi": round(result.anchors.long_run_anchor_mi, 1),
                "acwr": None if result.anchors.acwr is None else round(result.anchors.acwr, 2),
                "gap_days": result.anchors.gap.gap_days,
                "gap_severity": result.anchors.gap.severity.value,
            },
            "sync": {"live": result.sync.live, "message": result.sync.message},
            "plan_version": plan.version,
        }

    def get_plan(self) -> Optional[PlanVersion]:
        return self.storage.get_latest_plan_version()

    def get_plan_payload(self) -> dict:
        plan = self.get_plan()
        if plan is None:
            return {"error": "No plan exists yet. Run `usain-bot init` first."}
        milestone = self.config.goal("half_marathon_benchmark")
        capable_week = None
        if milestone:
            capable_week = next((w.to_dict() for w in plan.weeks if w.long_run_mi >= milestone.distance_mi), None)
        return {
            "plan_version": plan.version,
            "trigger": plan.trigger,
            "rationale": plan.rationale,
            "weeks": [w.to_dict() for w in plan.weeks],
            "half_marathon_capability_week": capable_week,
        }

    def get_plan_history(self) -> list[PlanVersion]:
        return self.storage.get_plan_history()

    def get_history(self, days: int = 90) -> list[ClassifiedActivity]:
        activities = self.storage.get_activities()
        classified = classify_activities(activities)
        cutoff = date.today() - timedelta(days=days)
        return [c for c in classified if c.activity.date >= cutoff]

    def sync(self, as_of: Optional[date] = None) -> agent.SyncResult:
        return agent.sync_activities(self.storage, self.adapter, as_of or date.today())
