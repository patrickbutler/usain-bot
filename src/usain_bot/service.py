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
from . import planner
from .classification import compute_anchors, prepare_classified
from .config import Config
from .garmin_adapter.base import GarminAdapter
from .models import Anchors, ClassifiedActivity, HealthFlag, PlanVersion, RunFeedback
from .projection import project_next_7_days
from .storage.base import StorageBackend
from .validation import validate_plan


class CoachService:
    def __init__(self, config: Config, storage: StorageBackend, adapter: GarminAdapter):
        self.config = config
        self.storage = storage
        self.adapter = adapter
        self._lock = threading.RLock()
        self._cached: Optional[agent.InvocationResult] = None
        self._cached_flag: Optional[str] = None
        # A proposed-but-unapproved plan revision. Held in memory only —
        # a draft must never be mistaken for the official plan.
        self._draft_plan: Optional[PlanVersion] = None
        self._draft_constraints: Optional[planner.PlanConstraints] = None

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

    def clear_health_flag(self) -> agent.InvocationResult:
        """Undo an active health flag (mis-tap, or the symptom resolved)
        and recompute today without the conservative branch. The original
        flag stays in history — it happened — but it stops constraining
        today's recommendation."""
        with self._lock:
            self._cached_flag = None
            self._cached = None
        return self.refresh_today()

    @property
    def active_health_flag(self) -> Optional[str]:
        with self._lock:
            return self._cached_flag

    # --- run feelings (subjective-state memory) ------------------------------

    def record_run_feeling(self, score: int, comment: Optional[str] = None,
                            activity_date: Optional[date] = None) -> RunFeedback:
        score = max(1, min(5, int(score)))
        fb = RunFeedback(timestamp=datetime.utcnow(), score=score, comment=comment, activity_date=activity_date)
        self.storage.save_run_feedback(fb)
        with self._lock:
            self._cached = None  # today's ceiling may now be lower
        return fb

    def get_recent_feelings_payload(self, days: int = 14) -> dict:
        feedback = self.storage.get_recent_run_feedback(days=days)
        mean = round(sum(f.score for f in feedback) / len(feedback), 2) if feedback else None
        return {
            "window_days": days,
            "count": len(feedback),
            "mean_score": mean,
            "entries": [
                {
                    "timestamp": f.timestamp.isoformat(), "score": f.score, "comment": f.comment,
                    "activity_date": f.activity_date.isoformat() if f.activity_date else None,
                }
                for f in feedback
            ],
        }

    def get_run_feelings_payload(self, days: int = 90, limit: int = 60) -> dict:
        """Every run in the window paired with its subjective score, if any.

        The join is done here rather than in the browser so the "already
        rated" rule (§ never ask about a run twice) has exactly one
        implementation: a run is rated iff a feedback entry names its date.
        """
        classified = self.get_history(days=days)
        runs = [c for c in classified if c.run_class.value != "cross_training"]
        feedback = self.storage.get_recent_run_feedback(days=days + 7, limit=500)
        by_date: dict[date, RunFeedback] = {}
        for f in feedback:
            if f.activity_date is None:
                continue
            # Newest entry for a date wins — a re-score corrects the old one.
            prior = by_date.get(f.activity_date)
            if prior is None or f.timestamp >= prior.timestamp:
                by_date[f.activity_date] = f

        entries = []
        for c in sorted(runs, key=lambda c: c.activity.date, reverse=True)[:limit]:
            fb = by_date.get(c.activity.date)
            entries.append({
                "date": c.activity.date.isoformat(),
                "run_class": c.run_class.value,
                "distance_mi": round(c.activity.distance_mi, 2),
                "name": c.activity.name,
                "score": fb.score if fb else None,
                "comment": fb.comment if fb else None,
            })
        rated = [e for e in entries if e["score"] is not None]
        return {
            "window_days": days,
            "runs": entries,
            "rated_count": len(rated),
            "unrated_count": len(entries) - len(rated),
            "mean_score": round(sum(e["score"] for e in rated) / len(rated), 2) if rated else None,
        }

    def get_unrated_recent_runs(self, days: int = 10) -> list[dict]:
        """Recent runs with no feeling logged — what the coach should ask
        about. Matched by date; a run is 'rated' if any feedback entry
        names that date, or was recorded after the run happened on a day
        with no explicit activity_date."""
        classified = self.get_history(days=days)
        runs = [c for c in classified if c.run_class.value != "cross_training"]
        feedback = self.storage.get_recent_run_feedback(days=days + 7, limit=100)
        rated_dates = {f.activity_date for f in feedback if f.activity_date}
        out = []
        for c in sorted(runs, key=lambda c: c.activity.date, reverse=True):
            if c.activity.date in rated_dates:
                continue
            out.append({
                "date": c.activity.date.isoformat(),
                "run_class": c.run_class.value,
                "distance_mi": round(c.activity.distance_mi, 2),
                "name": c.activity.name,
            })
        return out

    # --- draft / publish plan revisions --------------------------------------

    def propose_plan_revision(self, constraints: planner.PlanConstraints, rationale: str,
                               as_of: Optional[date] = None) -> dict:
        """Recompute the plan under new constraints and hold it as a DRAFT.

        Nothing is persisted here — the draft lives in memory until the
        athlete explicitly approves it. That's deliberate: a plan change is
        a training decision, so it gets reviewed before it becomes the
        thing the athlete trains off."""
        as_of = as_of or date.today()
        current = self.get_plan()
        anchors = self.get_current_anchors(as_of)
        base_hm = agent._read_int_preference(self.storage, agent.PREF_HM_DELAY_WEEKS)
        base_ultra = agent._read_int_preference(self.storage, agent.PREF_ULTRA_DELAY_WEEKS)

        draft = planner.revise_plan(
            self.config, anchors, as_of,
            version=(current.version + 1) if current else 1,
            constraints=constraints, rationale=rationale,
            base_hm_delay=base_hm, base_ultra_delay=base_ultra,
        )
        with self._lock:
            self._draft_plan = draft
            self._draft_constraints = constraints

        marathon_goal = self.config.goal("marathon")
        issues = validate_plan(
            draft.weeks, date.fromisoformat(marathon_goal.date)
        ) if marathon_goal and marathon_goal.date else []

        return {
            "draft": True,
            "requested_changes": constraints.describe(),
            "rationale": rationale,
            "summary": planner.summarize_plan(draft),
            "current_summary": planner.summarize_plan(current) if current else None,
            "diff": planner.diff_plan_versions(current, draft),
            "validation": {
                "valid": not any(i.severity == "error" for i in issues),
                "issues": [i.to_dict() for i in issues],
            },
            "next_step": (
                "This is a DRAFT and has not been saved. Show the athlete what changed and ask "
                "explicitly whether to make it official. Only call publish_draft_plan after they say yes."
            ),
        }

    def publish_draft_plan(self, approval_note: str = "") -> dict:
        """Promote the held draft to the official plan, recording the
        reasoning alongside the version for historical reference."""
        with self._lock:
            draft = self._draft_plan
            constraints = self._draft_constraints
        if draft is None:
            return {"error": "No draft plan to publish. Call propose_plan_revision first."}

        current = self.get_plan()
        draft.version = (current.version + 1) if current else 1
        draft.diff_from_prior = planner.diff_plan_versions(current, draft)
        detail = "; ".join(constraints.describe()) if constraints else ""
        draft.rationale = " ".join(filter(None, [
            draft.rationale,
            f"Athlete-approved changes: {detail}." if detail else "",
            f"Approval note: {approval_note}" if approval_note else "",
        ]))

        self.storage.save_plan_version(draft)

        # Persist the approved shape. This is what makes the change stick:
        # the plan is regenerated from anchors on every invocation, so
        # without these the next page load would silently rebuild the old
        # plan and the Upcoming tab would appear not to have changed.
        if constraints:
            if constraints.hm_delay_weeks is not None:
                self.storage.set_preference(agent.PREF_HM_DELAY_WEEKS, str(constraints.hm_delay_weeks))
            if constraints.ultra_delay_weeks is not None:
                self.storage.set_preference(agent.PREF_ULTRA_DELAY_WEEKS, str(constraints.ultra_delay_weeks))
            if constraints.pacing_mode is not None:
                self.storage.set_preference(agent.PREF_PACING_MODE, constraints.pacing_mode.value)
            if constraints.max_weeks_at_peak is not None:
                self.storage.set_preference(agent.PREF_MAX_WEEKS_AT_PEAK, str(constraints.max_weeks_at_peak))
            if constraints.peak_long_run_cap is not None:
                self.storage.set_preference(agent.PREF_PEAK_LONG_RUN_CAP, str(constraints.peak_long_run_cap))

        self.apply_plan_update(draft)
        with self._lock:
            self._draft_plan = None
            self._draft_constraints = None

        return {
            "published": True,
            "plan_version": draft.version,
            "rationale": draft.rationale,
            "diff": draft.diff_from_prior,
            "summary": planner.summarize_plan(draft),
            "message": f"Plan v{draft.version} is now official and live on the Upcoming tab.",
        }

    def discard_draft_plan(self) -> dict:
        with self._lock:
            had = self._draft_plan is not None
            self._draft_plan = None
            self._draft_constraints = None
        return {"discarded": had}

    def get_draft_plan_payload(self) -> Optional[dict]:
        with self._lock:
            draft = self._draft_plan
        if draft is None:
            return None
        return {"version_if_published": draft.version, "summary": planner.summarize_plan(draft),
                "weeks": [w.to_dict() for w in draft.weeks]}

    # --- milestone scheduling preferences ------------------------------------

    def push_milestone(self, milestone: str, weeks: int) -> dict:
        """Delay a flexible-date milestone (half marathon or 50K) by N
        weeks. Persisted as a preference so it survives the plan being
        regenerated from anchors on every invocation; the planner fills
        the extra time with normal build/back-off weeks to hold the base
        rather than going flat."""
        key = {"half_marathon": agent.PREF_HM_DELAY_WEEKS, "ultra_50k": agent.PREF_ULTRA_DELAY_WEEKS}.get(milestone)
        if key is None:
            return {"error": "milestone must be 'half_marathon' or 'ultra_50k' (the marathon date is fixed)."}
        current = int(self.storage.get_preference(key) or 0)
        new_value = max(0, current + int(weeks))
        self.storage.set_preference(key, str(new_value))
        result = self.refresh_today()
        return {
            "milestone": milestone,
            "delay_weeks_total": new_value,
            "message": (
                f"{milestone.replace('_', ' ').title()} pushed by {weeks} week(s) "
                f"(total delay now {new_value}). Plan regenerated — the extra weeks are filled with "
                "normal build/back-off weeks so the base is maintained."
            ),
            "plan_version": result.plan.version,
        }

    def get_plan_validation(self) -> dict:
        plan = self.get_plan()
        if plan is None:
            return {"error": "No plan exists yet."}
        marathon_goal = self.config.goal("marathon")
        if not marathon_goal or not marathon_goal.date:
            return {"error": "No marathon date configured to validate against."}
        issues = validate_plan(plan.weeks, date.fromisoformat(marathon_goal.date))
        return {
            "plan_version": plan.version,
            "valid": not any(i.severity == "error" for i in issues),
            "error_count": sum(1 for i in issues if i.severity == "error"),
            "warning_count": sum(1 for i in issues if i.severity == "warning"),
            "issues": [i.to_dict() for i in issues],
        }

    def get_today_payload(self, as_of: Optional[date] = None) -> dict:
        """Single source of truth for "today" as JSON — used by both the
        REST API (GET /api/today) and the get_today_recommendation chat
        tool, so they can never disagree on shape or content."""
        result = self.get_today(as_of)
        rec = result.recommendation
        plan = result.plan
        next7 = project_next_7_days(
            rec.date, result.anchors.runs_per_week or self.config.athlete.available_run_days_per_week,
            plan.weeks[0].quality_sessions, plan.weeks[0].is_backoff, rec,
            long_run_was_yesterday=any(
                g.name == "rest_day_after_long_run" for g in rec.guardrail_results
            ),
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
            "active_health_flag": self.active_health_flag,
            "runs_per_week_actual": result.anchors.runs_per_week,
            "unrated_recent_runs": self.get_unrated_recent_runs(),
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
            "validation": self.get_plan_validation(),
        }

    def get_plan_history(self) -> list[PlanVersion]:
        return self.storage.get_plan_history()

    def get_plan_history_payload(self) -> dict:
        """The full lineage of the plan, newest first: every version with
        its trigger, rationale, the stored text diff, and a structured
        per-week diff against the immediately preceding version (computed
        on the fly from the already-loaded weeks — nothing extra to store
        or keep in sync)."""
        versions = self.storage.get_plan_history()
        entries = []
        for i, pv in enumerate(versions):
            prior = versions[i - 1] if i > 0 else None
            week_diffs = planner.diff_plan_weeks(prior, pv)
            changed_weeks = [d for d in week_diffs if d.change_type != "unchanged"]
            entries.append({
                "version": pv.version,
                "created_at": pv.created_at.isoformat(),
                "trigger": pv.trigger,
                "rationale": pv.rationale,
                "diff_text": pv.diff_from_prior,
                "week_diffs": [d.to_dict() for d in changed_weeks],
                "weeks_changed_count": len(changed_weeks),
            })
        entries.reverse()  # newest first
        return {"versions": entries}

    def get_current_anchors(self, as_of: Optional[date] = None) -> Anchors:
        """Read-only anchors (classify stored activities, compute load) —
        no plan regeneration, no persistence. Use this instead of
        get_today()/refresh_today() whenever a caller just needs anchors
        (e.g. to feed a plan mutation like shift_marathon) rather than a
        full "today" recommendation; get_today() runs the whole decision
        procedure and can persist a new plan version as a side effect,
        which would race with a caller that already fetched a plan
        version and is about to save its own edit to it."""
        as_of = as_of or date.today()
        activities = self.storage.get_activities()
        classified = prepare_classified(activities, self.config.sync.merge_gap_hours)
        planned_runs_14d = self.config.athlete.available_run_days_per_week * 2
        return compute_anchors(classified, as_of, planned_runs_trailing_14d=planned_runs_14d)

    def get_history(self, days: int = 90) -> list[ClassifiedActivity]:
        activities = self.storage.get_activities()
        classified = prepare_classified(activities, self.config.sync.merge_gap_hours)
        cutoff = date.today() - timedelta(days=days)
        return [c for c in classified if c.activity.date >= cutoff]

    def sync(self, as_of: Optional[date] = None) -> agent.SyncResult:
        return agent.sync_activities(self.storage, self.adapter, as_of or date.today(),
                                      overlap_days=self.config.sync.overlap_days)
