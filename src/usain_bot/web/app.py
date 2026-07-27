"""FastAPI app: REST endpoints over CoachService, plus the static
frontend. Every number returned here comes from the same
agent/planner/guardrails functions the CLI uses — this is a view onto
the same system, not a second implementation of it.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import agent
from ..chat import run_chat_turn
from ..chat.providers import LLMProviderError, get_llm_provider
from ..config import Config
from ..garmin_adapter.base import GarminAdapter
from ..service import CoachService
from ..storage.base import StorageBackend

logger = logging.getLogger("usain_bot.web")

STATIC_DIR = Path(__file__).parent / "static"


class HealthFlagRequest(BaseModel):
    flag: str
    note: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class RunFeelingRequest(BaseModel):
    score: int
    comment: Optional[str] = None
    activity_date: Optional[str] = None


class PushMilestoneRequest(BaseModel):
    milestone: str
    weeks: int


def create_app(config: Config, storage: StorageBackend, adapter: GarminAdapter) -> FastAPI:
    app = FastAPI(title="usain-bot")
    service = CoachService(config, storage, adapter)
    app.state.service = service
    app.state.config = config

    @app.get("/api/status")
    def status():
        plan = service.get_plan()
        last_sync = service.storage.get_last_sync_time()
        return {
            "has_plan": plan is not None,
            "plan_version": plan.version if plan else None,
            "last_sync": last_sync.isoformat() if last_sync else None,
            "athlete": {
                "available_run_days_per_week": config.athlete.available_run_days_per_week,
                "injury_history": config.athlete.injury_history,
            },
            "goals": [{"name": g.name, "distance_mi": g.distance_mi, "date": g.date} for g in config.goals],
            "chat_provider": config.chat.provider,
        }

    @app.get("/api/today")
    def get_today():
        return service.get_today_payload()

    @app.post("/api/today/refresh")
    def refresh_today():
        service.refresh_today()
        return service.get_today_payload()

    @app.get("/api/plan")
    def get_plan():
        return service.get_plan_payload()

    @app.get("/api/plan/history")
    def get_plan_history():
        return service.get_plan_history_payload()

    @app.get("/api/history")
    def get_history(days: int = 90):
        days = max(7, min(days, 365))
        classified = service.get_history(days=days)
        weekly: dict[str, float] = {}
        activities = []
        for c in sorted(classified, key=lambda c: c.activity.date):
            week_key = (c.activity.date - timedelta(days=c.activity.date.weekday())).isoformat()
            weekly[week_key] = weekly.get(week_key, 0.0) + c.activity.distance_mi
            activities.append({
                "date": c.activity.date.isoformat(),
                "type": c.activity.activity_type.value,
                "run_class": c.run_class.value,
                "distance_mi": round(c.activity.distance_mi, 2),
                "duration_s": c.activity.duration_s,
                "avg_hr": c.activity.avg_hr,
                "name": c.activity.name,
            })
        weekly_series = [{"week_start": k, "total_mi": round(v, 1)} for k, v in sorted(weekly.items())]
        return {"activities": list(reversed(activities)), "weekly_volume": weekly_series}

    @app.post("/api/sync")
    def sync():
        result = service.sync()
        return {"new_activities": result.activities_count, "live": result.live, "message": result.message}

    @app.post("/api/health-flag")
    def set_health_flag(req: HealthFlagRequest):
        from ..chat.tools import VALID_HEALTH_FLAGS

        if req.flag not in VALID_HEALTH_FLAGS:
            raise HTTPException(400, f"flag must be one of {VALID_HEALTH_FLAGS}")
        result = service.set_health_flag(req.flag, req.note)
        return {"applied": True, "active_flag": service.active_health_flag,
                "today": result.recommendation.to_dict()}

    @app.delete("/api/health-flag")
    def clear_health_flag():
        """Undo an active flag (mis-tap, or symptom resolved). The flag
        stays in history; it just stops constraining today."""
        result = service.clear_health_flag()
        return {"cleared": True, "active_flag": None, "today": result.recommendation.to_dict()}

    @app.get("/api/plan/validation")
    def plan_validation():
        return service.get_plan_validation()

    @app.get("/api/feelings")
    def get_feelings(days: int = 14):
        return service.get_recent_feelings_payload(days=max(1, min(days, 365)))

    @app.post("/api/feelings")
    def record_feeling(req: RunFeelingRequest):
        if not 1 <= req.score <= 5:
            raise HTTPException(400, "score must be between 1 and 5")
        activity_date = date.fromisoformat(req.activity_date) if req.activity_date else None
        fb = service.record_run_feeling(req.score, req.comment, activity_date)
        return {"recorded": True, "score": fb.score}

    @app.post("/api/milestone/push")
    def push_milestone(req: PushMilestoneRequest):
        result = service.push_milestone(req.milestone, req.weeks)
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result

    @app.post("/api/backfill")
    def backfill():
        """Full-history import. Long-running and rate-limit aware — see
        agent.backfill_history."""
        result = agent.backfill_history(config, storage, adapter)
        return {
            "new_count": result.new_count, "updated_count": result.updated_count,
            "chunks_fetched": result.chunks_fetched, "complete": result.complete,
            "earliest_date": result.earliest_date.isoformat() if result.earliest_date else None,
            "message": result.message,
        }

    @app.post("/api/dedupe")
    def dedupe(apply: bool = False):
        result = agent.dedupe_activities(storage, apply=apply)
        return {
            "duplicate_groups": result.groups, "duplicate_count": result.duplicate_count,
            "removed": result.removed, "applied": result.applied,
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(400, "message must not be empty")
        try:
            provider = get_llm_provider(config)
        except LLMProviderError as exc:
            raise HTTPException(503, str(exc)) from exc
        result = run_chat_turn(provider, service, req.message)
        return {"reply": result.reply, "tool_calls": result.tool_calls}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
