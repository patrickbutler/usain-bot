"""FastAPI app: REST endpoints over CoachService, plus the static
frontend. Every number returned here comes from the same
agent/planner/guardrails functions the CLI uses — this is a view onto
the same system, not a second implementation of it.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
        return {
            "versions": [
                {
                    "version": pv.version, "created_at": pv.created_at.isoformat(),
                    "trigger": pv.trigger, "rationale": pv.rationale, "diff_from_prior": pv.diff_from_prior,
                }
                for pv in service.get_plan_history()
            ]
        }

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
        return {"applied": True, "today": result.recommendation.to_dict()}

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
