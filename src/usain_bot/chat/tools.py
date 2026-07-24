"""Tool definitions + dispatch for the chat UI.

This is the enforcement point for the whole system's core rule even in
free-form conversation: the LLM never computes a mileage, date, or
guardrail value itself — it only ever picks one of these tools and
supplies arguments; every number in every tool result comes from
agent.py/planner.py/guardrails.py, the same functions the CLI uses. If a
tool can't answer something, the system prompt tells the model to say so
rather than estimate.

Nothing here is Anthropic-specific — `ToolSpec`/dispatch work against
the provider-agnostic types in chat/providers/base.py, so this file
doesn't change if the provider does.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Callable

from .. import planner
from ..service import CoachService
from .providers.base import ToolSpec

VALID_HEALTH_FLAGS = ("hip", "back", "fatigue")


def tool_get_today_recommendation(service: CoachService, input: dict) -> dict:
    return service.get_today_payload(date.today())


def tool_get_plan_overview(service: CoachService, input: dict) -> dict:
    return service.get_plan_payload()


def tool_get_plan_history(service: CoachService, input: dict) -> dict:
    payload = service.get_plan_history_payload()
    limit = int(input.get("limit", 10))
    payload["versions"] = payload["versions"][:max(1, min(limit, 50))]
    return payload


def tool_get_run_history(service: CoachService, input: dict) -> dict:
    days = int(input.get("days", 90))
    days = max(7, min(days, 365))
    classified = service.get_history(days=days)
    by_class: dict[str, int] = {}
    total_mi = 0.0
    runs = []
    for c in sorted(classified, key=lambda c: c.activity.date, reverse=True):
        by_class[c.run_class.value] = by_class.get(c.run_class.value, 0) + 1
        total_mi += c.activity.distance_mi
        runs.append({
            "date": c.activity.date.isoformat(),
            "type": c.activity.activity_type.value,
            "run_class": c.run_class.value,
            "distance_mi": round(c.activity.distance_mi, 2),
            "avg_hr": c.activity.avg_hr,
            "name": c.activity.name,
        })
    return {
        "window_days": days,
        "total_activities": len(runs),
        "total_distance_mi": round(total_mi, 1),
        "count_by_class": by_class,
        "activities": runs[:100],
    }


def tool_ease_upcoming_week(service: CoachService, input: dict) -> dict:
    plan = service.get_plan()
    if plan is None:
        return {"error": "No plan exists yet."}
    week_number = input.get("week_number")
    factor = float(input.get("factor", planner.OVERRIDE_EASIER_WEEK_FACTOR))
    as_of = date.today()
    result = planner.ease_week(plan, week_number, as_of, "requested via chat", factor=factor)
    return _apply_override_result(service, result)


def tool_shift_marathon_date(service: CoachService, input: dict) -> dict:
    plan = service.get_plan()
    if plan is None:
        return {"error": "No plan exists yet."}
    weeks = int(input["weeks"])
    today = service.get_today()
    result = planner.shift_marathon(plan, service.config, today.anchors, date.today(), weeks, "requested via chat")
    return _apply_override_result(service, result)


def tool_set_long_run_day_preference(service: CoachService, input: dict) -> dict:
    plan = service.get_plan()
    if plan is None:
        return {"error": "No plan exists yet."}
    day = str(input["day"])
    result = planner.note_long_run_day(plan, day, "requested via chat")
    return _apply_override_result(service, result)


def tool_set_health_flag(service: CoachService, input: dict) -> dict:
    flag = str(input["flag"]).lower()
    if flag not in VALID_HEALTH_FLAGS:
        return {"error": f"Unknown flag '{flag}'. Must be one of {VALID_HEALTH_FLAGS}."}
    note = input.get("note")
    result = service.set_health_flag(flag, note)
    return {
        "flag_set": flag, "note": note,
        "today_recommendation": result.recommendation.to_dict(),
        "message": "Flag persisted and today's recommendation recomputed on the conservative branch.",
    }


def tool_search_coaching_references(service: CoachService, input: dict) -> dict:
    query = str(input["query"])
    hits = service.storage.search_references(query, top_k=int(input.get("top_k", 3)))
    if not hits:
        return {"results": [], "note": "No matching references indexed. The athlete can add some with `usain-bot reference add <file>`."}
    return {"results": [{"title": h.title, "text": h.text, "score": h.score} for h in hits]}


def tool_trigger_garmin_sync(service: CoachService, input: dict) -> dict:
    sync_result = service.sync()
    return {"new_activities": sync_result.activities_count, "live": sync_result.live, "message": sync_result.message}


def _apply_override_result(service: CoachService, result: planner.OverrideResult) -> dict:
    if not result.applied or result.plan is None:
        return {"applied": False, "warnings": result.warnings}
    result.plan.diff_from_prior = planner.diff_plan_versions(service.get_plan(), result.plan)
    service.storage.save_plan_version(result.plan)
    service.apply_plan_update(result.plan)
    return {
        "applied": True, "rationale": result.rationale, "warnings": result.warnings,
        "new_plan_version": result.plan.version, "diff": result.plan.diff_from_prior,
    }


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_today_recommendation",
        description=(
            "Get today's run recommendation: type, distance, effort guidance, the binding "
            "guardrail constraint, and a rolling 7-day projection. Always call this before "
            "answering any question about what to run today or this week."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_plan_overview",
        description=(
            "Get the full current training plan: every week's block, target volume, long run, "
            "quality sessions, and back-off flag, plus the projected half-marathon-capability week. "
            "Call this for any question about the overall plan, upcoming weeks, or milestones."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_plan_history",
        description=(
            "Get the version history of the plan itself: every past change (daily reprojections, "
            "gap-protocol adjustments, overrides), when it happened, why (trigger + rationale), and "
            "which weeks actually changed vs. the version before it. Use this for any question about "
            "how or why the plan has changed, e.g. 'why is my long run different from last week' or "
            "'what changed after I asked to ease up'."
        ),
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max versions to return, newest first, default 10, max 50"}},
        },
    ),
    ToolSpec(
        name="get_run_history",
        description="Get the athlete's classified run history (long/easy/quality/recovery/cross-training) over a trailing window, with per-run detail and aggregate counts/mileage.",
        input_schema={
            "type": "object",
            "properties": {"days": {"type": "integer", "description": "Trailing window in days, default 90, max 365"}},
        },
    ),
    ToolSpec(
        name="ease_upcoming_week",
        description="Reduce a specific (or, if omitted, the next upcoming) week's volume and long run by a factor and mark it a manual back-off. Use when the athlete asks to make a week easier/lighter.",
        input_schema={
            "type": "object",
            "properties": {
                "week_number": {"type": "integer", "description": "Specific week number to ease; omit for the next upcoming week"},
                "factor": {"type": "number", "description": "Fraction of original volume to keep, 0.3-0.95, default 0.75"},
            },
        },
    ),
    ToolSpec(
        name="shift_marathon_date",
        description="Shift the marathon date by N weeks (positive = later, negative = earlier) and regenerate the downstream arc (taper, recovery, ultra block start) from it.",
        input_schema={
            "type": "object",
            "properties": {"weeks": {"type": "integer", "description": "Weeks to shift; positive is later, negative is earlier"}},
            "required": ["weeks"],
        },
    ),
    ToolSpec(
        name="set_long_run_day_preference",
        description="Record which day of the week the athlete wants their long run on. Recorded as a preference note on the plan, not a hard guardrail-checked schedule.",
        input_schema={
            "type": "object",
            "properties": {"day": {"type": "string", "description": "Day of week, e.g. 'Sunday'"}},
            "required": ["day"],
        },
    ),
    ToolSpec(
        name="set_health_flag",
        description=(
            "Force the conservative branch because of a physical symptom: caps today's run at the "
            "last completed distance, drops quality work, and suggests cross-training instead. Use "
            "whenever the athlete mentions hip, back, or general fatigue/soreness — do not wait to be "
            "asked explicitly to 'set a flag'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "flag": {"type": "string", "enum": list(VALID_HEALTH_FLAGS)},
                "note": {"type": "string", "description": "Optional free-text note on the symptom"},
            },
            "required": ["flag"],
        },
    ),
    ToolSpec(
        name="search_coaching_references",
        description="Search the athlete's indexed coaching reference articles for guidance relevant to a topic (e.g. ACWR, return-to-running). Use this to ground advice and cite a source rather than asserting exercise-science claims unsupported by the indexed references.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "description": "default 3"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="trigger_garmin_sync",
        description="Manually pull the latest Garmin activities before answering, if the athlete wants to make sure the data is fresh (e.g. just finished a run).",
        input_schema={"type": "object", "properties": {}},
    ),
]

_DISPATCH: dict[str, Callable[[CoachService, dict], dict]] = {
    "get_today_recommendation": tool_get_today_recommendation,
    "get_plan_overview": tool_get_plan_overview,
    "get_plan_history": tool_get_plan_history,
    "get_run_history": tool_get_run_history,
    "ease_upcoming_week": tool_ease_upcoming_week,
    "shift_marathon_date": tool_shift_marathon_date,
    "set_long_run_day_preference": tool_set_long_run_day_preference,
    "set_health_flag": tool_set_health_flag,
    "search_coaching_references": tool_search_coaching_references,
    "trigger_garmin_sync": tool_trigger_garmin_sync,
}


def execute_tool(name: str, input: dict, service: CoachService) -> str:
    handler = _DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        result = handler(service, input)
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, not a crash
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, default=str)
