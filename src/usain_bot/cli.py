"""Command-line entrypoint. `usain-bot run` is the everyday command,
invoked manually before a run; `init` does the first-run flow; `override`
applies a conversational plan change; `plan` and `sync` are utilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from . import agent
from .chat.providers import get_required_env_var
from .config import Config, GarminCredentials, load_config
from .garmin_adapter.base import GarminAdapter
from .garmin_adapter.live import GarminConnectAdapter
from .garmin_adapter.mock import MockGarminAdapter
from .models import ReferenceDoc
from .projection import DayProjection, project_next_7_days
from .storage import get_storage_backend

VALID_FLAGS = ("hip", "back", "fatigue")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_adapter(mock_fixture: Optional[str]) -> GarminAdapter:
    if mock_fixture:
        return MockGarminAdapter(mock_fixture)
    return GarminConnectAdapter(GarminCredentials.from_env())


def _format_text_output(result: "agent.InvocationResult", config: Config, next7: list[DayProjection]) -> str:
    rec = result.recommendation
    anchors = result.anchors
    plan = result.plan
    lines = []
    lines.append("=" * 70)
    lines.append(f"TODAY'S RECOMMENDATION — {rec.date.isoformat()}")
    lines.append("=" * 70)
    lines.append(f"Run type:            {rec.run_type}")
    dist_str = f"{rec.target_distance_mi:.2f} mi" if rec.target_distance_mi is not None else "n/a"
    lines.append(f"Target distance:     {dist_str}")
    if rec.time_on_feet_min:
        lines.append(f"Time on feet target: {rec.time_on_feet_min:.0f} min (distance is secondary beyond ~16 mi)")
    lines.append(f"Effort/pace:         {rec.effort_guidance}")
    lines.append(f"Binding constraint:  {rec.binding_constraint}")
    if rec.conflicts:
        lines.append("Conflicting signals: " + " | ".join(rec.conflicts))
    lines.append(f"What unlocks more:   {rec.unlock_next_time}")
    lines.append("")
    lines.append("Reasoning:")
    for r in rec.reasoning:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append(f"Anchors as of {anchors.as_of.isoformat()}: acute={anchors.acute_load_mi:.1f} mi, "
                  f"chronic={anchors.chronic_load_mi:.1f} mi/wk, long-run anchor={anchors.long_run_anchor_mi:.1f} mi, "
                  f"gap={anchors.gap.gap_days}d ({anchors.gap.severity.value})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("ROLLING FORWARD VIEW")
    lines.append("-" * 70)
    lines.append("Next 7 days:")
    for day in next7:
        dist = f"{day.distance_mi:.1f} mi" if day.distance_mi else ""
        lines.append(f"  {day.date.isoformat()} ({day.weekday:<9}) {day.run_type:<20} {dist} {day.note}")
    lines.append("")
    lines.append("Remainder of the plan, week by week:")
    for w in plan.weeks:
        flag = " [back-off]" if w.is_backoff else ""
        lines.append(
            f"  wk{w.week_number:>3} {w.start_date.isoformat()} {w.block:<28} "
            f"vol={w.target_volume_mi:6.1f}mi LR={w.long_run_mi:5.1f}mi q={w.quality_sessions}{flag}"
        )
        if w.notes:
            lines.append(f"        note: {w.notes}")
    lines.append("")
    milestone = config.goal("half_marathon_benchmark")
    if milestone:
        capable_week = next((w for w in plan.weeks if w.long_run_mi >= milestone.distance_mi), None)
        if capable_week:
            lines.append(
                f"Half-marathon capability ({milestone.distance_mi} mi long run): "
                f"projected week {capable_week.week_number}, {capable_week.start_date.isoformat()}."
            )
    lines.append("=" * 70)
    return "\n".join(lines)


def _write_json_output(result: "agent.InvocationResult", out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"recommendation_{result.recommendation.date.isoformat()}.json"
    payload = {
        "recommendation": result.recommendation.to_dict(),
        "plan": result.plan.to_dict(),
        "anchors": {
            "as_of": result.anchors.as_of.isoformat(),
            "acute_load_mi": result.anchors.acute_load_mi,
            "chronic_load_mi": result.anchors.chronic_load_mi,
            "long_run_anchor_mi": result.anchors.long_run_anchor_mi,
            "adherence_rate": result.anchors.adherence_rate,
            "acwr": result.anchors.acwr,
            "gap_days": result.anchors.gap.gap_days,
            "gap_severity": result.anchors.gap.severity.value,
        },
        "sync": {"live": result.sync.live, "message": result.sync.message},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    adapter_ = _build_adapter(args.mock_fixture)

    result = agent.run_invocation(
        config, storage, adapter_,
        as_of=date.fromisoformat(args.date) if args.date else None,
        health_flag=args.flag, dry_run=args.dry_run,
    )

    next7 = project_next_7_days(
        result.recommendation.date, config.athlete.available_run_days_per_week,
        result.plan.weeks[0].quality_sessions, result.plan.weeks[0].is_backoff, result.recommendation,
    )

    print(_format_text_output(result, config, next7))
    if not result.sync.live:
        print(f"\n[!] {result.sync.message}", file=sys.stderr)

    if not args.dry_run:
        out_dir = Path(config.storage.data_dir) / "output"
        json_path = _write_json_output(result, out_dir)
        print(f"\n(machine-readable output written to {json_path})")
    else:
        print("\n[dry-run mode: nothing written to storage]")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    adapter_ = _build_adapter(args.mock_fixture)

    report = agent.first_run_report(config, storage, adapter_, as_of=date.fromisoformat(args.date) if args.date else None)

    print("=" * 70)
    print("FIRST-RUN REPORT")
    print("=" * 70)
    print(report.sync.message)
    a = report.anchors
    print(f"Actual weekly volume (chronic, 28d avg): {a.chronic_load_mi:.1f} mi/wk")
    print(f"Actual long-run capacity (21d anchor):   {a.long_run_anchor_mi:.1f} mi")
    print(f"Trailing 7-day acute load:                {a.acute_load_mi:.1f} mi")
    if a.acwr is not None:
        print(f"ACWR:                                     {a.acwr:.2f}")
    else:
        print("ACWR:                                     undefined (cold start)")
    if report.baseline_conflict:
        print(f"\n[!] {report.baseline_conflict}")
    for note in report.adherence_notes:
        print(f"- {note}")

    print("\n" + "-" * 70)
    print("PROPOSED MACRO PLAN (v1) — base -> 13.1 capability -> marathon block -> "
          "taper -> marathon -> recovery -> 50K block (TBD) -> 50K (TBD)")
    print("-" * 70)
    for w in report.proposed_plan.weeks:
        flag = " [back-off]" if w.is_backoff else ""
        print(f"  wk{w.week_number:>3} {w.start_date.isoformat()} {w.block:<28} "
              f"vol={w.target_volume_mi:6.1f}mi LR={w.long_run_mi:5.1f}mi q={w.quality_sessions}{flag}")
        if w.notes:
            print(f"        note: {w.notes}")

    print("\nOpen uncertainties:")
    for u in report.uncertainties:
        print(f"  - {u}")

    if args.yes:
        agent.confirm_first_run(storage, report)
        print("\nConfirmed automatically (--yes). Plan v1 persisted.")
        return 0

    answer = input("\nPersist this as plan version 1? [y/N] ").strip().lower()
    if answer == "y":
        agent.confirm_first_run(storage, report)
        print("Plan v1 persisted.")
    else:
        print("Not persisted. Re-run `usain-bot init` after adjusting config.yaml if needed.")
    return 0


def cmd_override(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    result = agent.override_plan(config, storage, args.text, as_of=date.fromisoformat(args.date) if args.date else None)

    if result.applied:
        print(f"Override applied: {result.rationale}")
        if result.warnings:
            print("\nFlags on this change:")
            for w in result.warnings:
                print(f"  [!] {w}")
    else:
        print("Override not applied.")
        for w in result.warnings:
            print(f"  {w}")
    return 0 if result.applied else 1


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    if args.history:
        for pv in storage.get_plan_history():
            print(f"v{pv.version} ({pv.created_at.isoformat()}) trigger={pv.trigger}: {pv.rationale}")
            if pv.diff_from_prior:
                print(f"  {pv.diff_from_prior}")
        return 0

    latest = storage.get_latest_plan_version()
    if latest is None:
        print("No plan yet — run `usain-bot init` first.")
        return 1
    for w in latest.weeks:
        print(f"wk{w.week_number:>3} {w.start_date.isoformat()} {w.block:<28} "
              f"vol={w.target_volume_mi:6.1f}mi LR={w.long_run_mi:5.1f}mi q={w.quality_sessions}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    adapter_ = _build_adapter(args.mock_fixture)
    result = agent.sync_activities(storage, adapter_, date.fromisoformat(args.date) if args.date else date.today())
    print(result.message)
    return 0 if result.live else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web.app import create_app

    config = load_config(args.config)
    storage = get_storage_backend(config)
    adapter_ = _build_adapter(args.mock_fixture)
    app = create_app(config, storage, adapter_)

    print(f"usain-bot web UI at http://{args.host}:{args.port}")
    required_var = get_required_env_var(config.chat.provider)
    if required_var and not os.environ.get(required_var):
        print(f"[!] {required_var} not set for chat provider '{config.chat.provider}' — every tab except Chat will still work.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "warning")
    return 0


def cmd_reference_add(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    path = Path(args.file)
    content = path.read_text(encoding="utf-8")
    doc = ReferenceDoc(
        doc_id=args.id or f"{path.stem}-{uuid.uuid4().hex[:8]}",
        title=args.title or path.stem.replace("-", " ").replace("_", " ").title(),
        source=args.source or str(path),
        added_at=datetime.utcnow(),
        content=content,
    )
    storage.save_reference(doc)
    print(f"Saved reference '{doc.title}' (doc_id={doc.doc_id}), chunked and indexed for retrieval.")
    return 0


def cmd_reference_list(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    docs = storage.list_references()
    if not docs:
        print("No references stored yet. Add one with `usain-bot reference add <file.md>`.")
        return 0
    for d in docs:
        print(f"{d.doc_id}  {d.title!r}  source={d.source}  added={d.added_at.isoformat()}")
    return 0


def cmd_reference_search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = get_storage_backend(config)
    results = storage.search_references(args.query, top_k=args.top_k)
    if not results:
        print("No matching references.")
        return 0
    for r in results:
        print(f"[{r.title}] (score={r.score:.1f})")
        print(r.text[:300] + ("..." if len(r.text) > 300 else ""))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usain-bot", description="Local-first, agentic adaptive running coach.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Debug-level structured logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Get today's recommendation and rolling forward view")
    p_run.add_argument("--dry-run", action="store_true", help="Show the recommendation without writing to storage")
    p_run.add_argument("--flag", choices=VALID_FLAGS, help="Force the conservative branch (hip/back/fatigue)")
    p_run.add_argument("--date", help="Override 'today' (ISO date) — mainly for testing")
    p_run.add_argument("--mock-fixture", help="Path to a mock Garmin activities JSON fixture (skips live Garmin)")
    p_run.set_defaults(func=cmd_run)

    p_init = sub.add_parser("init", help="First-run flow: pull history, propose macro plan, confirm before persisting")
    p_init.add_argument("--yes", action="store_true", help="Auto-confirm and persist plan v1 without prompting")
    p_init.add_argument("--date", help="Override 'today' (ISO date)")
    p_init.add_argument("--mock-fixture", help="Path to a mock Garmin activities JSON fixture")
    p_init.set_defaults(func=cmd_init)

    p_override = sub.add_parser("override", help="Apply a conversational plan change")
    p_override.add_argument("text", help="e.g. \"make next week easier\"")
    p_override.add_argument("--date", help="Override 'today' (ISO date)")
    p_override.set_defaults(func=cmd_override)

    p_plan = sub.add_parser("plan", help="Show the current (or historical) plan")
    p_plan.add_argument("--history", action="store_true", help="Show full plan version lineage instead of the latest")
    p_plan.set_defaults(func=cmd_plan)

    p_sync = sub.add_parser("sync", help="Sync Garmin activities without computing a recommendation")
    p_sync.add_argument("--date", help="Override 'today' (ISO date)")
    p_sync.add_argument("--mock-fixture", help="Path to a mock Garmin activities JSON fixture")
    p_sync.set_defaults(func=cmd_sync)

    p_ref = sub.add_parser("reference", help="Manage coaching reference articles/notes")
    ref_sub = p_ref.add_subparsers(dest="reference_command", required=True)

    p_ref_add = ref_sub.add_parser("add", help="Add a reference doc (chunked + indexed for retrieval)")
    p_ref_add.add_argument("file", help="Path to a markdown/text file")
    p_ref_add.add_argument("--title", help="Doc title (defaults to filename)")
    p_ref_add.add_argument("--source", help="Where this came from (URL, citation, etc.)")
    p_ref_add.add_argument("--id", help="Explicit doc_id (defaults to a generated one)")
    p_ref_add.set_defaults(func=cmd_reference_add)

    p_ref_list = ref_sub.add_parser("list", help="List stored reference docs")
    p_ref_list.set_defaults(func=cmd_reference_list)

    p_ref_search = ref_sub.add_parser("search", help="Search reference chunks by keyword")
    p_ref_search.add_argument("query")
    p_ref_search.add_argument("--top-k", type=int, default=3)
    p_ref_search.set_defaults(func=cmd_reference_search)

    p_serve = sub.add_parser("serve", help="Run the local web UI (chat + upcoming + history)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8420)
    p_serve.add_argument("--mock-fixture", help="Path to a mock Garmin activities JSON fixture (for local testing)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
