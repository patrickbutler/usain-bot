"""Orchestrates the decision procedure from §4.3. Pulls together
garmin_adapter, classification, guardrails, planner, and storage — but
implements no guardrail math itself. Every ceiling comes from
guardrails.py; this module's job is to compute inputs, take the min,
and explain the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median
from typing import Optional

from . import guardrails as gr
from .classification import RUNNING_CLASSES, compute_anchors, prepare_classified
from .config import Config
from .garmin_adapter.base import GarminAdapter, GarminUnavailableError
from .ingest import IngestReport, find_duplicate_groups, sanitize_activities
from .models import (
    Activity,
    Anchors,
    ClassifiedActivity,
    ConversationEntry,
    GapSeverity,
    GuardrailResult,
    HealthFlag,
    PlanVersion,
    Recommendation,
    RunClass,
)
from .planner import OverrideResult, apply_override, diff_plan_versions, generate_macro_plan
from .storage.base import StorageBackend

logger = logging.getLogger("usain_bot.agent")

FIRST_RUN_LOOKBACK_WEEKS = 12
HEALTH_FLAG_CAP_FACTOR = 0.85

# Durable athlete choices, persisted outside the plan (which regenerates
# from anchors every invocation and would otherwise forget them).
PREF_HM_DELAY_WEEKS = "hm_delay_weeks"
PREF_ULTRA_DELAY_WEEKS = "ultra_delay_weeks"

# Subjective-state bias: when recent runs have felt bad, cap today below
# what the physical guardrails alone would allow. 1=awful .. 5=great.
FEELING_ROUGH_THRESHOLD = 2.5   # mean at/below this => hold at last completed distance
FEELING_MEH_THRESHOLD = 3.2     # mean at/below this => mild trim
FEELING_ROUGH_FACTOR = 0.85
FEELING_MEH_FACTOR = 0.95
FEELING_LOOKBACK_DAYS = 14


def _read_int_preference(storage: StorageBackend, key: str) -> int:
    raw = storage.get_preference(key)
    try:
        return max(0, int(raw)) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


@dataclass
class SyncResult:
    activities_count: int          # newly inserted
    live: bool
    message: str
    updated_count: int = 0         # existing rows corrected (edited on Garmin)
    rejected_count: int = 0        # dropped by the ingest quality gate
    repaired_count: int = 0        # fields nulled/recomputed at ingest
    deduped_count: int = 0         # duplicate rows removed automatically


def _ingest(storage: StorageBackend, activities: list[Activity], as_of: date) -> tuple[int, int, IngestReport, int]:
    """The one write path into storage. Validates and repairs first, saves,
    then removes any duplicates the new rows created. Every caller
    (incremental sync, backfill chunk) goes through here so data quality
    is a property of ingestion, not a command the athlete has to run."""
    report = sanitize_activities(activities, today=as_of)
    new_count, updated_count = storage.save_activities(report.accepted)
    removed = _dedupe_stored(storage)
    return new_count, updated_count, report, removed


def _dedupe_stored(storage: StorageBackend) -> int:
    """Collapse same-run-different-id duplicates, keeping the richest
    record. Runs after every write; a no-op on clean data."""
    groups = find_duplicate_groups(storage.get_activities())
    losers = [a.activity_id for group in groups for a in group[1:]]
    if not losers:
        return 0
    logger.info("Removing %d duplicate activity row(s): %s", len(losers), losers)
    return storage.delete_activities(losers)


def sync_activities(
    storage: StorageBackend,
    adapter: GarminAdapter,
    as_of: date,
    lookback_weeks: int = FIRST_RUN_LOOKBACK_WEEKS,
    overlap_days: int = 30,
) -> SyncResult:
    """Incremental sync. Always re-pulls a trailing *overlap window* (not
    just "everything since last sync") so activities the athlete edited
    on Garmin after they were first synced get their corrections picked
    up — storage.save_activities upserts, so a changed distance/duration/
    name updates the existing row instead of being ignored as a dupe.

    Validation and deduplication happen automatically here (see
    ingest.py); there is no separate command to remember.

    Degrades gracefully if Garmin is unreachable: use cached data, say
    so, bias conservative (the gap/ACWR guardrails do that naturally
    once they see less-recent data)."""
    last_sync = storage.get_last_sync_time()
    if last_sync:
        start = min(last_sync.date(), as_of) - timedelta(days=overlap_days)
    else:
        start = as_of - timedelta(weeks=lookback_weeks)

    try:
        fresh = adapter.fetch_activities(start, as_of)
        new_count, updated_count, report, deduped = _ingest(storage, fresh, as_of)
        storage.set_last_sync_time(datetime.utcnow())
        msg = f"Synced Garmin: {new_count} new, {updated_count} updated ({start} to {as_of})."
        extra = report.summary()
        if deduped:
            extra = f"{extra}, {deduped} duplicate row(s) removed" if extra else f"{deduped} duplicate row(s) removed"
        if extra:
            msg += f" Data quality: {extra}."
        return SyncResult(new_count, True, msg, updated_count,
                           report.rejected_count, report.repaired_count, deduped)
    except GarminUnavailableError as exc:
        logger.warning("Garmin unreachable, falling back to cache: %s", exc)
        return SyncResult(0, False, f"Garmin unreachable ({exc}). Using cached data — recommendation biased conservative.")


@dataclass
class BackfillResult:
    new_count: int
    updated_count: int
    chunks_fetched: int
    earliest_date: Optional[date]
    complete: bool
    message: str
    rejected_count: int = 0
    repaired_count: int = 0
    deduped_count: int = 0


def backfill_history(
    config: Config,
    storage: StorageBackend,
    adapter: GarminAdapter,
    as_of: Optional[date] = None,
    max_years: int = 15,
    sleep_fn=None,
) -> BackfillResult:
    """One-time full-history pull, walking backwards in chunks.

    Garmin rate-limits aggressively (a single wide request is what
    returned 429), so this requests a bounded window at a time and pauses
    between chunks; the adapter itself additionally retries with long
    backoff on 429. Stops after `backfill_max_empty_chunks` consecutive
    empty windows (assumed past the start of history) or `max_years`.

    Safe to re-run: save_activities upserts, so re-pulled activities are
    deduped by activity_id rather than duplicated. Validation and
    duplicate removal run automatically per chunk (see ingest.py)."""
    import time

    sleep_fn = sleep_fn or time.sleep
    as_of = as_of or date.today()
    chunk_days = config.sync.backfill_chunk_days
    pause = config.sync.backfill_pause_s
    max_empty = config.sync.backfill_max_empty_chunks

    total_new = total_updated = chunks = 0
    total_rejected = total_repaired = total_deduped = 0
    empty_streak = 0
    earliest_seen: Optional[date] = None
    window_end = as_of
    hard_floor = as_of - timedelta(days=365 * max_years)

    def _quality_suffix() -> str:
        bits = []
        if total_rejected:
            bits.append(f"{total_rejected} rejected as unusable")
        if total_repaired:
            bits.append(f"{total_repaired} field(s) repaired")
        if total_deduped:
            bits.append(f"{total_deduped} duplicate row(s) removed")
        return f" Data quality: {', '.join(bits)}." if bits else ""

    while window_end > hard_floor:
        window_start = window_end - timedelta(days=chunk_days)
        try:
            fetched = adapter.fetch_activities(window_start, window_end)
        except GarminUnavailableError as exc:
            return BackfillResult(
                total_new, total_updated, chunks, earliest_seen, False,
                f"Backfill stopped early after {chunks} chunk(s): {exc}. "
                f"Imported {total_new} new / {total_updated} updated so far — safe to re-run to resume."
                + _quality_suffix(),
                total_rejected, total_repaired, total_deduped,
            )

        chunks += 1
        if fetched:
            new_c, upd_c, report, deduped = _ingest(storage, fetched, as_of)
            total_new += new_c
            total_updated += upd_c
            total_rejected += report.rejected_count
            total_repaired += report.repaired_count
            total_deduped += deduped
            if report.accepted:
                oldest = min(a.date for a in report.accepted)
                earliest_seen = oldest if earliest_seen is None else min(earliest_seen, oldest)
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= max_empty:
                break

        window_end = window_start - timedelta(days=1)
        if pause:
            sleep_fn(pause)

    storage.set_last_sync_time(datetime.utcnow())
    return BackfillResult(
        total_new, total_updated, chunks, earliest_seen, True,
        f"Backfill complete: {total_new} new, {total_updated} updated across {chunks} chunk(s)"
        + (f", earliest activity {earliest_seen.isoformat()}." if earliest_seen else " (no activities found).")
        + _quality_suffix(),
        total_rejected, total_repaired, total_deduped,
    )


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _this_week_classified(classified: list[ClassifiedActivity], as_of: date) -> list[ClassifiedActivity]:
    ws = _week_start(as_of)
    return [c for c in classified if ws <= c.activity.date < as_of]


def _recent_easy_distance(classified: list[ClassifiedActivity], as_of: date, fallback: float) -> float:
    dists = [
        c.activity.distance_mi for c in classified
        if c.run_class == RunClass.EASY and as_of - timedelta(days=27) <= c.activity.date <= as_of
        and c.activity.distance_mi > 0
    ]
    return median(dists) if dists else fallback


def decide_run_type(this_week: list[ClassifiedActivity], week_plan_quality_target: int, is_backoff: bool,
                     easy_only: bool) -> str:
    if easy_only:
        return "easy"
    has_long_this_week = any(c.run_class == RunClass.LONG for c in this_week)
    if not has_long_this_week:
        return "long"
    quality_count = sum(1 for c in this_week if c.run_class == RunClass.QUALITY)
    if not is_backoff and quality_count < min(week_plan_quality_target, 2):
        last = max(this_week, key=lambda c: c.activity.date, default=None)
        if last is None or last.run_class != RunClass.QUALITY:
            return "quality"
    return "easy"


def compute_recommendation(
    config: Config,
    storage: StorageBackend,
    classified: list[ClassifiedActivity],
    anchors: Anchors,
    plan: PlanVersion,
    as_of: date,
    health_flag: Optional[str] = None,
) -> Recommendation:
    guardrail_results: list[GuardrailResult] = []
    reasoning: list[str] = []
    conflicts: list[str] = []

    gap = anchors.gap
    action = gr.gap_action(gap.gap_days, anchors.long_run_anchor_mi)
    if gap.severity != GapSeverity.SHORT:
        reasoning.append(f"Return-from-gap protocol applied first: {action.description}")

    this_week_plan = plan.weeks[0]
    this_week = _this_week_classified(classified, as_of)
    easy_only = not action.long_run_allowed and gap.severity != GapSeverity.SHORT

    run_type = decide_run_type(this_week, this_week_plan.quality_sessions, this_week_plan.is_backoff, easy_only)

    if health_flag:
        run_type = "easy"
        reasoning.append(
            f"Health flag '--flag {health_flag}' active: forcing conservative branch — "
            "cap at last completed distance, no quality, consider cross-training instead."
        )

    zone = gr.acwr_zone(anchors.acwr)
    already_this_week = sum(
        c.activity.distance_mi for c in this_week if c.run_class in RUNNING_CLASSES
    )

    candidates: list[GuardrailResult] = []

    # Cold start: no training history at all (new athlete, or Garmin has
    # never synced). Every load-derived cap is 0 here, which would
    # otherwise recommend a 0.0 mi run — technically "safe" but useless
    # and alarming as a first impression. Fall back to a conservative
    # fraction of the configured baseline and say plainly that it's a
    # stated-baseline guess, not evidence. This branch is deliberately
    # narrow: it requires a total absence of history, so an athlete who
    # has simply used up this week's cap still correctly gets 0/rest.
    cold_start = anchors.chronic_load_mi <= 0 and anchors.acute_load_mi <= 0
    if cold_start:
        starter = max(config.athlete.baseline_long_run_mi * 0.4, 1.0)
        candidates.append(GuardrailResult(
            "cold_start_baseline", starter,
            f"No training history available yet, so load anchors are all zero. Using a "
            f"conservative {starter:.1f} mi from the configured baseline "
            f"({config.athlete.baseline_long_run_mi:.1f} mi) — this is a stated assumption, "
            "not evidence. Run `usain-bot backfill` to replace it with real data.",
        ))
        reasoning.append(
            "Cold start: no Garmin history is loaded, so today's number comes from config.yaml, "
            "not from your actual training. Back-fill your history before trusting it."
        )
    else:
        remaining_weekly = max(gr.max_weekly_volume(anchors.chronic_load_mi) - already_this_week, 0.0)
        candidates.append(GuardrailResult(
            "weekly_volume_cap", remaining_weekly,
            f"Chronic load {anchors.chronic_load_mi:.1f} mi/wk x1.10 = "
            f"{gr.max_weekly_volume(anchors.chronic_load_mi):.1f} mi cap; {already_this_week:.1f} mi already this week.",
        ))

    if run_type == "long":
        base_increment_cap = gr.next_long_run_distance(anchors.long_run_anchor_mi)
        candidates.append(GuardrailResult(
            "long_run_increment", base_increment_cap,
            f"min(1.0 mi, 10% of {anchors.long_run_anchor_mi:.1f} mi anchor) = "
            f"{gr.long_run_increment(anchors.long_run_anchor_mi):.2f} mi increment.",
        ))

        pct_cap = gr.max_long_run_by_weekly_volume_pct(this_week_plan.target_volume_mi)
        candidates.append(GuardrailResult(
            "long_run_pct_of_weekly_volume", pct_cap,
            f"Long run capped at 35% of this week's {this_week_plan.target_volume_mi:.1f} mi target.",
        ))

        if gap.severity == GapSeverity.SHORT and action.max_long_run_mi is not None:
            candidates.append(GuardrailResult(
                "gap_protocol", action.max_long_run_mi, action.description,
            ))
        if easy_only:
            candidates.append(GuardrailResult("gap_protocol_no_long_run", 0.0, action.description))

        if zone == gr.ACWRZone.YELLOW:
            candidates.append(GuardrailResult(
                "acwr_yellow_hold_flat", anchors.long_run_anchor_mi,
                f"ACWR {anchors.acwr:.2f} in yellow zone (1.3-1.5): hold flat, no increase.",
            ))
            conflicts.append("ACWR yellow zone conflicts with a planned increment — the hold wins.")
        elif zone == gr.ACWRZone.RED:
            reduced = anchors.long_run_anchor_mi * 0.8
            candidates.append(GuardrailResult(
                "acwr_red_reduction", reduced,
                f"ACWR {anchors.acwr:.2f} in red zone (>1.5): forcing a reduction regardless of plan.",
            ))
            conflicts.append("ACWR red zone overrides any plan-driven increase — reduction wins.")
        elif zone == gr.ACWRZone.UNDEFINED:
            reasoning.append(
                "ACWR undefined (cold start, <28 days of chronic load): treated as a non-binding "
                "signal, not used to justify pushing forward."
            )

        if this_week_plan.is_backoff:
            backoff_targets = gr.backoff_week_targets(this_week_plan.target_volume_mi / gr.BACKOFF_VOLUME_PCT_LOW, anchors.long_run_anchor_mi)
            candidates.append(GuardrailResult(
                "backoff_week", backoff_targets.long_run_mi,
                "Scheduled back-off week (3:1 cadence): long run capped at 70% of recent peak, no increase.",
            ))
    else:
        # At cold start every load-derived number is 0, so fall back to the
        # configured baseline rather than a 0-mile "recommendation".
        easy_fallback = (
            max(config.athlete.baseline_long_run_mi * 0.4, 1.0) if cold_start
            else anchors.chronic_load_mi / max(config.athlete.available_run_days_per_week, 1)
        )
        recent_easy = _recent_easy_distance(classified, as_of, fallback=easy_fallback)
        multiplier = 1.1 if run_type == "quality" else 1.0
        candidates.append(GuardrailResult(
            f"{run_type}_typical_distance", recent_easy * multiplier,
            f"Recent {run_type} distance baseline ({recent_easy:.1f} mi median, trailing 28d)."
            if not cold_start else
            f"No history yet — {recent_easy:.1f} mi derived from the configured baseline.",
        ))

    if health_flag:
        flag_cap = anchors.long_run_anchor_mi * HEALTH_FLAG_CAP_FACTOR if run_type == "long" else \
            _recent_easy_distance(classified, as_of, fallback=3.0) * HEALTH_FLAG_CAP_FACTOR
        candidates.append(GuardrailResult(
            f"health_flag_{health_flag}", flag_cap,
            f"Health flag active: capped below last completed distance as a precaution.",
        ))

    # Subjective state: how recent runs actually FELT is a real signal the
    # physical anchors can't see. Deterministic (not left to the LLM):
    # a rough recent average caps today below what load math alone allows.
    feeling_guardrail = _feeling_guardrail(storage, classified, anchors, as_of, run_type)
    if feeling_guardrail is not None:
        candidates.append(feeling_guardrail)
        conflicts.append(
            "Recent runs have been feeling rough — that conservative signal is applied on top of the load math."
        )

    guardrail_results = candidates
    binding = min(
        (g for g in guardrail_results if g.max_value is not None),
        key=lambda g: g.max_value,
    )
    target_distance = max(round(binding.max_value, 2), 0.0)

    reasoning.append(f"Anchors: acute={anchors.acute_load_mi:.1f} mi, chronic={anchors.chronic_load_mi:.1f} mi/wk, "
                      f"long-run anchor={anchors.long_run_anchor_mi:.1f} mi, "
                      f"ACWR={'n/a' if anchors.acwr is None else f'{anchors.acwr:.2f}'} ({zone.value}).")
    reasoning.append(f"Run type for today: {run_type} (based on what's already in the trailing 7 days).")
    reasoning.append(f"Binding constraint: {binding.name} -> {target_distance:.2f} mi. {binding.reason}")

    if len(conflicts) == 0 and gap.severity != GapSeverity.SHORT:
        conflicts.append(f"Gap protocol ({gap.severity.value}) took priority over normal progression.")

    unlock_next_time = _unlock_message(binding, anchors, zone)
    effort = _effort_guidance(run_type, zone)

    time_on_feet = None
    if target_distance > 16:
        pace_min_per_mi = 10.5
        time_on_feet = round(target_distance * pace_min_per_mi, 1)

    return Recommendation(
        date=as_of,
        run_type=run_type,
        target_distance_mi=target_distance,
        time_on_feet_min=time_on_feet,
        effort_guidance=effort,
        binding_constraint=f"{binding.name}: {target_distance:.2f} mi",
        reasoning=reasoning,
        unlock_next_time=unlock_next_time,
        guardrail_results=guardrail_results,
        conflicts=conflicts,
    )


def _feeling_guardrail(
    storage: StorageBackend,
    classified: list[ClassifiedActivity],
    anchors: Anchors,
    as_of: date,
    run_type: str,
) -> Optional[GuardrailResult]:
    """Turn recent run-feeling scores into a hard ceiling candidate.
    Returns None when there's no recent feedback or it's been fine."""
    feedback = storage.get_recent_run_feedback(days=FEELING_LOOKBACK_DAYS)
    if not feedback:
        return None
    mean_score = sum(f.score for f in feedback) / len(feedback)
    if mean_score > FEELING_MEH_THRESHOLD:
        return None

    base = anchors.long_run_anchor_mi if run_type == "long" else _recent_easy_distance(
        classified, as_of, fallback=anchors.chronic_load_mi / 3 if anchors.chronic_load_mi else 3.0
    )
    if base <= 0:
        return None

    if mean_score <= FEELING_ROUGH_THRESHOLD:
        factor, label = FEELING_ROUGH_FACTOR, "rough"
    else:
        factor, label = FEELING_MEH_FACTOR, "below par"

    return GuardrailResult(
        "recent_run_feeling", base * factor,
        f"Recent runs reported feeling {label} (mean {mean_score:.1f}/5 across {len(feedback)} "
        f"check-in(s), trailing {FEELING_LOOKBACK_DAYS}d): holding at/below last completed distance.",
    )


def _effort_guidance(run_type: str, zone: gr.ACWRZone) -> str:
    if run_type == "long":
        return "Conversational, aerobic effort throughout. Walk breaks are fine, especially beyond mile 10."
    if run_type == "quality":
        if zone in (gr.ACWRZone.YELLOW, gr.ACWRZone.RED):
            return "Downgrade to easy today — quality is deferred while ACWR is elevated."
        return "Tempo/threshold effort — comfortably hard, sustainable for the segment, not a max effort."
    return "Easy, conversational pace. This is aerobic base, not a workout."


def _unlock_message(binding: GuardrailResult, anchors: Anchors, zone: gr.ACWRZone) -> str:
    if binding.name == "weekly_volume_cap":
        return "More room next week as chronic load (28-day average) rises with consistent training."
    if binding.name in ("acwr_yellow_hold_flat", "acwr_red_reduction"):
        return "ACWR needs to settle back under 1.3 (fewer acute spikes relative to the 28-day average) before increasing again."
    if binding.name.startswith("gap_protocol"):
        return "Once you've logged consecutive completed easy runs, the gap protocol will ease and normal progression resumes."
    if binding.name == "long_run_pct_of_weekly_volume":
        return "Raising this week's overall volume (while staying under the weekly cap) would raise the 35% long-run ceiling too."
    if binding.name == "backoff_week":
        return "This is a scheduled back-off week — next week resumes normal progression."
    if binding.name == "recent_run_feeling":
        return "A couple of runs that feel good again will lift this cap — tell the coach how each one went."
    if binding.name == "cold_start_baseline":
        return "Run `usain-bot backfill` to import your Garmin history — every number here becomes evidence-based once it lands."
    return "Consistent, injury-free weeks are what unlock the next increment."


def _cited_reference_note(storage: StorageBackend, query: str) -> str:
    """§6: when updating the plan, consult the references store and cite
    whichever reference informed the change. Returns an empty string if
    nothing relevant is indexed yet (e.g. no references added)."""
    try:
        hits = storage.search_references(query, top_k=1)
    except Exception:  # noqa: BLE001 - references are best-effort, never block a plan update
        return ""
    if not hits:
        return ""
    hit = hits[0]
    snippet = hit.text.strip().replace("\n", " ")[:180]
    return f" Per reference '{hit.title}': \"{snippet}...\""


def _log_decision(rec: Recommendation) -> None:
    logger.info(
        "recommendation date=%s run_type=%s distance=%.2f binding=%s",
        rec.date, rec.run_type, rec.target_distance_mi or 0.0, rec.binding_constraint,
    )
    for g in rec.guardrail_results:
        logger.info("guardrail name=%s max_value=%s reason=%s", g.name, g.max_value, g.reason)


@dataclass
class InvocationResult:
    recommendation: Recommendation
    plan: PlanVersion
    anchors: Anchors
    sync: SyncResult


def run_invocation(
    config: Config,
    storage: StorageBackend,
    adapter: GarminAdapter,
    as_of: Optional[date] = None,
    health_flag: Optional[str] = None,
    dry_run: bool = False,
) -> InvocationResult:
    """§4.3 steps 1-8, end to end."""
    as_of = as_of or date.today()

    sync = sync_activities(storage, adapter, as_of, overlap_days=config.sync.overlap_days)
    activities = storage.get_activities()
    classified = prepare_classified(activities, config.sync.merge_gap_hours)

    planned_runs_14d = config.athlete.available_run_days_per_week * 2
    anchors = compute_anchors(classified, as_of, planned_runs_trailing_14d=planned_runs_14d)

    gap = anchors.gap
    action = gr.gap_action(gap.gap_days, anchors.long_run_anchor_mi)

    prior_plan = storage.get_latest_plan_version()
    next_version = (prior_plan.version + 1) if prior_plan else 1

    # Athlete-requested milestone pushes persist across regenerations
    # (the plan itself is regenerated from anchors every invocation, so
    # anything meant to stick has to live outside the plan).
    hm_delay = _read_int_preference(storage, PREF_HM_DELAY_WEEKS)
    ultra_delay = _read_int_preference(storage, PREF_ULTRA_DELAY_WEEKS)
    run_days = anchors.runs_per_week or config.athlete.available_run_days_per_week

    plan_kwargs = dict(hm_delay_weeks=hm_delay, ultra_delay_weeks=ultra_delay, run_days_per_week=run_days)

    if action.regenerate_plan:
        trigger = "gap_regeneration"
        rationale = f"Gap of {gap.gap_days} days exceeds 6 weeks: {action.description}"
        rationale += _cited_reference_note(storage, "return to running after long break detraining")
        plan = generate_macro_plan(config, anchors, as_of, next_version, trigger, rationale, **plan_kwargs)
    else:
        trigger = "scheduled_reprojection" if gap.severity == GapSeverity.SHORT else "gap_detected"
        rationale = (
            "Rolling re-projection from current anchors." if gap.severity == GapSeverity.SHORT
            else f"Gap of {gap.gap_days} days ({gap.severity.value}): {action.description}"
        )
        if anchors.runs_per_week and anchors.runs_per_week != config.athlete.available_run_days_per_week:
            rationale += (
                f" Running frequency derived from actuals: {anchors.runs_per_week} day(s)/week "
                f"(config says {config.athlete.available_run_days_per_week}) — plan adapted to what's "
                "actually being run."
            )
        if gap.severity != GapSeverity.SHORT:
            rationale += _cited_reference_note(storage, "return to running after a break gap protocol")
        plan = generate_macro_plan(
            config, anchors, as_of, next_version, trigger, rationale,
            gap_hold_weeks=action.backtrack_weeks,
            gap_easy_only=action.easy_only,
            **plan_kwargs,
        )

    if health_flag:
        storage.save_health_flag(HealthFlag(timestamp=datetime.utcnow(), flag=health_flag))

    # Every saved version gets a diff recorded, not just conversational
    # overrides — otherwise the plan_versions lineage is dominated by
    # daily reprojections with no record of what actually changed.
    plan.diff_from_prior = diff_plan_versions(prior_plan, plan)

    recommendation = compute_recommendation(config, storage, classified, anchors, plan, as_of, health_flag)
    _log_decision(recommendation)

    if not dry_run:
        storage.save_plan_version(plan)
        storage.save_conversation_entry(ConversationEntry(
            timestamp=datetime.utcnow(), role="agent",
            text=f"Recommended {recommendation.run_type} run, {recommendation.target_distance_mi} mi.",
            metadata={"binding_constraint": recommendation.binding_constraint, "health_flag": health_flag},
        ))

    return InvocationResult(recommendation, plan, anchors, sync)


@dataclass
class FirstRunReport:
    anchors: Anchors
    proposed_plan: PlanVersion
    baseline_conflict: Optional[str]
    adherence_notes: list[str]
    uncertainties: list[str]
    sync: SyncResult


def first_run_report(config: Config, storage: StorageBackend, adapter: GarminAdapter, as_of: Optional[date] = None) -> FirstRunReport:
    """§9 steps 1-3, 5: pull history, report what was found, generate the
    macro plan, and list open uncertainties. Does NOT persist plan v1 —
    that happens only after the athlete confirms (see confirm_first_run)."""
    as_of = as_of or date.today()
    sync = sync_activities(storage, adapter, as_of, lookback_weeks=FIRST_RUN_LOOKBACK_WEEKS,
                            overlap_days=config.sync.overlap_days)
    activities = storage.get_activities()
    classified = prepare_classified(activities, config.sync.merge_gap_hours)
    anchors = compute_anchors(classified, as_of, planned_runs_trailing_14d=config.athlete.available_run_days_per_week * 2)

    baseline_conflict = None
    configured_baseline = config.athlete.baseline_long_run_mi
    if anchors.long_run_anchor_mi > 0 and abs(anchors.long_run_anchor_mi - configured_baseline) > 1.0:
        baseline_conflict = (
            f"config.yaml baseline_long_run_mi is {configured_baseline:.1f} mi, but Garmin history shows "
            f"an actual long-run anchor of {anchors.long_run_anchor_mi:.1f} mi. Using the Garmin-derived "
            "value — the plan is a hypothesis, Garmin is the evidence."
        )

    adherence_notes = []
    if anchors.adherence_rate is not None:
        adherence_notes.append(f"Adherence over trailing 14 days: {anchors.adherence_rate * 100:.0f}% of planned runs completed.")
    if anchors.gap.severity != GapSeverity.SHORT:
        adherence_notes.append(f"Gap detected: {anchors.gap.gap_days} days since last run ({anchors.gap.severity.value}).")

    plan = generate_macro_plan(
        config, anchors, as_of, version=1, trigger="first_run",
        rationale="Initial macro plan generated from 12 weeks of Garmin history on first invocation.",
    )
    plan.diff_from_prior = diff_plan_versions(None, plan)

    uncertainties = [
        "Injury flare risk is inferred only from pace/HR/volume signals in Garmin data — it has no "
        "direct visibility into hip or lower-back symptoms. Use --flag hip / --flag back proactively.",
        "Quality-session detection is heuristic (naming + HR ratio) since Garmin's raw payload doesn't "
        "reliably label workout type; misclassified sessions would skew the 80/20 easy/hard split.",
        f"Half-marathon-capability timing assumes the {config.athlete.available_run_days_per_week}-day/week "
        "cadence holds; adherence data over the next few weeks will confirm or revise that.",
    ]
    if anchors.acwr is None:
        uncertainties.append("ACWR is undefined at cold start (< 28 days of chronic load) — treated as non-binding until then.")

    return FirstRunReport(anchors, plan, baseline_conflict, adherence_notes, uncertainties, sync)


def confirm_first_run(storage: StorageBackend, report: FirstRunReport) -> None:
    storage.save_plan_version(report.proposed_plan)
    storage.save_conversation_entry(ConversationEntry(
        timestamp=datetime.utcnow(), role="agent",
        text="Plan v1 confirmed and persisted on first run.",
        metadata={"trigger": "first_run"},
    ))


def override_plan(config: Config, storage: StorageBackend, text: str, as_of: Optional[date] = None) -> OverrideResult:
    as_of = as_of or date.today()
    current = storage.get_latest_plan_version()
    if current is None:
        return OverrideResult(None, False, "", ["No existing plan to override — run `usain-bot init` first."])

    activities = storage.get_activities()
    classified = prepare_classified(activities, config.sync.merge_gap_hours)
    anchors = compute_anchors(classified, as_of, planned_runs_trailing_14d=config.athlete.available_run_days_per_week * 2)

    storage.save_conversation_entry(ConversationEntry(
        timestamp=datetime.utcnow(), role="user", text=text, metadata={},
    ))

    result = apply_override(current, text, config, anchors, as_of)
    if result.applied and result.plan is not None:
        result.plan.diff_from_prior = diff_plan_versions(current, result.plan)
        storage.save_plan_version(result.plan)
        storage.save_conversation_entry(ConversationEntry(
            timestamp=datetime.utcnow(), role="agent", text=result.rationale,
            metadata={"trigger": "user_override", "warnings": result.warnings},
        ))
    else:
        storage.save_conversation_entry(ConversationEntry(
            timestamp=datetime.utcnow(), role="agent",
            text="Override not applied: " + "; ".join(result.warnings), metadata={},
        ))
    return result
