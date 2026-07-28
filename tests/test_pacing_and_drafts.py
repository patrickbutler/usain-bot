"""Tests for the phase-10 feedback: pacing modes, the mandatory rest day
after a long run, the draft -> approve -> publish plan revision workflow,
run-feeling history, and the hidden Jamaican-mode setting.

The load-bearing assertion in here is
`TestDraftPublishSurvivesRegeneration` — the plan is regenerated from
anchors on every invocation, so a conversational change that isn't
persisted as a constraint silently reverts on the next page load. That
was the actual bug behind "I asked chat to smooth the plan and the
Upcoming tab didn't change".
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from usain_bot import agent
from usain_bot import guardrails as gr
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import Anchors, GapInfo, GapSeverity, Recommendation
from usain_bot.planner import (
    HIGH_MILEAGE_LONG_RUN_MI,
    PacingMode,
    PlanConstraints,
    generate_macro_plan,
    scheduled_long_run,
    summarize_plan,
)
from usain_bot.projection import project_next_7_days
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend
from usain_bot.web.app import create_app

AS_OF = date(2026, 7, 24)


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def storage(config):
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


@pytest.fixture
def adapter(fixture_path):
    return MockGarminAdapter(fixture_path)


@pytest.fixture
def service(config, storage, adapter, monkeypatch):
    monkeypatch.setattr(
        "usain_bot.service.date",
        type("FixedDate", (), {"today": staticmethod(lambda: AS_OF), "fromisoformat": date.fromisoformat}),
    )
    svc = CoachService(config, storage, adapter)
    svc.sync(as_of=AS_OF)   # populate storage from the fixture history
    return svc


@pytest.fixture
def client(config, storage, adapter, monkeypatch):
    monkeypatch.setattr(
        "usain_bot.service.date",
        type("FixedDate", (), {"today": staticmethod(lambda: AS_OF), "fromisoformat": date.fromisoformat}),
    )
    return TestClient(create_app(config, storage, adapter))


@pytest.fixture
def easy_recommendation():
    return Recommendation(
        date=AS_OF, run_type="easy", target_distance_mi=4.0, time_on_feet_min=40.0,
        effort_guidance="conversational", binding_constraint="acwr", reasoning=[],
        unlock_next_time="", guardrail_results=[], conflicts=[],
    )


@pytest.fixture
def warm_anchors():
    return Anchors(
        as_of=AS_OF, acute_load_mi=22.0, chronic_load_mi=24.0,
        long_run_anchor_mi=8.0, adherence_rate=0.9, acwr=22.0 / 24.0,
        gap=GapInfo(gap_days=1, severity=GapSeverity.SHORT, last_run_date=date(2026, 7, 23)),
    )


# --- item 6: at least one day off after a long run ---------------------------

class TestRestDayAfterLongRun:
    def test_day_after_long_run_requires_rest(self):
        assert gr.requires_rest_after_long_run(1) is True

    def test_two_days_later_is_free_again(self):
        assert gr.requires_rest_after_long_run(2) is False

    def test_same_day_is_not_the_rest_day(self):
        # 0 means the long run *is* today; the rest day is tomorrow.
        assert gr.requires_rest_after_long_run(0) is False

    def test_unknown_history_never_forces_rest(self):
        assert gr.requires_rest_after_long_run(None) is False

    def test_projection_puts_rest_the_day_after_the_long_run(self, easy_recommendation):
        # 7 run days/week is the hardest case: every weekday is a run day,
        # so the rest day can only come from the long-run rule.
        days = project_next_7_days(
            as_of=AS_OF, run_days_per_week=7, quality_target=1, is_backoff=False,
            recommendation=easy_recommendation,
        )
        long_idx = next(i for i, d in enumerate(days) if d.run_type == "long")
        assert days[(long_idx + 1) % 7].run_type == "rest"

    def test_projection_rests_today_when_long_run_was_yesterday(self, easy_recommendation):
        days = project_next_7_days(
            as_of=AS_OF, run_days_per_week=6, quality_target=1, is_backoff=False,
            recommendation=easy_recommendation, long_run_was_yesterday=True,
        )
        assert days[0].run_type == "rest"
        assert days[0].distance_mi is None

    def test_agent_caps_today_at_zero_the_day_after_a_long_run(self, config, storage, adapter, monkeypatch):
        """The rest day is a hard ceiling, not advice: it enters the same
        min-across-ceilings the other guardrails use."""
        monkeypatch.setattr(agent, "_days_since_last_long_run", lambda *a, **k: 1)
        result = agent.run_invocation(config, storage, adapter, as_of=AS_OF, dry_run=True)
        assert result.recommendation.target_distance_mi == 0.0
        assert "rest" in result.recommendation.binding_constraint.lower()


# --- item 7: two pacing modes ------------------------------------------------

class TestScheduledLongRun:
    def test_interpolates_linearly_across_the_build(self):
        assert scheduled_long_run(8.0, 20.0, 0, 12) == pytest.approx(8.0)
        assert scheduled_long_run(8.0, 20.0, 6, 12) == pytest.approx(14.0)
        assert scheduled_long_run(8.0, 20.0, 12, 12) == pytest.approx(20.0)

    def test_clamps_past_the_end_of_the_build(self):
        assert scheduled_long_run(8.0, 20.0, 99, 12) == pytest.approx(20.0)

    def test_no_runway_means_target_immediately(self):
        assert scheduled_long_run(8.0, 20.0, 0, 0) == pytest.approx(20.0)

    def test_target_below_baseline_is_not_a_ramp(self):
        assert scheduled_long_run(20.0, 12.0, 3, 10) == pytest.approx(12.0)


class TestPacingModes:
    def test_smoothed_spends_fewer_weeks_at_the_peak(self, config, warm_anchors):
        smoothed = summarize_plan(generate_macro_plan(
            config, warm_anchors, AS_OF, 1, "test", "test",
            pacing_mode=PacingMode.MILESTONE_SMOOTHED))
        asap = summarize_plan(generate_macro_plan(
            config, warm_anchors, AS_OF, 1, "test", "test",
            pacing_mode=PacingMode.RAMP_ASAP))
        assert smoothed["weeks_at_peak_long_run"] <= asap["weeks_at_peak_long_run"]
        assert smoothed["weeks_over_15mi_long_run"] < asap["weeks_over_15mi_long_run"]

    def test_smoothed_is_the_default(self, config, warm_anchors):
        default = summarize_plan(generate_macro_plan(config, warm_anchors, AS_OF, 1, "test", "test"))
        smoothed = summarize_plan(generate_macro_plan(
            config, warm_anchors, AS_OF, 1, "test", "test",
            pacing_mode=PacingMode.MILESTONE_SMOOTHED))
        assert default == smoothed

    def test_max_weeks_at_peak_is_respected(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, AS_OF, 1, "test", "test", max_weeks_at_peak=2)
        assert summarize_plan(plan)["weeks_at_peak_long_run"] <= 2

    def test_peak_long_run_cap_is_respected(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, AS_OF, 1, "test", "test", peak_long_run_cap=16.0)
        build = [w for w in plan.weeks if w.block in ("base_building", "marathon_block", "ultra_build")]
        assert max(w.long_run_mi for w in build) <= 16.0 + 1e-6

    def test_fewer_run_days_lowers_the_volume_the_week_can_absorb(self, config, warm_anchors):
        """`run_days_per_week` is an advertised revision knob, so it has to
        actually change the plan — a low-frequency week cannot absorb the
        same mileage."""
        five = generate_macro_plan(config, warm_anchors, AS_OF, 1, "t", "t", run_days_per_week=5)
        two = generate_macro_plan(config, warm_anchors, AS_OF, 1, "t", "t", run_days_per_week=2)
        assert summarize_plan(two)["peak_weekly_volume_mi"] < summarize_plan(five)["peak_weekly_volume_mi"]

    def test_no_quality_sessions_below_four_run_days(self, config, warm_anchors):
        """Three days a week leaves no room for a quality session next to
        the long run and its recovery day. (Taper sharpening is a separate
        decision and keeps its session.)"""
        three = generate_macro_plan(config, warm_anchors, AS_OF, 1, "t", "t", run_days_per_week=3)
        five = generate_macro_plan(config, warm_anchors, AS_OF, 1, "t", "t", run_days_per_week=5)
        build_quality = lambda p: sum(w.quality_sessions for w in p.weeks if w.block == "marathon_block")
        assert build_quality(three) == 0
        assert build_quality(five) > 0

    def test_high_mileage_long_runs_stay_strategic(self, config, warm_anchors):
        """Runs over 15 miles are injury-prone, so under the default
        smoothed pacing they should be a minority of the build."""
        plan = generate_macro_plan(config, warm_anchors, AS_OF, 1, "test", "test")
        build = [w for w in plan.weeks if w.block in ("base_building", "marathon_block", "ultra_build")]
        over = [w for w in build if w.long_run_mi > HIGH_MILEAGE_LONG_RUN_MI]
        assert len(over) < len(build) / 2


# --- items 8 + 9: draft -> approve -> publish, and it must stick -------------

class TestDraftPlanRevision:
    def test_proposal_does_not_save_a_version(self, service):
        service.get_today(as_of=AS_OF)
        before = service.get_plan().version
        result = service.propose_plan_revision(
            PlanConstraints(max_weeks_at_peak=2), "smooth the peak", as_of=AS_OF)
        assert result["draft"] is True
        assert service.get_plan().version == before

    def test_proposal_reports_what_changed(self, service):
        service.get_today(as_of=AS_OF)
        result = service.propose_plan_revision(
            PlanConstraints(peak_long_run_cap=16.0), "cap the long runs", as_of=AS_OF)
        assert result["summary"]["peak_long_run_mi"] <= 16.0
        assert result["current_summary"] is not None
        assert "peak long run capped at 16 mi" in result["requested_changes"]
        assert "draft" in result["next_step"].lower()

    def test_proposal_asks_before_publishing(self, service):
        service.get_today(as_of=AS_OF)
        result = service.propose_plan_revision(PlanConstraints(max_weeks_at_peak=2), "x", as_of=AS_OF)
        assert "publish_draft_plan" in result["next_step"]

    def test_publish_without_a_draft_is_an_error(self, service):
        assert "error" in service.publish_draft_plan()

    def test_publish_saves_a_new_version_with_reasoning(self, service):
        service.get_today(as_of=AS_OF)
        before = service.get_plan().version
        service.propose_plan_revision(PlanConstraints(max_weeks_at_peak=2), "smooth the peak", as_of=AS_OF)
        published = service.publish_draft_plan("approved in chat")

        assert published["published"] is True
        assert published["plan_version"] == before + 1
        assert service.get_plan().version == before + 1
        stored = service.get_plan()
        assert "smooth the peak" in stored.rationale
        assert "approved in chat" in stored.rationale
        assert "at most 2 week(s) at the peak long run" in stored.rationale
        assert stored.diff_from_prior

    def test_discard_drops_the_draft(self, service):
        service.get_today(as_of=AS_OF)
        service.propose_plan_revision(PlanConstraints(max_weeks_at_peak=2), "x", as_of=AS_OF)
        assert service.discard_draft_plan()["discarded"] is True
        assert service.get_draft_plan_payload() is None
        assert "error" in service.publish_draft_plan()


class TestDraftPublishSurvivesRegeneration:
    def test_published_shape_is_not_regenerated_away(self, service):
        """The exact failure the athlete reported: ask for fewer weeks at
        the peak, get told it's done, then reload and find the old plan."""
        service.get_today(as_of=AS_OF)
        before = summarize_plan(service.get_plan())["weeks_at_peak_long_run"]

        service.propose_plan_revision(
            PlanConstraints(max_weeks_at_peak=2, peak_long_run_cap=16.0), "smooth it", as_of=AS_OF)
        service.publish_draft_plan("yes, make it official")
        published = summarize_plan(service.get_plan())
        assert published["weeks_at_peak_long_run"] <= 2
        assert published["peak_long_run_mi"] <= 16.0

        # Force a full regeneration — this is what a page reload the next
        # day does. The approved constraints must be read back from
        # preferences, not silently dropped.
        service._cached = None
        service.refresh_today(as_of=AS_OF)
        after = summarize_plan(service.get_plan())
        assert after["weeks_at_peak_long_run"] <= 2
        assert after["peak_long_run_mi"] <= 16.0
        assert before >= after["weeks_at_peak_long_run"]

    def test_constraints_are_persisted_as_preferences(self, service, storage):
        service.get_today(as_of=AS_OF)
        service.propose_plan_revision(
            PlanConstraints(pacing_mode=PacingMode.RAMP_ASAP, max_weeks_at_peak=3,
                            peak_long_run_cap=18.0), "ramp me", as_of=AS_OF)
        service.publish_draft_plan()
        assert storage.get_preference(agent.PREF_PACING_MODE) == PacingMode.RAMP_ASAP.value
        assert storage.get_preference(agent.PREF_MAX_WEEKS_AT_PEAK) == "3"
        assert float(storage.get_preference(agent.PREF_PEAK_LONG_RUN_CAP)) == 18.0

    def test_read_plan_constraints_round_trips(self, service, storage):
        service.get_today(as_of=AS_OF)
        service.propose_plan_revision(PlanConstraints(max_weeks_at_peak=2), "x", as_of=AS_OF)
        service.publish_draft_plan()
        constraints = agent.read_plan_constraints(storage)
        assert constraints.max_weeks_at_peak == 2

    def test_unchanged_plan_does_not_bump_the_version(self, service):
        """Regenerating an identical plan must not manufacture history —
        otherwise the lineage fills with meaningless versions."""
        service.get_today(as_of=AS_OF)
        first = service.get_plan().version
        service._cached = None
        service.refresh_today(as_of=AS_OF)
        assert service.get_plan().version == first


class TestPlanRevisionEndpoints:
    def test_propose_then_publish_over_http(self, client):
        client.get("/api/today")
        before = client.get("/api/plan").json()["plan_version"]

        proposed = client.post("/api/plan/revision/propose", json={
            "rationale": "smooth the ramp", "max_weeks_at_peak": 2, "peak_long_run_cap": 16.0,
        })
        assert proposed.status_code == 200
        assert proposed.json()["draft"] is True
        assert client.get("/api/plan").json()["plan_version"] == before

        published = client.post("/api/plan/revision/publish", json={"approval_note": "go"})
        assert published.status_code == 200
        assert client.get("/api/plan").json()["plan_version"] == before + 1

    def test_publish_without_draft_returns_400(self, client):
        client.get("/api/today")
        assert client.post("/api/plan/revision/publish", json={}).status_code == 400

    def test_bad_pacing_mode_returns_400(self, client):
        client.get("/api/today")
        resp = client.post("/api/plan/revision/propose", json={"pacing_mode": "sprint"})
        assert resp.status_code == 400

    def test_discard_endpoint(self, client):
        client.get("/api/today")
        client.post("/api/plan/revision/propose", json={"max_weeks_at_peak": 2})
        assert client.delete("/api/plan/revision").json()["discarded"] is True


# --- item 3: run feelings in History ----------------------------------------

class TestRunFeelingsHistory:
    def test_runs_start_unrated(self, service):
        payload = service.get_run_feelings_payload(days=90)
        assert payload["runs"]
        assert payload["rated_count"] == 0
        assert payload["mean_score"] is None
        assert all(r["score"] is None for r in payload["runs"])

    def test_scoring_a_run_attaches_it(self, service):
        target = service.get_run_feelings_payload(days=90)["runs"][0]
        service.record_run_feeling(4, "felt strong", date.fromisoformat(target["date"]))
        payload = service.get_run_feelings_payload(days=90)
        rated = next(r for r in payload["runs"] if r["date"] == target["date"])
        assert rated["score"] == 4
        assert rated["comment"] == "felt strong"
        assert payload["rated_count"] == 1
        assert payload["mean_score"] == 4.0

    def test_a_rated_run_is_never_asked_about_again(self, service):
        target = service.get_run_feelings_payload(days=10)["runs"][0]
        service.record_run_feeling(2, None, date.fromisoformat(target["date"]))
        unrated = service.get_unrated_recent_runs(days=10)
        assert target["date"] not in [r["date"] for r in unrated]

    def test_rescoring_a_run_keeps_the_newest_score(self, service):
        target = service.get_run_feelings_payload(days=90)["runs"][0]
        run_date = date.fromisoformat(target["date"])
        service.record_run_feeling(2, "rough", run_date)
        service.record_run_feeling(5, "actually great", run_date)
        payload = service.get_run_feelings_payload(days=90)
        rated = next(r for r in payload["runs"] if r["date"] == target["date"])
        assert rated["score"] == 5
        assert payload["rated_count"] == 1

    def test_endpoint_returns_runs_with_scores(self, client):
        client.get("/api/today")   # pulls the fixture history into storage
        runs = client.get("/api/runs/feelings?days=90").json()
        assert runs["runs"]
        first = runs["runs"][0]["date"]
        client.post("/api/feelings", json={"score": 3, "activity_date": first})
        after = client.get("/api/runs/feelings?days=90").json()
        assert after["rated_count"] == 1
        assert next(r for r in after["runs"] if r["date"] == first)["score"] == 3


# --- item 5: hidden Jamaican-mode switch -------------------------------------

class TestJamaicanModeSetting:
    def test_defaults_to_on(self, client):
        body = client.get("/api/settings").json()
        assert body["jamaican_mode"] is True
        assert body["chat_provider"]

    def test_toggle_persists(self, client):
        assert client.post("/api/settings", json={"jamaican_mode": False}).json()["jamaican_mode"] is False
        assert client.get("/api/settings").json()["jamaican_mode"] is False
        assert client.post("/api/settings", json={"jamaican_mode": True}).json()["jamaican_mode"] is True

    def test_settings_page_is_served_but_unlinked(self, client):
        assert client.get("/settings").status_code == 200
        assert "/settings" not in client.get("/").text

    def test_voice_changes_but_coaching_rules_do_not(self):
        from usain_bot.chat.session import build_system_prompt

        jamaican = build_system_prompt(jamaican_mode=True)
        professional = build_system_prompt(jamaican_mode=False)
        assert jamaican != professional
        assert "Jamaican" in jamaican
        assert "Jamaican" not in professional
        # The guardrail contract is identical in both voices.
        for prompt in (jamaican, professional):
            assert "propose_plan_revision" in prompt
            assert "publish_draft_plan" in prompt
