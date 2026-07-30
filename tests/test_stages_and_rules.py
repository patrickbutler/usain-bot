"""Training stages, the maintain-stage oscillation, milestone tracking,
the milestone-shape rules with their repair loop, pacing-phrase
governance, and the never-abbreviated plan range view.

The load-bearing assertions here are the two hard rules:

* a max-mileage week is never followed by another one
  (`TestNoBackToBackPeakWeeks`), and
* an undated milestone builds conservatively to its peak long run and
  then tapers straight in, while a dated one ramps smoothly and maintains
  (`TestMilestoneShapeRules`).
"""

import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from usain_bot import milestones as ms
from usain_bot import pacing_intent, planner, rules, stages
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import Anchors, GapInfo, GapSeverity, PlanVersion, PlanWeek
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend
from usain_bot.validation import validate_plan
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
def service(config, storage, fixture_path, monkeypatch):
    monkeypatch.setattr(
        "usain_bot.service.date",
        type("FixedDate", (), {"today": staticmethod(lambda: AS_OF), "fromisoformat": date.fromisoformat}),
    )
    svc = CoachService(config, storage, MockGarminAdapter(fixture_path))
    svc.sync(as_of=AS_OF)
    return svc


@pytest.fixture
def client(config, storage, fixture_path, monkeypatch):
    monkeypatch.setattr(
        "usain_bot.service.date",
        type("FixedDate", (), {"today": staticmethod(lambda: AS_OF), "fromisoformat": date.fromisoformat}),
    )
    return TestClient(create_app(config, storage, MockGarminAdapter(fixture_path)))


def anchors_at(long_run_mi: float) -> Anchors:
    return Anchors(
        as_of=AS_OF, acute_load_mi=long_run_mi * 1.9, chronic_load_mi=long_run_mi * 2.0,
        long_run_anchor_mi=long_run_mi, adherence_rate=0.95, acwr=0.95,
        gap=GapInfo(gap_days=1, severity=GapSeverity.SHORT, last_run_date=AS_OF - timedelta(days=1)),
    )


def week(n, lr, block="marathon_block", backoff=False, stage=""):
    return PlanWeek(n, date(2026, 7, 27) + timedelta(weeks=n - 1), block, lr * 2.2, lr, 0, backoff, "", stage)


# --- the maintain cycle ------------------------------------------------------

class TestMaintainCycle:
    def test_matches_the_prescribed_pattern(self):
        """The athlete's own example: 22 -> 16 -> 18 -> back-off 15.4 -> 22."""
        peak = 22.0
        got = [round(peak * stages.maintain_long_run_fraction(i, 3), 1) for i in range(8)]
        assert got == [22.0, 16.1, 18.7, 15.4, 22.0, 16.1, 18.7, 15.4]

    def test_cycle_length_follows_backoff_cadence(self):
        assert stages.maintain_cycle_length(3) == 4
        assert stages.maintain_cycle_length(4) == 5

    def test_peak_is_once_per_cycle_and_never_adjacent(self):
        for cadence in (2, 3, 4, 5):
            cycle = stages.maintain_cycle_length(cadence)
            peaks = [stages.is_maintain_peak_week(i, cadence) for i in range(cycle * 3)]
            assert sum(peaks) == 3
            assert not any(a and b for a, b in zip(peaks, peaks[1:]))

    def test_last_week_of_each_cycle_is_the_backoff(self):
        for cadence in (3, 4):
            cycle = stages.maintain_cycle_length(cadence)
            assert stages.is_maintain_backoff_week(cycle - 1, cadence)
            assert not stages.is_maintain_backoff_week(0, cadence)

    def test_volume_holds_while_the_long_run_dips(self):
        """The whole point of maintain: keep mileage high, cut the strain."""
        peak_week = stages.stage_targets(stages.TrainingStage.MAINTAIN_MILEAGE, 22.0, 40.0, 0, 3)
        dip_week = stages.stage_targets(stages.TrainingStage.MAINTAIN_MILEAGE, 22.0, 40.0, 1, 3)
        assert dip_week[0] < peak_week[0]          # long run drops
        assert dip_week[1] == peak_week[1]         # volume does not

    def test_backoff_week_cuts_volume_too(self):
        _lr, vol, is_backoff = stages.stage_targets(
            stages.TrainingStage.MAINTAIN_MILEAGE, 22.0, 40.0, 3, 3)
        assert is_backoff and vol < 40.0


class TestStageClassification:
    def test_reduce_stage_is_below_the_high_mileage_line(self):
        lr, _vol, _b = stages.stage_targets(stages.TrainingStage.REDUCE_MILEAGE, 22.0, 40.0, 0)
        assert lr <= stages.HIGH_MILEAGE_LONG_RUN_MI

    def test_taper_sharpens_toward_the_race(self):
        far = stages.stage_targets(stages.TrainingStage.TAPER, 22.0, 40.0, 0, weeks_remaining_in_stage=3)
        near = stages.stage_targets(stages.TrainingStage.TAPER, 22.0, 40.0, 2, weeks_remaining_in_stage=1)
        assert near[0] < far[0] and near[1] < far[1]

    def test_every_generated_week_carries_a_stage(self, config):
        plan = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t")
        assert all(w.stage in stages.ALL_STAGES for w in plan.weeks)

    def test_stages_are_inferred_for_plans_saved_before_they_existed(self, config):
        legacy = [week(1, 10.0), week(2, 20.0), week(3, 16.0)]
        assert all(not w.stage for w in legacy)
        stages.annotate_stages(legacy)
        assert legacy[0].stage == stages.TrainingStage.INCREASE_MILEAGE.value
        assert legacy[1].stage == stages.TrainingStage.MAINTAIN_MILEAGE.value

    def test_stage_survives_a_save_and_reload(self, config, storage):
        plan = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t")
        storage.save_plan_version(plan)
        reloaded = storage.get_latest_plan_version()
        assert [w.stage for w in reloaded.weeks] == [w.stage for w in plan.weeks]

    def test_summary_collapses_weeks_into_phases(self, config):
        plan = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t")
        summary = stages.stage_summary(plan.weeks)
        assert summary and sum(s["weeks"] for s in summary) == len(plan.weeks)
        assert all(s["stage"] in stages.ALL_STAGES for s in summary)


# --- the hard rule -----------------------------------------------------------

class TestNoBackToBackPeakWeeks:
    @pytest.mark.parametrize("baseline", [8.0, 14.0, 18.0, 20.0, 22.0])
    def test_generated_plans_never_stack_peak_weeks(self, config, baseline):
        plan = planner.generate_macro_plan(config, anchors_at(baseline), AS_OF, 1, "t", "t")
        assert planner.summarize_plan(plan)["max_consecutive_peak_weeks"] <= 1

    def test_an_athlete_already_at_peak_goes_straight_to_maintain(self, config):
        """Nothing left to climb, so the plan must hold — not sit at 22."""
        plan = planner.generate_macro_plan(config, anchors_at(22.0), AS_OF, 1, "t", "t")
        pre = [w for w in plan.weeks if w.block in planner.PRE_MARATHON_BUILD_BLOCKS]
        maintain = [w for w in pre if w.stage == stages.TrainingStage.MAINTAIN_MILEAGE.value]
        assert len(maintain) > len(pre) / 2
        assert planner.summarize_plan(plan)["max_consecutive_peak_weeks"] <= 1

    def test_the_rule_has_one_definition(self):
        """Both the validator and the rules engine call this, so the two
        can never drift apart on what counts as a repeated peak."""
        assert stages.is_repeated_peak(22.0, 22.0, 22.0) is True
        assert stages.is_repeated_peak(22.0, 21.8, 22.0) is True     # within tolerance, flat
        assert stages.is_repeated_peak(21.6, 22.0, 22.0) is False    # still climbing
        assert stages.is_repeated_peak(22.0, 16.0, 22.0) is False    # stepped down
        assert stages.is_repeated_peak(10.0, 10.0, 0.0) is False     # no peak established

    def test_validator_catches_a_stacked_peak(self):
        bad = [week(1, 22.0), week(2, 22.0), week(3, 16.0)]
        issues = validate_plan(bad, date(2027, 2, 7))
        assert any(i.code == "back_to_back_peak" and i.severity == "error" for i in issues)

    def test_validator_accepts_an_oscillating_maintain_block(self):
        good = [week(1, 22.0), week(2, 16.1), week(3, 18.7), week(4, 15.4, backoff=True), week(5, 22.0)]
        for w in good:
            w.stage = stages.TrainingStage.MAINTAIN_MILEAGE.value
        issues = validate_plan(good, date(2027, 2, 7))
        assert not any(i.code == "back_to_back_peak" for i in issues)
        assert not any(i.code == "long_run_jump" for i in issues)

    def test_a_lower_peaking_block_is_judged_on_its_own_peak(self):
        """Two 17-milers in a row in the ultra block is just as wrong as
        two 22s in the marathon block."""
        weeks = [week(1, 22.0, "marathon_block"), week(2, 16.0, "marathon_block"),
                 week(3, 17.0, "ultra_build"), week(4, 17.0, "ultra_build")]
        issues = validate_plan(weeks, date(2027, 2, 7))
        assert any(i.code == "back_to_back_peak" for i in issues)


# --- milestones --------------------------------------------------------------

class TestMilestones:
    def test_dated_and_undated_are_separated(self, config):
        payload = ms.milestones_payload(config, AS_OF)
        assert "marathon" in payload["dated"]
        assert "ultra_50k" in payload["undated"]

    def test_each_milestone_knows_its_prep_target(self, config):
        marathon = ms.milestone_by_name(config, "marathon")
        assert marathon.peak_long_run_mi == 22.0 and marathon.taper_weeks == 2
        ultra = ms.milestone_by_name(config, "ultra_50k")
        assert ultra.peak_long_run_mi == 24.0

    def test_a_dated_milestone_wins_as_next(self, config):
        assert ms.next_milestone(config, AS_OF).name == "marathon"

    def test_undated_milestones_are_always_upcoming(self, config):
        ultra = ms.milestone_by_name(config, "ultra_50k")
        assert ultra.is_upcoming(date(2030, 1, 1)) is True
        assert ultra.weeks_away(AS_OF) is None

    def test_a_past_dated_milestone_stops_being_upcoming(self, config):
        assert "marathon" not in [m.name for m in ms.upcoming_milestones(config, date(2028, 1, 1))]

    def test_taper_start_precedes_the_race_by_the_taper_length(self, config):
        marathon = ms.milestone_by_name(config, "marathon")
        start = ms.taper_start(marathon)
        assert (marathon.target_date - start).days >= 7 * marathon.taper_weeks

    def test_description_states_the_consequence_of_the_date_state(self, config):
        assert "no date set" in ms.milestone_by_name(config, "ultra_50k").describe(AS_OF)
        assert "smooth ramp" in ms.milestone_by_name(config, "marathon").describe(AS_OF)


# --- milestone-shape rules + repair -----------------------------------------

def _generate(config, anchors, as_of, **kwargs):
    kwargs.pop("force_maintain_oscillation", None)
    mode = kwargs.pop("pacing_mode", None)
    if isinstance(mode, str):
        mode = planner.PacingMode(mode)
    return planner.generate_macro_plan(
        config, anchors, as_of, 1, "t", "t",
        pacing_mode=mode or planner.PacingMode.MILESTONE_SMOOTHED, **kwargs)


class TestMilestoneShapeRules:
    @pytest.mark.parametrize("baseline", [8.0, 14.0, 20.0])
    def test_generated_plans_satisfy_the_rules(self, config, baseline):
        _plan, report = rules.generate_valid_plan(_generate, config, anchors_at(baseline), AS_OF)
        assert report.ok, [v.message for v in report.violations]

    def test_undated_milestone_builds_to_its_peak_then_tapers(self, config):
        """Feedback rule 7: no date => conservative increase to the
        milestone's max long run, then straight into the taper."""
        plan, report = rules.generate_valid_plan(_generate, config, anchors_at(8.0), AS_OF)
        assert report.ok
        ultra = ms.milestone_by_name(config, "ultra_50k")
        assert not ultra.has_set_date
        build = [w for w in plan.weeks if w.block == "ultra_build"]
        assert max(w.long_run_mi for w in build) >= ultra.peak_long_run_mi - 0.1
        taper = [w for w in plan.weeks if w.block == "ultra_taper"]
        assert len(taper) == ultra.taper_weeks
        # The taper follows the peak promptly rather than holding for months.
        peak_week = max(build, key=lambda w: w.long_run_mi).week_number
        assert taper[0].week_number - peak_week <= rules.MAX_UNDATED_MAINTAIN_WEEKS + 1

    def test_dated_milestone_ramps_then_maintains_to_the_date(self, config):
        """Feedback rule 7: a set date means smooth ramp, then maintain all
        the way to the milestone — a long maintain phase is the goal, so
        long as those weeks oscillate instead of sitting at peak."""
        plan, report = rules.generate_valid_plan(_generate, config, anchors_at(20.0), AS_OF)
        assert report.ok
        pre = [w for w in plan.weeks if w.block in planner.PRE_MARATHON_BUILD_BLOCKS]
        assert pre[0].stage == stages.TrainingStage.INCREASE_MILEAGE.value
        assert pre[-1].stage == stages.TrainingStage.MAINTAIN_MILEAGE.value
        assert planner.summarize_plan(plan)["max_consecutive_peak_weeks"] <= 1

    def test_a_shortfall_is_reported_not_hidden(self, config):
        marathon = ms.milestone_by_name(config, "marathon")
        short = PlanVersion(1, __import__("datetime").datetime.utcnow(), "t", "r",
                            [week(1, 12.0), week(2, 13.0)])
        violations = rules.check_milestone_shape(
            short.weeks, marathon, ("base_building", "marathon_block"), ("marathon_taper",))
        codes = {v.code for v in violations}
        assert "peak_not_reached" in codes and "missing_taper" in codes

    def test_repair_hints_are_actionable(self, config):
        marathon = ms.milestone_by_name(config, "marathon")
        violations = rules.check_milestone_shape(
            [week(1, 12.0)], marathon, ("base_building", "marathon_block"), ("marathon_taper",))
        peak_miss = next(v for v in violations if v.code == "peak_not_reached")
        assert peak_miss.repair.get("pacing_mode") == "ramp_asap"

    def test_an_athlete_choice_is_honoured_not_overruled(self, config):
        """Ramping slower can cost you the 22 mi prerequisite. The plan
        must still be the one they asked for, with the cost reported as a
        warning — silently ramping them back up would be the coach
        overruling the athlete."""
        constraints = planner.PlanConstraints(ramp_rate=0.5)
        plan, report = planner.revise_plan(
            config, anchors_at(8.0), AS_OF, version=2,
            constraints=constraints, rationale="athlete asked to ramp slower")
        assert report.ok, "an explicit athlete choice must not be an error"
        assert report.warnings, "but its consequence must be surfaced"
        assert "accepted" in report.warnings[0].message
        assert "Trade-offs accepted" in plan.rationale

    def test_the_same_shortfall_is_an_error_when_nobody_asked_for_it(self, config):
        """Without a locked athlete choice, the loop must actually fix it
        rather than shrug."""
        _plan, report = rules.generate_valid_plan(_generate, config, anchors_at(8.0), AS_OF)
        assert report.ok and not report.warnings

    def test_the_repair_loop_cannot_undo_a_persisted_preference(self, service, storage):
        """A published cap is an athlete decision that outlives the draft.
        If the repair loop is allowed to clear it to satisfy a rule, the
        cap silently evaporates on the next reprojection — the exact
        silent-revert the draft/publish flow exists to prevent."""
        service.get_today(as_of=AS_OF)
        service.propose_plan_revision(
            planner.PlanConstraints(peak_long_run_cap=16.0), "cap me", as_of=AS_OF)
        service.publish_draft_plan("yes")
        assert planner.summarize_plan(service.get_plan())["peak_long_run_mi"] <= 16.0

        service._cached = None
        service.refresh_today(as_of=AS_OF)
        assert planner.summarize_plan(service.get_plan())["peak_long_run_mi"] <= 16.0

    def test_the_loop_reports_when_it_cannot_repair(self, config):
        """A generator that always returns a broken plan must surface the
        violations, not quietly hand one back as if it were fine."""
        broken = PlanVersion(1, __import__("datetime").datetime.utcnow(), "t", "r",
                             [week(1, 8.0), week(2, 8.0)])
        _plan, report = rules.generate_valid_plan(
            lambda *a, **k: broken, config, anchors_at(8.0), AS_OF)
        assert not report.ok and report.violations

    def test_checked_generation_is_what_the_agent_uses(self, config):
        plan, report = planner.generate_checked_plan(config, anchors_at(8.0), AS_OF, 1, "t", "rationale")
        assert report.ok
        assert "[!]" not in plan.rationale


# --- pacing phrases ----------------------------------------------------------

class TestPacingIntent:
    @pytest.mark.parametrize("phrase,intent", [
        ("can we ramp up slower?", "slow_down"),
        ("ramp up more gradually", "slow_down"),
        ("let's cool down for a bit", "cool_down"),
        ("time to cool-down", "cool_down"),
        ("ease off a little", "slow_down"),
        ("this is ramping up too fast", "slow_down"),
        ("build me up faster", "speed_up"),
        ("smooth out the ramp", "smooth"),
        ("hold steady for now", "hold"),
        ("stop increasing please", "hold"),
    ])
    def test_recognises_pacing_language(self, phrase, intent):
        got = pacing_intent.interpret_pacing_request(phrase)
        assert got is not None and got.intent == intent

    @pytest.mark.parametrize("phrase", [
        "what should I run today?", "my hip is sore", "how did last week go?", "",
    ])
    def test_ignores_everything_else(self, phrase):
        assert pacing_intent.interpret_pacing_request(phrase) is None

    def test_slower_means_a_gentler_ramp_rate(self):
        got = pacing_intent.interpret_pacing_request("ramp up slower")
        assert got.constraints.ramp_rate < 1.0

    def test_much_slower_is_gentler_than_slower(self):
        mild = pacing_intent.interpret_pacing_request("ramp up slower")
        strong = pacing_intent.interpret_pacing_request("ramp up a lot slower")
        assert strong.constraints.ramp_rate < mild.constraints.ramp_rate

    def test_cool_down_also_caps_the_long_run(self):
        got = pacing_intent.interpret_pacing_request("let's cool down")
        assert got.constraints.peak_long_run_cap == pacing_intent.COOL_DOWN_LONG_RUN_CAP_MI

    def test_explicit_numbers_outrank_the_phrase_default(self):
        base = pacing_intent.constraints_for_intent("cool_down")
        merged = pacing_intent.merge_constraints(base, planner.PlanConstraints(peak_long_run_cap=18.0))
        assert merged.peak_long_run_cap == 18.0
        assert merged.ramp_rate == base.ramp_rate      # phrase still sets the rate

    def test_a_slower_ramp_actually_changes_the_plan(self, config):
        """Governance is worthless if the plan comes back identical."""
        normal = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t")
        slower = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t", ramp_rate=0.5)
        normal_pre = [w.long_run_mi for w in normal.weeks if w.block in planner.PRE_MARATHON_BUILD_BLOCKS]
        slower_pre = [w.long_run_mi for w in slower.weeks if w.block in planner.PRE_MARATHON_BUILD_BLOCKS]
        assert slower_pre != normal_pre
        # A gentler ramp spends fewer weeks above the high-mileage line.
        over = lambda xs: sum(1 for x in xs if x > stages.HIGH_MILEAGE_LONG_RUN_MI)
        assert over(slower_pre) <= over(normal_pre)

    def test_ramp_rate_cannot_break_the_increment_guardrail(self, config):
        """"Build me up faster" gets the guardrail maximum, not more."""
        plan = planner.generate_macro_plan(config, anchors_at(8.0), AS_OF, 1, "t", "t", ramp_rate=4.0)
        marathon = ms.milestone_by_name(config, "marathon")
        errors = [i for i in validate_plan(plan.weeks, marathon.target_date)
                  if i.severity == "error" and i.code == "long_run_jump"]
        assert not errors


class TestPacingThroughTheService:
    def test_a_phrase_becomes_a_real_draft(self, service):
        service.get_today(as_of=AS_OF)
        result = service.propose_pacing_change("can we ramp up slower", as_of=AS_OF)
        assert result is not None
        assert result["draft"] is True
        assert result["pacing_intent"]["intent"] == "slow_down"
        assert result["rules"]["ok"] is True

    def test_an_unrecognised_phrase_returns_nothing_to_act_on(self, service):
        service.get_today(as_of=AS_OF)
        assert service.propose_pacing_change("how's my week looking", as_of=AS_OF) is None

    def test_endpoint_rejects_a_phrase_it_cannot_read(self, client):
        client.get("/api/today")
        assert client.post("/api/plan/revision/pacing", json={"message": "hello"}).status_code == 422

    def test_endpoint_produces_a_draft_without_publishing(self, client):
        client.get("/api/today")
        before = client.get("/api/plan").json()["plan_version"]
        resp = client.post("/api/plan/revision/pacing", json={"message": "let's cool down"})
        assert resp.status_code == 200
        assert resp.json()["pacing_intent"]["intent"] == "cool_down"
        assert client.get("/api/plan").json()["plan_version"] == before


# --- the never-abbreviated plan view -----------------------------------------

class TestPlanRangeView:
    def test_next_n_weeks_returns_exactly_n(self, service):
        service.get_today(as_of=AS_OF)
        payload = service.get_plan_range_payload(weeks_ahead=4, as_of=AS_OF)
        assert payload["week_count"] == 4
        assert payload["truncated"] is False
        assert len(payload["weeks"]) == 4

    def test_between_two_milestones(self, service):
        service.get_today(as_of=AS_OF)
        payload = service.get_plan_range_payload(
            from_milestone="half_marathon", to_milestone="marathon", as_of=AS_OF)
        assert payload["week_count"] > 1
        assert "half marathon" in payload["scope"] and "marathon" in payload["scope"]

    def test_whole_plan_is_never_trimmed(self, service):
        service.get_today(as_of=AS_OF)
        payload = service.get_plan_range_payload(as_of=AS_OF)
        assert payload["week_count"] == payload["total_plan_weeks"]
        assert payload["truncated"] is False

    def test_every_week_carries_its_stage(self, service):
        service.get_today(as_of=AS_OF)
        payload = service.get_plan_range_payload(weeks_ahead=8, as_of=AS_OF)
        assert all(w["stage"] in stages.ALL_STAGES for w in payload["weeks"])

    def test_range_includes_a_phase_summary(self, service):
        service.get_today(as_of=AS_OF)
        payload = service.get_plan_range_payload(as_of=AS_OF)
        assert payload["stages"]
        assert sum(s["weeks"] for s in payload["stages"]) == payload["week_count"]

    def test_unknown_milestone_is_an_error_not_an_empty_list(self, service):
        service.get_today(as_of=AS_OF)
        assert "error" in service.get_plan_range_payload(from_milestone="ironman", as_of=AS_OF)

    def test_endpoint_returns_the_full_range(self, client):
        client.get("/api/today")
        payload = client.get("/api/plan/range?weeks_ahead=6").json()
        assert payload["week_count"] == 6 and payload["truncated"] is False

    def test_endpoint_rejects_a_bad_date(self, client):
        client.get("/api/today")
        assert client.get("/api/plan/range?from_date=notadate").status_code == 400

    def test_milestones_endpoint(self, client):
        body = client.get("/api/milestones").json()
        assert body["next_milestone"] == "marathon"
        assert "ultra_50k" in body["undated"]


class TestPlanRangeChatTool:
    def test_tool_is_registered_and_returns_full_weeks(self, service):
        from usain_bot.chat.tools import TOOL_SPECS, execute_tool

        assert "get_plan_range" in {t.name for t in TOOL_SPECS}
        assert "get_milestones" in {t.name for t in TOOL_SPECS}
        service.get_today(as_of=AS_OF)
        result = json.loads(execute_tool("get_plan_range", {"weeks_ahead": 5}, service))
        assert result["week_count"] == 5

    def test_pacing_intent_is_governed_by_the_tool_layer(self, service):
        from usain_bot.chat.tools import execute_tool

        service.get_today(as_of=AS_OF)
        result = json.loads(execute_tool("propose_plan_revision",
                                          {"pacing_intent": "slow_down", "rationale": "athlete asked"},
                                          service))
        assert result["draft"] is True
        assert any("ramp slowed" in c for c in result["requested_changes"])

    def test_an_invalid_intent_is_rejected(self, service):
        from usain_bot.chat.tools import execute_tool

        service.get_today(as_of=AS_OF)
        result = json.loads(execute_tool("propose_plan_revision", {"pacing_intent": "sprint"}, service))
        assert "error" in result
