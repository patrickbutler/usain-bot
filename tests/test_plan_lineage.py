from datetime import date

import pytest

from usain_bot import agent
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import Anchors, GapInfo, GapSeverity
from usain_bot.planner import WeekDiff, diff_plan_versions, diff_plan_weeks, generate_macro_plan
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def warm_anchors() -> Anchors:
    return Anchors(
        as_of=date(2026, 7, 24), acute_load_mi=22.0, chronic_load_mi=24.0,
        long_run_anchor_mi=8.0, adherence_rate=0.9, acwr=22.0 / 24.0,
        gap=GapInfo(gap_days=1, severity=GapSeverity.SHORT, last_run_date=date(2026, 7, 23)),
    )


class TestDiffPlanWeeks:
    def test_first_plan_all_added(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        diffs = diff_plan_weeks(None, plan)
        assert len(diffs) == len(plan.weeks)
        assert all(d.change_type == "added" for d in diffs)

    def test_identical_plans_all_unchanged(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        plan2 = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 2, "test", "test")
        diffs = diff_plan_weeks(plan, plan2)
        assert all(d.change_type == "unchanged" for d in diffs)

    def test_changed_week_reports_changed_fields(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        anchors2 = Anchors(**{**warm_anchors.__dict__, "long_run_anchor_mi": 11.0})
        plan2 = generate_macro_plan(config, anchors2, date(2026, 7, 24), 2, "test", "test")
        diffs = diff_plan_weeks(plan, plan2)
        changed = [d for d in diffs if d.change_type == "changed"]
        assert changed
        assert "long_run_mi" in changed[0].changed_fields or "target_volume_mi" in changed[0].changed_fields

    def test_week_diff_to_dict_roundtrips_shape(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        d = diff_plan_weeks(None, plan)[0]
        payload = d.to_dict()
        assert payload["week_number"] == d.week_number
        assert payload["change_type"] == "added"
        assert payload["old"] is None
        assert payload["new"]["week_number"] == d.week_number

    def test_diff_plan_versions_text_still_matches_structured_diff(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        anchors2 = Anchors(**{**warm_anchors.__dict__, "long_run_anchor_mi": 11.0})
        plan2 = generate_macro_plan(config, anchors2, date(2026, 7, 24), 2, "test", "test")
        text = diff_plan_versions(plan, plan2)
        structured_changed = [d for d in diff_plan_weeks(plan, plan2) if d.change_type != "unchanged"]
        for d in structured_changed:
            assert f"week {d.week_number}" in text


class TestAgentRecordsDiffOnEveryVersion:
    @pytest.fixture
    def storage(self, config) -> LocalBackend:
        backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
        yield backend
        backend.close()

    @pytest.fixture
    def adapter(self, fixture_path) -> MockGarminAdapter:
        return MockGarminAdapter(fixture_path)

    def test_first_version_gets_initial_plan_diff(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        v1 = storage.get_plan_history()[0]
        assert v1.diff_from_prior is not None
        assert "Initial plan" in v1.diff_from_prior

    def test_second_reprojection_gets_a_diff_against_first(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 25))
        history = storage.get_plan_history()
        assert len(history) == 2
        assert history[1].diff_from_prior is not None
        assert "v1 -> v2" in history[1].diff_from_prior

    def test_dry_run_still_computes_diff_but_does_not_persist(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 25), dry_run=True)
        assert result.plan.diff_from_prior is not None
        assert len(storage.get_plan_history()) == 1  # dry run didn't persist v2

    def test_first_run_report_also_gets_diff(self, config, storage, adapter):
        report = agent.first_run_report(config, storage, adapter, as_of=date(2026, 7, 24))
        assert report.proposed_plan.diff_from_prior is not None
        assert "Initial plan" in report.proposed_plan.diff_from_prior


class TestPlanHistoryPayload:
    @pytest.fixture
    def storage(self, config) -> LocalBackend:
        backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
        yield backend
        backend.close()

    @pytest.fixture
    def service(self, config, storage, fixture_path) -> CoachService:
        return CoachService(config, storage, MockGarminAdapter(fixture_path))

    def test_empty_history(self, service):
        payload = service.get_plan_history_payload()
        assert payload["versions"] == []

    def test_newest_first(self, service):
        service.get_today(date(2026, 7, 24))
        service.refresh_today(date(2026, 7, 25))
        payload = service.get_plan_history_payload()
        versions = [v["version"] for v in payload["versions"]]
        assert versions == sorted(versions, reverse=True)

    def test_each_version_has_week_diffs_and_counts(self, service):
        service.get_today(date(2026, 7, 24))
        payload = service.get_plan_history_payload()
        v1 = payload["versions"][0]
        assert v1["weeks_changed_count"] == len(v1["week_diffs"])
        assert all(w["change_type"] == "added" for w in v1["week_diffs"])

    def test_override_shows_up_as_its_own_version_with_diff(self, service):
        from usain_bot import planner

        service.get_today(date(2026, 7, 24))
        plan = service.get_plan()
        result = planner.ease_week(plan, None, date(2026, 7, 24), "test override")
        result.plan.diff_from_prior = planner.diff_plan_versions(plan, result.plan)
        service.storage.save_plan_version(result.plan)
        service.apply_plan_update(result.plan)

        payload = service.get_plan_history_payload()
        latest = payload["versions"][0]
        assert latest["trigger"] == "user_override"
        assert latest["weeks_changed_count"] >= 1
