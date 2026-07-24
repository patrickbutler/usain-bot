from datetime import date

import pytest

from usain_bot.chat import tools
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import ReferenceDoc
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


@pytest.fixture
def service(config, storage, fixture_path) -> CoachService:
    return CoachService(config, storage, MockGarminAdapter(fixture_path))


class TestReadTools:
    def test_get_today_recommendation(self, service):
        result = tools.tool_get_today_recommendation(service, {})
        assert "recommendation" in result
        assert len(result["next_7_days"]) == 7

    def test_get_plan_overview_before_any_computation_errors_cleanly(self, service):
        result = tools.tool_get_plan_overview(service, {})
        assert "error" in result

    def test_get_plan_overview_after_today(self, service):
        service.get_today(date(2026, 7, 24))
        result = tools.tool_get_plan_overview(service, {})
        assert result["plan_version"] >= 1
        assert len(result["weeks"]) > 10

    def test_get_plan_history_empty(self, service):
        result = tools.tool_get_plan_history(service, {})
        assert result["versions"] == []

    def test_get_plan_history_after_changes(self, service):
        service.get_today(date(2026, 7, 24))
        service.refresh_today(date(2026, 7, 25))
        result = tools.tool_get_plan_history(service, {})
        assert len(result["versions"]) == 2
        assert result["versions"][0]["version"] == 2  # newest first
        assert "diff_text" in result["versions"][0]

    def test_get_plan_history_respects_limit(self, service):
        service.get_today(date(2026, 7, 24))
        for i in range(5):
            service.refresh_today(date(2026, 7, 25))
        result = tools.tool_get_plan_history(service, {"limit": 2})
        assert len(result["versions"]) == 2

    def test_get_run_history(self, service):
        service.sync(date(2026, 7, 24))
        result = tools.tool_get_run_history(service, {"days": 90})
        assert result["total_activities"] > 0
        assert "long" in result["count_by_class"] or "easy" in result["count_by_class"]

    def test_get_run_history_days_clamped(self, service):
        result = tools.tool_get_run_history(service, {"days": 99999})
        assert result["window_days"] == 365


class TestMutationTools:
    def test_ease_upcoming_week_without_plan_errors(self, service):
        result = tools.tool_ease_upcoming_week(service, {})
        assert result.get("applied") is False or "error" in result

    def test_ease_upcoming_week(self, service):
        service.get_today(date(2026, 7, 24))
        result = tools.tool_ease_upcoming_week(service, {})
        assert result["applied"] is True
        assert result["new_plan_version"] >= 2

    def test_shift_marathon_date(self, service):
        service.get_today(date(2026, 7, 24))
        result = tools.tool_shift_marathon_date(service, {"weeks": 2})
        assert result["applied"] is True
        assert any("2027-02-21" in w for w in result["warnings"])

    def test_set_long_run_day_preference(self, service):
        service.get_today(date(2026, 7, 24))
        result = tools.tool_set_long_run_day_preference(service, {"day": "Sunday"})
        assert result["applied"] is True

    def test_set_health_flag_forces_easy(self, service):
        service.get_today(date(2026, 7, 24))
        result = tools.tool_set_health_flag(service, {"flag": "hip", "note": "twinge"})
        assert result["flag_set"] == "hip"
        assert result["today_recommendation"]["run_type"] == "easy"

    def test_set_health_flag_rejects_unknown_flag(self, service):
        result = tools.tool_set_health_flag(service, {"flag": "elbow"})
        assert "error" in result


class TestReferenceTool:
    def test_search_with_no_references(self, service):
        result = tools.tool_search_coaching_references(service, {"query": "ACWR"})
        assert result["results"] == []

    def test_search_after_adding_reference(self, service):
        service.storage.save_reference(ReferenceDoc(
            doc_id="acwr-doc", title="ACWR guidance", source="test",
            added_at=__import__("datetime").datetime.utcnow(),
            content="The acute chronic workload ratio is a directional signal, not a sole gate.",
        ))
        result = tools.tool_search_coaching_references(service, {"query": "acute chronic workload ratio"})
        assert len(result["results"]) > 0


class TestSyncTool:
    def test_trigger_garmin_sync(self, service):
        result = tools.tool_trigger_garmin_sync(service, {})
        assert result["live"] is True


class TestExecuteToolDispatch:
    def test_unknown_tool_returns_error_json(self, service):
        import json
        raw = tools.execute_tool("not_a_real_tool", {}, service)
        assert json.loads(raw)["error"]

    def test_missing_required_arg_returns_error_not_crash(self, service):
        import json
        raw = tools.execute_tool("shift_marathon_date", {}, service)
        assert "error" in json.loads(raw)
