from datetime import date, datetime
from pathlib import Path

import pytest

from usain_bot import agent
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import ReferenceDoc
from usain_bot.storage.local import LocalBackend

REFERENCES_DIR = Path(__file__).parent.parent / "references"


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = get_storage_backend_local(config)
    yield backend
    backend.close()


def get_storage_backend_local(config) -> LocalBackend:
    return LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)


@pytest.fixture
def adapter(fixture_path) -> MockGarminAdapter:
    return MockGarminAdapter(fixture_path)


class TestRunInvocation:
    def test_produces_a_recommendation(self, config, storage, adapter):
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        assert result.recommendation.target_distance_mi is not None
        assert result.recommendation.target_distance_mi >= 0
        assert result.recommendation.binding_constraint
        assert result.recommendation.run_type in ("long", "quality", "easy")

    def test_persists_plan_and_activities_when_not_dry_run(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        assert storage.get_latest_plan_version() is not None
        assert len(storage.get_activities()) > 0
        assert storage.get_last_sync_time() is not None

    def test_dry_run_does_not_persist_plan(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24), dry_run=True)
        # activities/sync state are still recorded (sync happens regardless);
        # only the plan version + conversation entry are gated by dry_run.
        assert storage.get_latest_plan_version() is None

    def test_health_flag_forces_easy_and_is_persisted(self, config, storage, adapter):
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24), health_flag="hip")
        assert result.recommendation.run_type == "easy"
        flags = storage.get_recent_health_flags()
        assert any(f.flag == "hip" for f in flags)

    def test_second_invocation_reuses_cached_sync_window(self, config, storage, adapter):
        first = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        assert first.sync.live
        second = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        assert second.sync.live
        # second sync's window starts at/after the first sync's timestamp, so
        # no duplicate activities should be re-inserted.
        assert second.sync.activities_count == 0

    def test_recommendation_never_exceeds_hard_weekly_cap(self, config, storage, adapter):
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        max_allowed = result.anchors.chronic_load_mi * 1.10
        assert result.recommendation.target_distance_mi <= max_allowed + 1e-6

    def test_plan_version_increments_across_invocations(self, config, storage, adapter):
        r1 = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        r2 = agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 25))
        assert r2.plan.version == r1.plan.version + 1


class TestGapScenarios:
    def test_invocation_long_after_history_triggers_gap_protocol(self, config, storage, adapter):
        # last fixture activity is 2026-07-22; invoke 40 days later.
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 8, 31))
        assert result.recommendation.run_type == "easy"
        assert result.anchors.gap.severity.value in ("long", "severe")

    def test_severe_gap_regenerates_plan_from_current_actuals(self, config, storage, adapter):
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 10, 1))
        assert result.plan.trigger == "gap_regeneration"


class TestReferenceCitation:
    def test_gap_regeneration_cites_a_relevant_reference(self, config, storage, adapter):
        doc_path = REFERENCES_DIR / "return-to-running-after-a-break.md"
        storage.save_reference(ReferenceDoc(
            doc_id="return-to-running", title="Returning to Running After a Break",
            source=str(doc_path), added_at=datetime.utcnow(), content=doc_path.read_text(),
        ))
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 10, 1))
        assert result.plan.trigger == "gap_regeneration"
        assert "Per reference" in result.plan.rationale

    def test_no_citation_when_references_store_is_empty(self, config, storage, adapter):
        result = agent.run_invocation(config, storage, adapter, as_of=date(2026, 10, 1))
        assert "Per reference" not in result.plan.rationale


class TestFirstRunFlow:
    def test_first_run_report_and_confirm(self, config, storage, adapter):
        report = agent.first_run_report(config, storage, adapter, as_of=date(2026, 7, 24))
        assert report.anchors.chronic_load_mi > 0
        assert len(report.proposed_plan.weeks) > 10
        assert storage.get_latest_plan_version() is None  # not persisted yet

        agent.confirm_first_run(storage, report)
        latest = storage.get_latest_plan_version()
        assert latest is not None
        assert latest.version == 1


class TestOverrideFlow:
    def test_override_without_existing_plan_fails_gracefully(self, config, storage):
        result = agent.override_plan(config, storage, "make next week easier", as_of=date(2026, 7, 24))
        assert not result.applied

    def test_override_after_plan_exists(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        result = agent.override_plan(config, storage, "make next week easier", as_of=date(2026, 7, 24))
        assert result.applied
        history = storage.get_plan_history()
        assert len(history) >= 2

    def test_override_logs_conversation(self, config, storage, adapter):
        agent.run_invocation(config, storage, adapter, as_of=date(2026, 7, 24))
        agent.override_plan(config, storage, "make next week easier", as_of=date(2026, 7, 24))
        convo = storage.get_conversation_history()
        assert any(c.role == "user" and "easier" in c.text for c in convo)
