"""Data-pipeline tests: incremental sync with an edit-catching overlap
window, full-history backfill with rate-limit backoff, duplicate
detection/resolution, and the running-only activity-type filter."""

from datetime import date, datetime, timedelta

import pytest

from usain_bot import agent
from usain_bot.config import load_config
from usain_bot.garmin_adapter.base import GarminUnavailableError
from usain_bot.garmin_adapter.live import INCLUDED_TYPE_KEYS, GarminConnectAdapter, _is_rate_limited
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import Activity, ActivityType
from usain_bot.storage.local import LocalBackend


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    cfg.sync.backfill_pause_s = 0  # no real sleeping in tests
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


def act(aid, d: date, dist, minutes=60, name="Run", start_hour=7):
    return Activity(
        activity_id=aid, date=d, activity_type=ActivityType.RUNNING, distance_mi=dist,
        duration_s=minutes * 60, name=name,
        start_time=datetime(d.year, d.month, d.day, start_hour, 0),
    )


class TestUpsertAndEditDetection:
    def test_new_activities_inserted(self, storage):
        new, updated = storage.save_activities([act("1", date(2026, 7, 20), 5.0)])
        assert (new, updated) == (1, 0)

    def test_resaving_identical_activity_is_a_noop(self, storage):
        a = act("1", date(2026, 7, 20), 5.0)
        storage.save_activities([a])
        new, updated = storage.save_activities([a])
        assert (new, updated) == (0, 0)
        assert len(storage.get_activities()) == 1

    def test_edited_distance_updates_existing_row(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 5.0)])
        new, updated = storage.save_activities([act("1", date(2026, 7, 20), 5.6)])
        assert (new, updated) == (0, 1)
        assert storage.get_activities()[0].distance_mi == pytest.approx(5.6)

    def test_edited_name_and_duration_update(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 5.0, minutes=50, name="Run")])
        _, updated = storage.save_activities([act("1", date(2026, 7, 20), 5.0, minutes=48, name="Tempo")])
        assert updated == 1
        stored = storage.get_activities()[0]
        assert stored.name == "Tempo" and stored.duration_s == 48 * 60

    def test_start_time_round_trips(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 5.0, start_hour=6)])
        assert storage.get_activities()[0].start_time == datetime(2026, 7, 20, 6, 0)


class TestSyncOverlapWindow:
    def test_first_sync_uses_full_lookback(self, storage, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        result = agent.sync_activities(storage, adapter, date(2026, 7, 27))
        assert result.live and result.activities_count > 0

    def test_second_sync_repulls_overlap_window_and_catches_edits(self, storage):
        original = act("1", date(2026, 7, 20), 5.0)
        adapter = MockGarminAdapter.from_activities([original])
        agent.sync_activities(storage, adapter, date(2026, 7, 21), overlap_days=30)

        # Athlete edits the activity on Garmin after it was first synced.
        edited = act("1", date(2026, 7, 20), 5.75)
        adapter2 = MockGarminAdapter.from_activities([edited])
        result = agent.sync_activities(storage, adapter2, date(2026, 7, 25), overlap_days=30)

        assert result.updated_count == 1
        assert storage.get_activities()[0].distance_mi == pytest.approx(5.75)

    def test_narrow_overlap_window_misses_older_edits(self, storage):
        """Documents the tradeoff: overlap_days bounds how far back an edit
        can be detected."""
        original = act("1", date(2026, 6, 1), 5.0)
        adapter = MockGarminAdapter.from_activities([original])
        agent.sync_activities(storage, adapter, date(2026, 7, 20), overlap_days=1)

        edited = act("1", date(2026, 6, 1), 9.0)
        adapter2 = MockGarminAdapter.from_activities([edited])
        result = agent.sync_activities(storage, adapter2, date(2026, 7, 21), overlap_days=1)
        assert result.updated_count == 0  # outside the window, not re-requested

    def test_garmin_unavailable_degrades_gracefully(self, storage):
        class Broken:
            def fetch_activities(self, s, e):
                raise GarminUnavailableError("429 Too Many Requests")

        result = agent.sync_activities(storage, Broken(), date(2026, 7, 27))
        assert result.live is False
        assert "conservative" in result.message


class TestBackfill:
    def test_backfill_imports_full_history(self, config, storage, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        result = agent.backfill_history(config, storage, adapter, as_of=date(2026, 7, 27), sleep_fn=lambda s: None)
        assert result.complete
        assert result.new_count > 0
        assert result.earliest_date is not None
        assert len(storage.get_activities()) == result.new_count

    def test_backfill_is_idempotent(self, config, storage, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        first = agent.backfill_history(config, storage, adapter, as_of=date(2026, 7, 27), sleep_fn=lambda s: None)
        second = agent.backfill_history(config, storage, adapter, as_of=date(2026, 7, 27), sleep_fn=lambda s: None)
        assert second.new_count == 0
        assert len(storage.get_activities()) == first.new_count

    def test_backfill_stops_after_empty_chunks(self, config, storage):
        adapter = MockGarminAdapter.from_activities([act("1", date(2026, 7, 20), 5.0)])
        result = agent.backfill_history(config, storage, adapter, as_of=date(2026, 7, 27), sleep_fn=lambda s: None)
        assert result.complete
        # stops well short of max_years once it hits consecutive empty windows
        assert result.chunks_fetched <= 1 + config.sync.backfill_max_empty_chunks

    def test_backfill_reports_incomplete_on_rate_limit(self, config, storage):
        class RateLimited:
            def __init__(self):
                self.calls = 0

            def fetch_activities(self, s, e):
                self.calls += 1
                if self.calls > 1:
                    raise GarminUnavailableError("429 Too Many Requests after retries")
                return [act("1", e - timedelta(days=1), 5.0)]

        result = agent.backfill_history(config, storage, RateLimited(), as_of=date(2026, 7, 27), sleep_fn=lambda s: None)
        assert result.complete is False
        assert "re-run" in result.message


class TestDedupe:
    def test_no_duplicates_in_clean_data(self, storage, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        storage.save_activities(adapter.fetch_activities(date(2026, 1, 1), date(2026, 12, 31)))
        result = agent.dedupe_activities(storage)
        assert result.groups == []

    def test_detects_same_run_under_two_ids(self, storage):
        storage.save_activities([
            act("1", date(2026, 7, 20), 6.0, start_hour=7),
            act("2", date(2026, 7, 20), 6.02, start_hour=7),  # same run, re-uploaded
        ])
        result = agent.dedupe_activities(storage)
        assert len(result.groups) == 1
        assert result.duplicate_count == 1

    def test_dry_run_does_not_delete(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 6.0), act("2", date(2026, 7, 20), 6.0)])
        agent.dedupe_activities(storage, apply=False)
        assert len(storage.get_activities()) == 2

    def test_apply_removes_duplicates_keeping_one(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 6.0), act("2", date(2026, 7, 20), 6.0)])
        result = agent.dedupe_activities(storage, apply=True)
        assert len(result.removed) == 1
        assert len(storage.get_activities()) == 1

    def test_split_run_parts_are_not_treated_as_duplicates(self, storage):
        # Different start times and distances -> two real recordings that
        # sessions.py merges at read time, NOT storage duplicates to delete.
        storage.save_activities([
            act("1", date(2026, 7, 20), 6.2, minutes=62, start_hour=7),
            act("2", date(2026, 7, 20), 3.8, minutes=38, start_hour=8),
        ])
        result = agent.dedupe_activities(storage)
        assert result.groups == []

    def test_different_days_never_grouped(self, storage):
        storage.save_activities([act("1", date(2026, 7, 20), 6.0), act("2", date(2026, 7, 21), 6.0)])
        assert agent.dedupe_activities(storage).groups == []


class TestActivityTypeFilter:
    def test_included_type_keys_cover_required_running_types(self):
        assert {"running", "trail_running", "treadmill_running"} <= INCLUDED_TYPE_KEYS

    def test_cycling_and_strength_are_excluded(self):
        assert "cycling" not in INCLUDED_TYPE_KEYS
        assert "strength_training" not in INCLUDED_TYPE_KEYS

    def test_mock_adapter_filters_non_running(self, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        fetched = adapter.fetch_activities(date(2026, 1, 1), date(2026, 12, 31))
        assert fetched, "fixture should contain running activities"
        assert all(a.activity_type == ActivityType.RUNNING for a in fetched)

    def test_fixture_actually_contains_non_running_rows_to_filter(self, fixture_path):
        import json
        raw = json.loads(fixture_path.read_text())
        assert any(r["activity_type"] != "running" for r in raw), \
            "fixture must include non-running rows or the filter test proves nothing"


class TestRateLimitDetection:
    @pytest.mark.parametrize("message", [
        "429 Too Many Requests", "HTTP 429", "rate limit exceeded", "Too Many Requests",
    ])
    def test_recognizes_rate_limit_errors(self, message):
        assert _is_rate_limited(Exception(message))

    def test_does_not_flag_generic_errors(self):
        assert not _is_rate_limited(Exception("connection reset"))

    def test_retries_with_backoff_then_raises(self):
        slept: list[float] = []

        class Creds:
            email = "x"; password = "y"; token_store = "/tmp/none"

        adapter = GarminConnectAdapter(Creds(), sleep_fn=slept.append)

        class Client:
            def get_activities_by_date(self, s, e):
                raise Exception("429 Too Many Requests")

        adapter._client = Client()
        with pytest.raises(GarminUnavailableError):
            adapter.fetch_activities(date(2026, 7, 1), date(2026, 7, 27))
        assert slept, "should have backed off before giving up"
        assert slept == sorted(slept), "backoff should be non-decreasing"
        assert max(slept) >= 30, "rate-limit backoff should use the long schedule"
