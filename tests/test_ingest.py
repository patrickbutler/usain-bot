"""Ingest quality gate: validation and repair that runs automatically on
every write path. Corrupt data must never reach the load anchors, but a
single bad sensor field must never cost a whole legitimate run."""

from datetime import date, datetime, timedelta

import pytest

from usain_bot import agent
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.ingest import find_duplicate_groups, sanitize_activities
from usain_bot.models import Activity, ActivityType
from usain_bot.storage.local import LocalBackend

TODAY = date(2026, 7, 27)


def act(aid="1", d=TODAY, dist=5.0, minutes=50, hr=145, max_hr=175, elev=100.0, start_hour=7):
    return Activity(
        activity_id=aid, date=d, activity_type=ActivityType.RUNNING, distance_mi=dist,
        duration_s=minutes * 60, avg_hr=hr, max_hr=max_hr, elevation_gain_ft=elev, name="Run",
        start_time=datetime(d.year, d.month, d.day, start_hour, 0),
    )


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    cfg.sync.backfill_pause_s = 0
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


class TestAcceptsLegitimateRuns:
    def test_ordinary_run_passes(self):
        report = sanitize_activities([act()], today=TODAY)
        assert len(report.accepted) == 1 and not report.rejected

    def test_fast_track_rep_is_not_rejected(self):
        # 1 mile at 4:30 — fast, but real
        report = sanitize_activities([act(dist=1.0, minutes=4.5)], today=TODAY)
        assert len(report.accepted) == 1

    def test_slow_hike_jog_is_not_rejected(self):
        # 3 miles at 25 min/mi — slow, but real
        report = sanitize_activities([act(dist=3.0, minutes=75)], today=TODAY)
        assert len(report.accepted) == 1

    def test_ultra_distance_is_not_rejected(self):
        report = sanitize_activities([act(dist=50.0, minutes=600)], today=TODAY)
        assert len(report.accepted) == 1

    def test_whole_fixture_passes_the_gate(self, fixture_path):
        adapter = MockGarminAdapter(fixture_path)
        activities = adapter.fetch_activities(date(2026, 1, 1), date(2026, 12, 31))
        report = sanitize_activities(activities, today=TODAY)
        assert report.rejected == []
        assert len(report.accepted) == len(activities)


class TestRejectsUnusableData:
    @pytest.mark.parametrize("kwargs,fragment", [
        ({"dist": -3.0}, "negative"),
        ({"dist": 0.0}, "zero distance"),
        ({"minutes": 0}, "non-positive duration"),
        ({"dist": 400.0, "minutes": 3000}, "implausible distance"),
        ({"dist": 5.0, "minutes": 1}, "implausible pace"),          # 12s/mi — GPS teleport
        ({"dist": 1.0, "minutes": 120}, "implausible pace"),        # 2h for 1 mile
    ])
    def test_rejects(self, kwargs, fragment):
        report = sanitize_activities([act(**kwargs)], today=TODAY)
        assert report.accepted == []
        assert fragment in report.rejected[0][1]

    def test_rejects_future_dated_activity(self):
        report = sanitize_activities([act(d=TODAY + timedelta(days=10))], today=TODAY)
        assert report.accepted == []
        assert "future" in report.rejected[0][1]

    def test_tolerates_small_clock_skew(self):
        report = sanitize_activities([act(d=TODAY + timedelta(days=1))], today=TODAY)
        assert len(report.accepted) == 1

    def test_rejects_activity_without_id(self):
        bad = Activity(activity_id="", date=TODAY, activity_type=ActivityType.RUNNING,
                        distance_mi=5.0, duration_s=3000)
        assert sanitize_activities([bad], today=TODAY).accepted == []

    def test_drops_duplicate_ids_within_one_batch(self):
        report = sanitize_activities([act("1"), act("1")], today=TODAY)
        assert len(report.accepted) == 1
        assert "duplicate activity_id" in report.rejected[0][1]

    def test_one_bad_activity_does_not_discard_the_batch(self):
        report = sanitize_activities([act("1"), act("2", dist=-1.0), act("3")], today=TODAY)
        assert [a.activity_id for a in report.accepted] == ["1", "3"]
        assert report.rejected_count == 1


class TestRepairsRatherThanRejects:
    def test_impossible_avg_hr_is_nulled_run_is_kept(self):
        report = sanitize_activities([act(hr=400)], today=TODAY)
        assert len(report.accepted) == 1
        assert report.accepted[0].avg_hr is None
        assert report.repaired_count == 1

    def test_max_hr_below_avg_hr_is_nulled(self):
        report = sanitize_activities([act(hr=150, max_hr=120)], today=TODAY)
        assert report.accepted[0].max_hr is None
        assert report.accepted[0].avg_hr == 150

    def test_negative_elevation_is_nulled(self):
        report = sanitize_activities([act(elev=-500.0)], today=TODAY)
        assert report.accepted[0].elevation_gain_ft is None

    def test_pace_is_recomputed_from_distance_and_duration(self):
        # a stale/edited pace field must not survive
        a = act(dist=5.0, minutes=50)
        a = Activity(**{**a.__dict__, "avg_pace_min_per_mi": 99.0})
        report = sanitize_activities([a], today=TODAY)
        assert report.accepted[0].avg_pace_min_per_mi == pytest.approx(10.0)

    def test_clean_activity_is_returned_unchanged(self):
        a = act()
        a = Activity(**{**a.__dict__, "avg_pace_min_per_mi": 10.0})
        report = sanitize_activities([a], today=TODAY)
        assert report.accepted[0] is a
        assert report.repaired_count == 0


class TestDuplicateGrouping:
    def test_same_run_two_ids_grouped(self):
        groups = find_duplicate_groups([act("1"), act("2", dist=5.02)])
        assert len(groups) == 1 and len(groups[0]) == 2

    def test_split_run_parts_not_grouped(self):
        groups = find_duplicate_groups([
            act("1", dist=6.2, minutes=62, start_hour=7),
            act("2", dist=3.8, minutes=38, start_hour=8),
        ])
        assert groups == []

    def test_distinct_runs_same_day_not_grouped(self):
        groups = find_duplicate_groups([
            act("1", dist=10.0, minutes=100, start_hour=7),
            act("2", dist=3.0, minutes=30, start_hour=18),
        ])
        assert groups == []

    def test_grouping_is_deterministic(self):
        a, b = act("1"), act("2")
        assert find_duplicate_groups([a, b])[0][0].activity_id == \
               find_duplicate_groups([b, a])[0][0].activity_id


class TestQualityGateIsWiredIntoWritePaths:
    def test_sync_rejects_corrupt_activity_before_storage(self, storage):
        adapter = MockGarminAdapter.from_activities([act("1"), act("2", dist=999.0, minutes=6000)])
        result = agent.sync_activities(storage, adapter, TODAY)
        assert result.rejected_count == 1
        assert [a.activity_id for a in storage.get_activities()] == ["1"]

    def test_backfill_rejects_corrupt_activity(self, config, storage):
        adapter = MockGarminAdapter.from_activities([act("1"), act("2", dist=-5.0)])
        result = agent.backfill_history(config, storage, adapter, as_of=TODAY, sleep_fn=lambda s: None)
        assert result.rejected_count == 1
        assert len(storage.get_activities()) == 1

    def test_corrupt_data_never_reaches_the_anchors(self, storage):
        """The reason the gate exists: one bogus 300-mile run would wreck
        every load anchor and the plan built on them."""
        adapter = MockGarminAdapter.from_activities([
            act("1", d=TODAY - timedelta(days=2), dist=5.0, minutes=50),
            act("2", d=TODAY - timedelta(days=1), dist=300.0, minutes=3000),
        ])
        agent.sync_activities(storage, adapter, TODAY)
        from usain_bot.classification import compute_anchors, prepare_classified
        anchors = compute_anchors(prepare_classified(storage.get_activities()), TODAY)
        assert anchors.long_run_anchor_mi < 50
        assert anchors.acute_load_mi == pytest.approx(5.0)

    def test_sync_message_mentions_data_quality_when_relevant(self, storage):
        adapter = MockGarminAdapter.from_activities([act("1"), act("2", dist=0.0)])
        result = agent.sync_activities(storage, adapter, TODAY)
        assert "Data quality" in result.message

    def test_sync_message_stays_quiet_on_clean_data(self, storage, fixture_path):
        result = agent.sync_activities(storage, MockGarminAdapter(fixture_path), TODAY)
        assert "Data quality" not in result.message


class TestColdStartRecommendation:
    """A brand-new athlete (or one whose Garmin has never synced) must get
    a usable conservative number, not '0.0 mi' — while an athlete who has
    genuinely used up their weekly cap must still be told to rest."""

    def _service(self, config, storage, activities):
        from usain_bot.service import CoachService
        return CoachService(config, storage, MockGarminAdapter.from_activities(activities))

    def test_no_history_does_not_recommend_zero(self, config, storage):
        svc = self._service(config, storage, [])
        rec = svc.get_today(TODAY).recommendation
        assert rec.target_distance_mi > 0
        assert rec.binding_constraint.startswith("cold_start_baseline")

    def test_cold_start_number_is_derived_from_configured_baseline(self, config, storage):
        config.athlete.baseline_long_run_mi = 10.0
        svc = self._service(config, storage, [])
        assert svc.get_today(TODAY).recommendation.target_distance_mi == pytest.approx(4.0)

    def test_cold_start_says_plainly_it_is_not_evidence(self, config, storage):
        svc = self._service(config, storage, [])
        rec = svc.get_today(TODAY).recommendation
        assert any("not from your actual training" in r for r in rec.reasoning)
        assert "backfill" in rec.unlock_next_time

    def test_real_history_is_unaffected_by_the_cold_start_branch(self, config, storage, fixture_path):
        from usain_bot.service import CoachService
        svc = CoachService(config, storage, MockGarminAdapter(fixture_path))
        rec = svc.get_today(TODAY).recommendation
        assert not rec.binding_constraint.startswith("cold_start_baseline")

    def test_exhausted_weekly_cap_can_still_recommend_rest(self, config, storage):
        """Regression guard: the cold-start floor must not mask a genuine
        'you've hit your ceiling' zero."""
        big = [act(f"x{i}", d=TODAY - timedelta(days=i), dist=12.0, minutes=120) for i in range(1, 7)]
        svc = self._service(config, storage, big)
        rec = svc.get_today(TODAY).recommendation
        assert not rec.binding_constraint.startswith("cold_start_baseline")
