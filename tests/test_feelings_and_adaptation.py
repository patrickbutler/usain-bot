"""Coach adaptation: subjective run-feeling memory feeding the distance
ceiling, running frequency derived from actuals, flexible-milestone
pushes, and health-flag undo."""

from datetime import date, datetime, timedelta

import pytest

from usain_bot import agent
from usain_bot.classification import derive_run_days_per_week, prepare_classified
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.models import Activity, ActivityType, RunFeedback
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend

TODAY = date(2026, 7, 27)


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


def run_on(d: date, dist=5.0, hour=7):
    return Activity(
        activity_id=f"r{d.isoformat()}-{hour}", date=d, activity_type=ActivityType.RUNNING,
        distance_mi=dist, duration_s=int(dist * 9.5 * 60), avg_hr=145, max_hr=178, name="Run",
        start_time=datetime(d.year, d.month, d.day, hour, 0),
    )


class TestRunFeedbackStorage:
    def test_save_and_read_back(self, storage):
        storage.save_run_feedback(RunFeedback(datetime.utcnow(), 4, "felt smooth", date(2026, 7, 25)))
        got = storage.get_recent_run_feedback()
        assert len(got) == 1 and got[0].score == 4 and got[0].comment == "felt smooth"

    def test_old_feedback_falls_outside_window(self, storage):
        storage.save_run_feedback(RunFeedback(datetime.utcnow() - timedelta(days=40), 2, "old"))
        assert storage.get_recent_run_feedback(days=14) == []

    def test_returns_most_recent_first(self, storage):
        storage.save_run_feedback(RunFeedback(datetime.utcnow() - timedelta(days=3), 2, "older"))
        storage.save_run_feedback(RunFeedback(datetime.utcnow(), 5, "newer"))
        assert storage.get_recent_run_feedback()[0].comment == "newer"

    def test_window_is_applied_before_the_limit(self, storage):
        """Filtering after the LIMIT silently drops in-window entries once
        the table outgrows the limit, which makes rated runs look unrated."""
        for i in range(30):
            storage.save_run_feedback(
                RunFeedback(datetime.utcnow() - timedelta(minutes=i), 3, f"e{i}", date(2026, 7, 1)))
        storage.save_run_feedback(RunFeedback(datetime.utcnow() - timedelta(days=90), 1, "ancient"))
        got = storage.get_recent_run_feedback(days=14, limit=100)
        assert len(got) == 30
        assert "ancient" not in [f.comment for f in got]


class TestUndatedScoresBindToARun:
    """A score logged without a date — "that one was rough", the normal
    chat phrasing — must attach to an actual run. Stored dateless it can
    never mark a run rated, so the coach re-asks about runs the athlete
    already answered for while the feelings list shows the score against
    its recording date and looks correct."""

    def test_undated_score_attaches_to_the_latest_unrated_run(self, service):
        service.sync(as_of=TODAY)
        before = [r["date"] for r in service.get_unrated_recent_runs(days=10)]
        assert before, "fixture must supply unrated runs for this test to mean anything"

        fb = service.record_run_feeling(2, "rough one")
        assert fb.activity_date is not None
        assert fb.activity_date.isoformat() == before[0]

    def test_the_rated_run_stops_being_asked_about(self, service):
        service.sync(as_of=TODAY)
        before = service.get_unrated_recent_runs(days=10)
        service.record_run_feeling(2, "rough one")
        after = [r["date"] for r in service.get_unrated_recent_runs(days=10)]
        assert before[0]["date"] not in after
        assert len(after) < len(before)

    def test_consecutive_undated_scores_walk_back_through_runs(self, service):
        service.sync(as_of=TODAY)
        first = service.record_run_feeling(2, "rough")
        second = service.record_run_feeling(4, "much better")
        assert first.activity_date != second.activity_date

    def test_score_shows_against_the_run_not_the_recording_date(self, service):
        service.sync(as_of=TODAY)
        service.record_run_feeling(5, "flying")
        entries = service.get_recent_feelings_payload(days=21)["entries"]
        assert all(e["activity_date"] is not None for e in entries)
        linked = {r["date"]: r["score"] for r in service.get_run_feelings_payload(days=90)["runs"] if r["score"]}
        assert 5 in linked.values()

    def test_explicit_date_still_wins(self, service):
        service.sync(as_of=TODAY)
        fb = service.record_run_feeling(3, "meh", date(2026, 7, 18))
        assert fb.activity_date == date(2026, 7, 18)

    def test_no_unrated_runs_leaves_the_entry_unlinked(self, service):
        """Nothing to attach to is a real state — it must not invent a date."""
        fb = service.record_run_feeling(4, "great", None)
        assert fb.activity_date is None


class TestFeelingsAffectRecommendation:
    def test_rough_feelings_lower_the_ceiling(self, config, storage, service):
        baseline = agent.run_invocation(config, storage, service.adapter, as_of=TODAY)
        baseline_distance = baseline.recommendation.target_distance_mi

        for _ in range(3):
            storage.save_run_feedback(RunFeedback(datetime.utcnow(), 1, "legs are dead"))

        after = agent.run_invocation(config, storage, service.adapter, as_of=TODAY)
        assert after.recommendation.target_distance_mi <= baseline_distance
        assert any(g.name == "recent_run_feeling" for g in after.recommendation.guardrail_results)

    def test_good_feelings_add_no_constraint(self, config, storage, service):
        for _ in range(3):
            storage.save_run_feedback(RunFeedback(datetime.utcnow(), 5, "flying"))
        result = agent.run_invocation(config, storage, service.adapter, as_of=TODAY)
        assert not any(g.name == "recent_run_feeling" for g in result.recommendation.guardrail_results)

    def test_no_feedback_adds_no_constraint(self, config, storage, service):
        result = agent.run_invocation(config, storage, service.adapter, as_of=TODAY)
        assert not any(g.name == "recent_run_feeling" for g in result.recommendation.guardrail_results)

    def test_service_records_feeling_and_clamps_score(self, service):
        fb = service.record_run_feeling(99, "typo")
        assert fb.score == 5
        assert service.record_run_feeling(-4).score == 1

    def test_feelings_payload_reports_mean(self, service):
        service.record_run_feeling(2)
        service.record_run_feeling(4)
        payload = service.get_recent_feelings_payload()
        assert payload["count"] == 2 and payload["mean_score"] == pytest.approx(3.0)

    def test_unrated_runs_shrink_once_rated(self, service):
        service.get_today(TODAY)  # sync so activities exist
        before = service.get_unrated_recent_runs(days=30)
        assert before, "fixture should have recent runs to rate"
        rated_date = before[0]["date"]
        service.record_run_feeling(4, "good", date.fromisoformat(rated_date))
        after = service.get_unrated_recent_runs(days=30)
        # Feedback is keyed by date, so rating a day covers every run on it
        # (e.g. a double day) — the rated date must simply be gone.
        assert len(after) < len(before)
        assert all(r["date"] != rated_date for r in after)


class TestAdaptiveRunDays:
    def test_derives_frequency_from_actual_runs(self):
        # 3 runs/week for 4 weeks
        acts = []
        for wk in range(4):
            base = TODAY - timedelta(days=7 * wk + 1)
            for offset in (0, 2, 4):
                acts.append(run_on(base - timedelta(days=offset)))
        classified = prepare_classified(acts)
        assert derive_run_days_per_week(classified, TODAY) == 3

    def test_detects_a_higher_frequency(self):
        acts = [run_on(TODAY - timedelta(days=d)) for d in range(1, 25)]  # daily
        classified = prepare_classified(acts)
        assert derive_run_days_per_week(classified, TODAY) >= 6

    def test_none_when_no_recent_running(self):
        assert derive_run_days_per_week([], TODAY) is None

    def test_anchors_carry_derived_frequency(self, service):
        service.sync(TODAY)  # get_current_anchors reads storage, it doesn't pull
        anchors = service.get_current_anchors(TODAY)
        assert anchors.runs_per_week is not None and anchors.runs_per_week >= 1

    def test_plan_rationale_notes_frequency_mismatch(self, config, storage, service):
        config.athlete.available_run_days_per_week = 6  # config says 6, data says ~3
        result = agent.run_invocation(config, storage, service.adapter, as_of=TODAY)
        assert "derived from actuals" in result.plan.rationale


class TestMilestonePush:
    def test_push_half_marathon_persists_and_delays(self, service):
        service.get_today(TODAY)
        before = next(w for w in service.get_plan().weeks if w.block == "half_marathon")
        result = service.push_milestone("half_marathon", 3)
        assert result["delay_weeks_total"] == 3
        after = next(w for w in service.get_plan().weeks if w.block == "half_marathon")
        assert after.start_date > before.start_date

    def test_push_accumulates_across_calls(self, service):
        service.get_today(TODAY)
        service.push_milestone("half_marathon", 2)
        result = service.push_milestone("half_marathon", 2)
        assert result["delay_weeks_total"] == 4

    def test_pushed_weeks_are_filled_not_left_flat(self, service):
        """The delay must be absorbed by real build/back-off weeks so the
        base is maintained, not by a gap in the plan."""
        service.get_today(TODAY)
        service.push_milestone("half_marathon", 4)
        plan = service.get_plan()
        hm_idx = next(i for i, w in enumerate(plan.weeks) if w.block == "half_marathon")
        pre = plan.weeks[:hm_idx]
        assert all(w.target_volume_mi > 0 for w in pre if w.block == "base_building")
        # weeks stay contiguous (no holes) — the plan filled the time
        assert [w.week_number for w in plan.weeks] == list(range(1, len(plan.weeks) + 1))

    def test_marathon_cannot_be_pushed_this_way(self, service):
        service.get_today(TODAY)
        assert "error" in service.push_milestone("marathon", 2)

    def test_push_survives_plan_regeneration(self, service):
        service.get_today(TODAY)
        service.push_milestone("ultra_50k", 3)
        service.refresh_today(TODAY)   # regenerates from anchors
        assert service.storage.get_preference(agent.PREF_ULTRA_DELAY_WEEKS) == "3"


class TestHealthFlagUndo:
    def test_flag_then_clear_restores_distance(self, service):
        service.get_today(TODAY)
        before = service.get_today_payload(TODAY)["recommendation"]["target_distance_mi"]

        service.set_health_flag("fatigue")
        assert service.active_health_flag == "fatigue"
        flagged = service.get_today_payload(TODAY)["recommendation"]["target_distance_mi"]
        assert flagged <= before

        service.clear_health_flag()
        assert service.active_health_flag is None
        restored = service.get_today_payload(TODAY)["recommendation"]["target_distance_mi"]
        assert restored == pytest.approx(before)

    def test_clearing_keeps_the_flag_in_history(self, service):
        service.set_health_flag("hip", "twinge")
        service.clear_health_flag()
        assert any(f.flag == "hip" for f in service.storage.get_recent_health_flags())

    def test_clear_without_active_flag_is_safe(self, service):
        service.clear_health_flag()
        assert service.active_health_flag is None

    def test_active_flag_surfaces_in_payload(self, service):
        service.set_health_flag("back")
        assert service.get_today_payload(TODAY)["active_health_flag"] == "back"
