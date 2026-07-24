from datetime import date, timedelta

import pytest

from usain_bot.classification import classify_activities, compute_anchors, detect_gap
from usain_bot.models import Activity, ActivityType, GapSeverity, RunClass


def _run(d, dist, hr=140, maxhr=178, name=None, days_offset=0):
    base = date(2026, 6, 1) + timedelta(days=days_offset)
    return Activity(
        activity_id=f"a{d}-{dist}-{days_offset}", date=base, activity_type=ActivityType.RUNNING,
        distance_mi=dist, duration_s=int(dist * 9 * 60), avg_hr=hr, max_hr=maxhr, name=name,
    )


class TestClassifyActivities:
    def test_cross_training_by_type(self):
        act = Activity("c1", date(2026, 6, 1), ActivityType.CYCLING, 15.0, 3000)
        result = classify_activities([act])
        assert result[0].run_class == RunClass.CROSS_TRAINING

    def test_longest_run_in_window_is_long(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity("r1", d0, ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178),
            Activity("r2", d0 + timedelta(days=2), ActivityType.RUNNING, 4.5, 2400, avg_hr=140, max_hr=178),
            Activity("r3", d0 + timedelta(days=5), ActivityType.RUNNING, 8.0, 4500, avg_hr=145, max_hr=178),
        ]
        result = classify_activities(acts)
        by_id = {c.activity.activity_id: c.run_class for c in result}
        assert by_id["r3"] == RunClass.LONG

    def test_single_run_in_window_is_not_trivially_long(self):
        # A lone run in an otherwise empty week shouldn't be "long" just
        # because it's the max of a list of one.
        act = Activity("r1", date(2026, 6, 1), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)
        result = classify_activities([act])
        assert result[0].run_class != RunClass.LONG

    def test_1_4x_median_triggers_long_even_if_not_literal_max_tie(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity("r1", d0, ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178),
            Activity("r2", d0 + timedelta(days=1), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178),
            Activity("r3", d0 + timedelta(days=2), ActivityType.RUNNING, 6.0, 3300, avg_hr=142, max_hr=178),
        ]
        result = classify_activities(acts)
        by_id = {c.activity.activity_id: c.run_class for c in result}
        # median of window at r3's date is 4.0; 6.0 >= 1.4*4.0=5.6 -> long
        assert by_id["r3"] == RunClass.LONG

    def test_quality_detected_by_keyword(self):
        act = Activity("q1", date(2026, 6, 1), ActivityType.RUNNING, 5.0, 2000, avg_hr=150, max_hr=178, name="Tempo run")
        result = classify_activities([act])
        assert result[0].run_class == RunClass.QUALITY

    def test_quality_detected_by_hr_ratio(self):
        act = Activity("q2", date(2026, 6, 1), ActivityType.RUNNING, 5.0, 2000, avg_hr=160, max_hr=180)
        result = classify_activities([act])
        assert result[0].run_class == RunClass.QUALITY

    def test_recovery_short_and_low_hr(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity("e1", d0, ActivityType.RUNNING, 5.0, 2700, avg_hr=140, max_hr=178),
            Activity("e2", d0 + timedelta(days=1), ActivityType.RUNNING, 5.0, 2700, avg_hr=140, max_hr=178),
            Activity("rec", d0 + timedelta(days=2), ActivityType.RUNNING, 2.0, 1300, avg_hr=120, max_hr=178),
        ]
        result = classify_activities(acts)
        by_id = {c.activity.activity_id: c.run_class for c in result}
        assert by_id["rec"] == RunClass.RECOVERY

    def test_exact_tie_does_not_spuriously_crown_a_long_run(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity("e1", d0, ActivityType.RUNNING, 5.0, 2700, avg_hr=140, max_hr=178),
            Activity("e2", d0 + timedelta(days=2), ActivityType.RUNNING, 5.0, 2700, avg_hr=141, max_hr=178),
        ]
        result = classify_activities(acts)
        classes = {c.activity.activity_id: c.run_class for c in result}
        assert classes["e1"] == RunClass.EASY
        assert classes["e2"] == RunClass.EASY

    def test_default_easy_for_clearly_non_max_runs(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity("e1", d0, ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178),
            Activity("e2", d0 + timedelta(days=2), ActivityType.RUNNING, 4.1, 2250, avg_hr=141, max_hr=178),
            Activity("long", d0 + timedelta(days=5), ActivityType.RUNNING, 8.0, 4500, avg_hr=146, max_hr=178),
        ]
        result = classify_activities(acts)
        classes = {c.activity.activity_id: c.run_class for c in result}
        assert classes["e1"] == RunClass.EASY
        assert classes["e2"] == RunClass.EASY
        assert classes["long"] == RunClass.LONG

    def test_empty_input(self):
        assert classify_activities([]) == []


class TestDetectGap:
    def test_no_gap_recent_run(self):
        acts = [Activity("r1", date(2026, 6, 20), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)]
        classified = classify_activities(acts)
        gap = detect_gap(classified, as_of=date(2026, 6, 21))
        assert gap.severity == GapSeverity.SHORT
        assert gap.gap_days == 1

    def test_medium_gap_7_days(self):
        acts = [Activity("r1", date(2026, 6, 1), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)]
        classified = classify_activities(acts)
        gap = detect_gap(classified, as_of=date(2026, 6, 8))
        assert gap.severity == GapSeverity.MEDIUM
        assert gap.gap_days == 7

    def test_long_gap_21_days(self):
        acts = [Activity("r1", date(2026, 6, 1), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)]
        classified = classify_activities(acts)
        gap = detect_gap(classified, as_of=date(2026, 6, 22))
        assert gap.severity == GapSeverity.LONG

    def test_severe_gap_over_6_weeks(self):
        acts = [Activity("r1", date(2026, 1, 1), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)]
        classified = classify_activities(acts)
        gap = detect_gap(classified, as_of=date(2026, 6, 1))
        assert gap.severity == GapSeverity.SEVERE

    def test_no_runs_at_all_is_severe(self):
        gap = detect_gap([], as_of=date(2026, 6, 1))
        assert gap.severity == GapSeverity.SEVERE
        assert gap.last_run_date is None


class TestComputeAnchors:
    def test_zero_chronic_load_cold_start(self):
        anchors = compute_anchors([], as_of=date(2026, 6, 1))
        assert anchors.chronic_load_mi == 0.0
        assert anchors.acwr is None
        assert anchors.long_run_anchor_mi == 0.0

    def test_acute_and_chronic_load(self):
        d0 = date(2026, 6, 1)
        acts = [
            Activity(f"r{i}", d0 - timedelta(days=i * 3), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)
            for i in range(9)  # spans 0..24 days back
        ]
        classified = classify_activities(acts)
        anchors = compute_anchors(classified, as_of=d0)
        # trailing 7 days (d0-6..d0): days back 0,3,6 -> 3 runs * 4.0 = 12.0
        assert anchors.acute_load_mi == pytest.approx(12.0)
        assert anchors.chronic_load_mi > 0

    def test_long_run_anchor_from_completed_runs_only(self):
        d0 = date(2026, 6, 20)
        acts = [
            Activity("long1", d0 - timedelta(days=5), ActivityType.RUNNING, 10.0, 6000, avg_hr=148, max_hr=178),
            Activity("e1", d0 - timedelta(days=4), ActivityType.RUNNING, 3.0, 1700, avg_hr=138, max_hr=178),
        ]
        classified = classify_activities(acts)
        anchors = compute_anchors(classified, as_of=d0)
        assert anchors.long_run_anchor_mi == pytest.approx(10.0)

    def test_adherence_rate(self):
        d0 = date(2026, 6, 20)
        acts = [
            Activity(f"r{i}", d0 - timedelta(days=i * 2), ActivityType.RUNNING, 4.0, 2200, avg_hr=140, max_hr=178)
            for i in range(6)
        ]
        classified = classify_activities(acts)
        anchors = compute_anchors(classified, as_of=d0, planned_runs_trailing_14d=6)
        assert anchors.adherence_rate is not None
        assert 0.0 <= anchors.adherence_rate <= 1.5
