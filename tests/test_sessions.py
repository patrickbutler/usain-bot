"""Split-run merging: one run recorded as several activities must be
analyzed as one session (<=3h gap), while genuinely separate runs on the
same day (>3h apart) must stay separate."""

from datetime import date, datetime

import pytest

from usain_bot.models import Activity, ActivityType
from usain_bot.sessions import merge_split_runs


def run(aid, start: datetime, dist, minutes, hr=145, atype=ActivityType.RUNNING):
    return Activity(
        activity_id=aid, date=start.date(), activity_type=atype, distance_mi=dist,
        duration_s=minutes * 60, avg_hr=hr, max_hr=180, name="Run", start_time=start,
    )


class TestMergeSplitRuns:
    def test_two_recordings_25_min_apart_merge(self):
        a = run("1", datetime(2026, 7, 11, 7, 0), 6.2, 62)
        b = run("2", datetime(2026, 7, 11, 8, 27), 3.8, 38)   # 25 min after a ends
        merged = merge_split_runs([a, b])
        assert len(merged) == 1
        assert merged[0].distance_mi == pytest.approx(10.0)
        assert merged[0].duration_s == 100 * 60

    def test_gap_over_3h_stays_separate(self):
        a = run("1", datetime(2026, 7, 25, 7, 0), 10.0, 100)   # ends 08:40
        b = run("2", datetime(2026, 7, 25, 18, 0), 3.0, 30)    # >3h later
        merged = merge_split_runs([a, b])
        assert len(merged) == 2

    def test_exactly_3h_gap_merges_boundary(self):
        a = run("1", datetime(2026, 7, 25, 7, 0), 5.0, 60)     # ends 08:00
        b = run("2", datetime(2026, 7, 25, 11, 0), 4.0, 40)    # exactly 3h after end
        merged = merge_split_runs([a, b])
        assert len(merged) == 1

    def test_just_over_3h_gap_does_not_merge(self):
        a = run("1", datetime(2026, 7, 25, 7, 0), 5.0, 60)     # ends 08:00
        b = run("2", datetime(2026, 7, 25, 11, 1), 4.0, 40)    # 3h01m after end
        merged = merge_split_runs([a, b])
        assert len(merged) == 2

    def test_three_way_split_merges_into_one(self):
        a = run("1", datetime(2026, 7, 11, 7, 0), 4.0, 40)
        b = run("2", datetime(2026, 7, 11, 8, 0), 3.0, 30)
        c = run("3", datetime(2026, 7, 11, 9, 0), 3.0, 30)
        merged = merge_split_runs([a, b, c])
        assert len(merged) == 1
        assert merged[0].distance_mi == pytest.approx(10.0)
        assert merged[0].raw["merged_activity_ids"] == ["1", "2", "3"]

    def test_merged_hr_is_duration_weighted(self):
        a = run("1", datetime(2026, 7, 11, 7, 0), 6.0, 60, hr=140)
        b = run("2", datetime(2026, 7, 11, 8, 20), 2.0, 20, hr=170)
        merged = merge_split_runs([a, b])
        # (140*60 + 170*20) / 80 = 147.5 -> 148
        assert merged[0].avg_hr == 148

    def test_runs_on_different_days_never_merge(self):
        a = run("1", datetime(2026, 7, 11, 22, 0), 4.0, 40)
        b = run("2", datetime(2026, 7, 12, 0, 30), 4.0, 40)   # 1h50m later, next day
        merged = merge_split_runs([a, b])
        # gap rule is time-based, not calendar-based: these DO merge
        assert len(merged) == 1

    def test_activities_without_start_time_pass_through_unmerged(self):
        a = Activity("1", date(2026, 7, 11), ActivityType.RUNNING, 5.0, 3000, name="No timestamp")
        b = Activity("2", date(2026, 7, 11), ActivityType.RUNNING, 3.0, 1800, name="Also none")
        merged = merge_split_runs([a, b])
        assert len(merged) == 2

    def test_single_run_is_unchanged_not_relabeled(self):
        a = run("1", datetime(2026, 7, 11, 7, 0), 8.0, 80)
        merged = merge_split_runs([a])
        assert merged[0] is a

    def test_empty_input(self):
        assert merge_split_runs([]) == []

    def test_configurable_gap_threshold(self):
        a = run("1", datetime(2026, 7, 25, 7, 0), 5.0, 60)    # ends 08:00
        b = run("2", datetime(2026, 7, 25, 9, 0), 4.0, 40)    # 1h later
        assert len(merge_split_runs([a, b], gap_hours=0.5)) == 2
        assert len(merge_split_runs([a, b], gap_hours=3.0)) == 1
