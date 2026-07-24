import math

import pytest

from usain_bot import guardrails as gr


class TestLongRunIncrement:
    def test_pct_binds_at_low_anchor(self):
        # 8 mi anchor: 10% = 0.8 mi, under the 1.0 mi cap -> pct binds
        assert gr.long_run_increment(8.0) == pytest.approx(0.8)

    def test_abs_cap_binds_at_high_anchor(self):
        # 16 mi anchor: 10% = 1.6 mi, over the 1.0 mi cap -> abs cap binds
        assert gr.long_run_increment(16.0) == pytest.approx(1.0)

    def test_crossover_point(self):
        # at exactly 10 mi, 10% == 1.0 mi -- both rules agree
        assert gr.long_run_increment(10.0) == pytest.approx(1.0)

    def test_zero_anchor(self):
        assert gr.long_run_increment(0.0) == 0.0

    def test_negative_anchor_treated_as_zero(self):
        assert gr.long_run_increment(-5.0) == 0.0

    def test_next_long_run_distance(self):
        assert gr.next_long_run_distance(8.0) == pytest.approx(8.8)
        assert gr.next_long_run_distance(16.0) == pytest.approx(17.0)


class TestLongRunPctOfWeeklyVolume:
    def test_default_35_pct(self):
        assert gr.max_long_run_by_weekly_volume_pct(40.0) == pytest.approx(14.0)

    def test_ultra_block_relaxes_to_40_pct(self):
        assert gr.max_long_run_by_weekly_volume_pct(40.0, in_ultra_block=True) == pytest.approx(16.0)

    def test_long_run_exceeding_35_pct_is_flagged_by_caller(self):
        # 20 mi long run against a 40 mi week is 50% -- guardrail caps at 14, so
        # a caller taking min() would correctly reject the raw 20 mi.
        cap = gr.max_long_run_by_weekly_volume_pct(40.0)
        proposed_long_run = 20.0
        assert min(cap, proposed_long_run) == pytest.approx(14.0)


class TestMaxWeeklyVolume:
    def test_growth_factor(self):
        assert gr.max_weekly_volume(20.0) == pytest.approx(22.0)

    def test_zero_chronic_load(self):
        assert gr.max_weekly_volume(0.0) == 0.0


class TestACWR:
    def test_zero_chronic_load_is_undefined(self):
        assert gr.acwr(10.0, 0.0) is None
        assert gr.acwr_zone(gr.acwr(10.0, 0.0)) == gr.ACWRZone.UNDEFINED

    def test_green_zone(self):
        value = gr.acwr(20.0, 20.0)
        assert value == pytest.approx(1.0)
        assert gr.acwr_zone(value) == gr.ACWRZone.GREEN

    def test_green_zone_upper_bound(self):
        assert gr.acwr_zone(1.3) == gr.ACWRZone.GREEN

    def test_yellow_zone(self):
        assert gr.acwr_zone(1.4) == gr.ACWRZone.YELLOW
        assert gr.acwr_zone(1.5) == gr.ACWRZone.YELLOW

    def test_red_zone(self):
        assert gr.acwr_zone(1.51) == gr.ACWRZone.RED
        assert gr.acwr_zone(2.5) == gr.ACWRZone.RED

    def test_detraining_zone(self):
        assert gr.acwr_zone(0.79) == gr.ACWRZone.DETRAINING
        assert gr.acwr_zone(0.0) == gr.ACWRZone.DETRAINING

    def test_boundary_0_8_is_green_not_detraining(self):
        assert gr.acwr_zone(0.8) == gr.ACWRZone.GREEN

    def test_undefined_for_none(self):
        assert gr.acwr_zone(None) == gr.ACWRZone.UNDEFINED


class TestGapAction:
    def test_short_gap_caps_at_last_long_run_no_increase(self):
        action = gr.gap_action(3, last_completed_long_run_mi=9.0)
        assert action.severity == "short"
        assert action.long_run_allowed is True
        assert action.max_long_run_mi == pytest.approx(9.0)
        assert action.backtrack_weeks == 0

    def test_boundary_6_days_is_short(self):
        action = gr.gap_action(6, 9.0)
        assert action.severity == "short"

    def test_boundary_7_days_is_medium(self):
        action = gr.gap_action(7, 9.0)
        assert action.severity == "medium"
        assert action.backtrack_weeks == 1
        assert action.long_run_allowed is False

    def test_medium_gap_14_days(self):
        action = gr.gap_action(14, 9.0)
        assert action.severity == "medium"

    def test_boundary_15_days_is_long(self):
        action = gr.gap_action(15, 9.0)
        assert action.severity == "long"
        assert action.backtrack_weeks == math.ceil(15 / 7)

    def test_long_gap_one_week_backtrack_per_week_missed(self):
        # 21 days missed -> 3 weeks -> backtrack 3 weeks
        action = gr.gap_action(21, 9.0)
        assert action.severity == "long"
        assert action.backtrack_weeks == 3
        assert action.easy_only is True
        assert action.consecutive_easy_required == 3

    def test_boundary_42_days_is_long_not_severe(self):
        action = gr.gap_action(42, 9.0)
        assert action.severity == "long"
        assert action.regenerate_plan is False

    def test_gap_over_6_weeks_regenerates_plan(self):
        action = gr.gap_action(43, 9.0)
        assert action.severity == "severe"
        assert action.regenerate_plan is True
        assert action.easy_only is True

    def test_severe_gap_no_quality(self):
        action = gr.gap_action(90, 9.0)
        assert action.max_quality_sessions == 0


class TestBackoffWeek:
    def test_default_targets(self):
        targets = gr.backoff_week_targets(prior_week_volume_mi=30.0, recent_peak_long_run_mi=12.0)
        assert targets.volume_mi == pytest.approx(21.0)  # 70%
        assert targets.long_run_mi == pytest.approx(8.4)  # 70%
        assert targets.quality_sessions == 0

    def test_pct_is_clamped_into_70_75_range(self):
        low = gr.backoff_week_targets(30.0, 12.0, volume_pct=0.5)
        high = gr.backoff_week_targets(30.0, 12.0, volume_pct=0.95)
        assert low.volume_mi == pytest.approx(30.0 * 0.70)
        assert high.volume_mi == pytest.approx(30.0 * 0.75)

    def test_cadence_default_3_to_1(self):
        assert gr.should_insert_backoff_week(0) is False
        assert gr.should_insert_backoff_week(2) is False
        assert gr.should_insert_backoff_week(3) is True
        assert gr.should_insert_backoff_week(4) is True

    def test_custom_cadence(self):
        assert gr.should_insert_backoff_week(3, cadence=4) is False
        assert gr.should_insert_backoff_week(4, cadence=4) is True


class TestGeneralSafetyRules:
    def test_quality_sessions_capped_at_2_normally(self):
        assert gr.max_quality_sessions_per_week(is_return_from_gap_rebuild=False) == 2

    def test_zero_quality_during_gap_rebuild(self):
        assert gr.max_quality_sessions_per_week(is_return_from_gap_rebuild=True) == 0

    def test_easy_hard_split_80_20_within_tolerance(self):
        assert gr.is_valid_easy_hard_split(easy_mi=32.0, hard_mi=8.0) is True

    def test_easy_hard_split_too_much_hard(self):
        assert gr.is_valid_easy_hard_split(easy_mi=20.0, hard_mi=20.0) is False

    def test_easy_hard_split_zero_total(self):
        assert gr.is_valid_easy_hard_split(0.0, 0.0) is True

    def test_taper_week_volume_within_40_50_pct_reduction(self):
        result = gr.taper_week_volume(40.0)
        assert 20.0 <= result <= 24.0

    def test_taper_reduction_clamped(self):
        low = gr.taper_week_volume(40.0, reduction_pct=0.1)
        high = gr.taper_week_volume(40.0, reduction_pct=0.9)
        assert low == pytest.approx(40.0 * 0.60)
        assert high == pytest.approx(40.0 * 0.50)
