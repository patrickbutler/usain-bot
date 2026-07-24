from datetime import date, datetime

import pytest

from usain_bot import guardrails as gr
from usain_bot.config import Config, load_config
from usain_bot.models import Anchors, GapInfo, GapSeverity
from usain_bot.planner import apply_override, diff_plan_versions, generate_macro_plan


@pytest.fixture
def config() -> Config:
    return load_config("config.yaml")


@pytest.fixture
def cold_start_anchors() -> Anchors:
    return Anchors(
        as_of=date(2026, 7, 24), acute_load_mi=0.0, chronic_load_mi=0.0,
        long_run_anchor_mi=0.0, adherence_rate=None, acwr=None,
        gap=GapInfo(gap_days=10_000, severity=GapSeverity.SEVERE, last_run_date=None),
    )


@pytest.fixture
def warm_anchors() -> Anchors:
    return Anchors(
        as_of=date(2026, 7, 24), acute_load_mi=22.0, chronic_load_mi=24.0,
        long_run_anchor_mi=8.0, adherence_rate=0.9, acwr=22.0 / 24.0,
        gap=GapInfo(gap_days=1, severity=GapSeverity.SHORT, last_run_date=date(2026, 7, 23)),
    )


class TestGenerateMacroPlan:
    def test_uses_config_baseline_at_cold_start(self, config, cold_start_anchors):
        plan = generate_macro_plan(config, cold_start_anchors, date(2026, 7, 24), 1, "first_run", "test")
        # Week 1 is a build week, so it's the baseline plus one increment —
        # not the flat baseline itself — but it must be *derived from* the
        # configured baseline, not an arbitrarily larger/smaller number.
        expected = gr.next_long_run_distance(config.athlete.baseline_long_run_mi)
        assert plan.weeks[0].long_run_mi == pytest.approx(expected)

    def test_uses_actual_anchor_over_config_baseline(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        # anchor is 8.0, config baseline is also 8.0 in this repo's config.yaml,
        # so bump the anchor to prove actuals win.
        warm_anchors2 = Anchors(**{**warm_anchors.__dict__, "long_run_anchor_mi": 11.0})
        plan2 = generate_macro_plan(config, warm_anchors2, date(2026, 7, 24), 1, "test", "test")
        assert plan2.weeks[0].long_run_mi > plan.weeks[0].long_run_mi

    def test_full_arc_includes_all_blocks(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        blocks = {w.block for w in plan.weeks}
        assert "base_building" in blocks or "half_marathon_capability" in blocks
        assert "taper" in blocks
        assert "marathon" in blocks
        assert "recovery" in blocks
        assert any("ultra_specific_block" in b for b in blocks)
        assert any("ultra_50k" in b for b in blocks)

    def test_50k_and_ultra_block_are_unscheduled_placeholders(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        ultra_weeks = [w for w in plan.weeks if "TBD" in w.block]
        assert len(ultra_weeks) == 2
        for w in ultra_weeks:
            assert w.target_volume_mi == 0.0
            assert w.long_run_mi == 0.0
            assert "not" in w.notes.lower() or "NOT" in w.notes

    def test_marathon_week_hits_race_distance(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        marathon_week = next(w for w in plan.weeks if w.block == "marathon")
        assert marathon_week.long_run_mi == pytest.approx(26.2)

    def test_taper_reduces_volume_from_peak(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        taper_weeks = [w for w in plan.weeks if w.block == "taper"]
        build_weeks = [w for w in plan.weeks if w.block in ("base_building", "half_marathon_capability", "marathon_block")]
        assert taper_weeks
        peak_volume = max(w.target_volume_mi for w in build_weeks)
        for tw in taper_weeks:
            assert tw.target_volume_mi < peak_volume

    def test_backoff_weeks_recur_on_default_cadence(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        build_weeks = [w for w in plan.weeks if w.block in ("base_building", "half_marathon_capability", "marathon_block")]
        backoff_indices = [i for i, w in enumerate(build_weeks) if w.is_backoff]
        assert len(backoff_indices) >= 2
        # roughly every 3rd-4th week (cadence 3 + the backoff week itself = 4-week cycle)
        gaps = [b - a for a, b in zip(backoff_indices, backoff_indices[1:])]
        assert all(3 <= g <= 5 for g in gaps)

    def test_backoff_week_does_not_increase_long_run(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        build_weeks = [w for w in plan.weeks if w.block in ("base_building", "half_marathon_capability", "marathon_block")]
        for prior, cur in zip(build_weeks, build_weeks[1:]):
            if cur.is_backoff:
                assert cur.long_run_mi <= prior.long_run_mi

    def test_long_run_never_exceeds_pct_of_weekly_volume(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        for w in plan.weeks:
            if w.target_volume_mi > 0 and w.block not in ("marathon",):
                pct = w.long_run_mi / w.target_volume_mi
                assert pct <= gr.LONG_RUN_PCT_OF_WEEKLY_VOLUME_ULTRA_BLOCK + 1e-6

    def test_gap_hold_weeks_suppresses_growth(self, config, warm_anchors):
        plan_normal = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        plan_held = generate_macro_plan(
            config, warm_anchors, date(2026, 7, 24), 1, "test", "test",
            gap_hold_weeks=3, gap_easy_only=True,
        )
        assert plan_held.weeks[0].long_run_mi == 0.0
        assert plan_held.weeks[0].block == "return_to_running"
        assert plan_normal.weeks[0].long_run_mi > 0

    def test_missing_marathon_date_raises(self, warm_anchors):
        cfg = load_config("config.yaml")
        cfg.goal("marathon").date = None
        with pytest.raises(ValueError):
            generate_macro_plan(cfg, warm_anchors, date(2026, 7, 24), 1, "test", "test")


class TestDiffPlanVersions:
    def test_no_prior_plan(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        diff = diff_plan_versions(None, plan)
        assert "Initial plan" in diff

    def test_detects_changed_week(self, config, warm_anchors):
        plan1 = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        warm_anchors2 = Anchors(**{**warm_anchors.__dict__, "long_run_anchor_mi": 11.0})
        plan2 = generate_macro_plan(config, warm_anchors2, date(2026, 7, 24), 2, "test", "test")
        diff = diff_plan_versions(plan1, plan2)
        assert "week 1" in diff


class TestApplyOverride:
    def test_ease_next_week(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        result = apply_override(plan, "Can you make next week easier", config, warm_anchors, date(2026, 7, 24))
        assert result.applied
        old_week2 = plan.weeks[1]
        new_week2 = result.plan.weeks[1]
        assert new_week2.target_volume_mi < old_week2.target_volume_mi
        assert new_week2.is_backoff is True

    def test_shift_marathon_back(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        result = apply_override(plan, "shift the marathon block back two weeks", config, warm_anchors, date(2026, 7, 24))
        assert result.applied
        marathon_week = next(w for w in result.plan.weeks if w.block == "marathon")
        assert "2027-02-21" in marathon_week.notes

    def test_move_long_run_day(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        result = apply_override(plan, "I want to move my long run to Sunday", config, warm_anchors, date(2026, 7, 24))
        assert result.applied
        assert any("sunday" in w.notes.lower() for w in result.plan.weeks)
        assert result.warnings

    def test_unrecognized_override_not_applied(self, config, warm_anchors):
        plan = generate_macro_plan(config, warm_anchors, date(2026, 7, 24), 1, "test", "test")
        result = apply_override(plan, "make me run a marathon tomorrow", config, warm_anchors, date(2026, 7, 24))
        assert not result.applied
        assert result.plan is None
        assert result.warnings
