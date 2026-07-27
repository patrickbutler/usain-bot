"""Deterministic plan-validation tests. The validator is the independent
referee for the athlete's milestone rules: 12 mi before the half, 20 mi
before the marathon, 3x >=30 mi weeks before the 50K, correct taper
lengths, and the fixed marathon date."""

from datetime import date, timedelta

import pytest

from usain_bot.config import load_config
from usain_bot.models import Anchors, GapInfo, GapSeverity, PlanWeek
from usain_bot.planner import generate_macro_plan
from usain_bot.validation import validate_plan

MARATHON_DATE = date(2027, 2, 7)


@pytest.fixture
def config():
    return load_config("config.yaml")


@pytest.fixture
def real_anchors() -> Anchors:
    """An athlete already running ~20 mi/wk with 8 mi long runs — the
    case that used to produce a beginner plan."""
    return Anchors(
        as_of=date(2026, 7, 27), acute_load_mi=20.0, chronic_load_mi=20.0, long_run_anchor_mi=8.0,
        adherence_rate=1.0, acwr=1.0,
        gap=GapInfo(gap_days=1, severity=GapSeverity.SHORT, last_run_date=date(2026, 7, 26)),
        runs_per_week=4,
    )


def week(n, start, block, vol, lr, backoff=False):
    return PlanWeek(n, start, block, vol, lr, 0, backoff, "")


def errors(issues):
    return [i for i in issues if i.severity == "error"]


class TestGeneratedPlanPassesTheRules:
    def test_real_athlete_plan_has_no_errors(self, config, real_anchors):
        plan = generate_macro_plan(config, real_anchors, date(2026, 7, 27), 1, "test", "test")
        assert errors(validate_plan(plan.weeks, MARATHON_DATE)) == []

    def test_plan_actually_reaches_20mi_before_marathon_taper(self, config, real_anchors):
        plan = generate_macro_plan(config, real_anchors, date(2026, 7, 27), 1, "test", "test")
        m_idx = next(i for i, w in enumerate(plan.weeks) if w.block == "marathon")
        taper_len = sum(1 for w in plan.weeks[:m_idx] if w.block == "marathon_taper")
        pre_taper = plan.weeks[: m_idx - taper_len]
        assert max(w.long_run_mi for w in pre_taper) >= 20.0

    def test_plan_actually_reaches_12mi_before_half_taper(self, config, real_anchors):
        plan = generate_macro_plan(config, real_anchors, date(2026, 7, 27), 1, "test", "test")
        hm_idx = next(i for i, w in enumerate(plan.weeks) if w.block == "half_marathon")
        taper_len = sum(1 for w in plan.weeks[:hm_idx] if w.block == "hm_taper")
        assert max(w.long_run_mi for w in plan.weeks[: hm_idx - taper_len]) >= 12.0

    def test_plan_has_3_consecutive_30mi_weeks_before_ultra_taper(self, config, real_anchors):
        plan = generate_macro_plan(config, real_anchors, date(2026, 7, 27), 1, "test", "test")
        u_idx = next(i for i, w in enumerate(plan.weeks) if w.block == "ultra_50k")
        taper_len = sum(1 for w in plan.weeks[:u_idx] if w.block == "ultra_taper")
        run = best = 0
        for w in plan.weeks[: u_idx - taper_len]:
            run = run + 1 if w.target_volume_mi >= 30.0 else 0
            best = max(best, run)
        assert best >= 3

    def test_cold_start_plan_also_validates(self, config):
        cold = Anchors(
            as_of=date(2026, 7, 27), acute_load_mi=0.0, chronic_load_mi=0.0, long_run_anchor_mi=0.0,
            adherence_rate=None, acwr=None,
            gap=GapInfo(gap_days=10_000, severity=GapSeverity.SEVERE, last_run_date=None),
        )
        plan = generate_macro_plan(config, cold, date(2026, 7, 27), 1, "test", "test")
        assert errors(validate_plan(plan.weeks, MARATHON_DATE)) == []


class TestValidatorCatchesViolations:
    def test_empty_plan_is_an_error(self):
        assert errors(validate_plan([], MARATHON_DATE))

    def test_missing_20mi_before_marathon_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 20, 10),
            week(2, start + timedelta(weeks=1), "marathon_taper", 15, 8),
            week(3, start + timedelta(weeks=2), "marathon_taper", 12, 6),
            week(4, MARATHON_DATE - timedelta(days=MARATHON_DATE.weekday()), "marathon", 10, 26.2),
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "marathon_prereq" in codes

    def test_wrong_marathon_date_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 30, 20),
            week(2, start + timedelta(weeks=1), "marathon_taper", 20, 12),
            week(3, start + timedelta(weeks=2), "marathon_taper", 15, 8),
            week(4, start + timedelta(weeks=3), "marathon", 12, 26.2),  # not the fixed date
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "marathon_date" in codes

    def test_too_short_marathon_taper_is_caught(self):
        race_week = MARATHON_DATE - timedelta(days=MARATHON_DATE.weekday())
        weeks = [
            week(1, race_week - timedelta(weeks=2), "base_building", 30, 20),
            week(2, race_week - timedelta(weeks=1), "marathon_taper", 20, 12),  # only 1 week
            week(3, race_week, "marathon", 12, 26.2),
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "marathon_taper_length" in codes

    def test_missing_12mi_before_half_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 20, 9),
            week(2, start + timedelta(weeks=1), "hm_taper", 14, 6),
            week(3, start + timedelta(weeks=2), "half_marathon", 16, 13.1),
            week(4, start + timedelta(weeks=3), "base_building", 30, 20),
            week(5, start + timedelta(weeks=4), "marathon_taper", 20, 12),
            week(6, start + timedelta(weeks=5), "marathon_taper", 15, 8),
            week(7, MARATHON_DATE - timedelta(days=MARATHON_DATE.weekday()), "marathon", 12, 26.2),
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "half_marathon_prereq" in codes

    def test_insufficient_30mi_weeks_before_ultra_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "ultra_build", 31, 16),
            week(2, start + timedelta(weeks=1), "ultra_build", 25, 16),   # breaks the streak
            week(3, start + timedelta(weeks=2), "ultra_build", 31, 16),
            week(4, start + timedelta(weeks=3), "ultra_taper", 20, 10),
            week(5, start + timedelta(weeks=4), "ultra_taper", 15, 8),
            week(6, start + timedelta(weeks=5), "ultra_50k", 12, 31.1),
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "ultra_prereq" in codes

    def test_long_run_jump_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 25, 10),
            week(2, start + timedelta(weeks=1), "base_building", 26, 16),  # +6 mi in one week
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "long_run_jump" in codes

    def test_volume_jump_is_caught(self):
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 20, 10),
            week(2, start + timedelta(weeks=1), "base_building", 30, 10.5),  # +50%
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "volume_jump" in codes

    def test_backoff_transitions_are_not_flagged_as_jumps(self):
        # Dropping into and climbing out of a back-off week is expected,
        # not a guardrail violation.
        start = date(2026, 7, 27)
        weeks = [
            week(1, start, "base_building", 25, 11),
            week(2, start + timedelta(weeks=1), "base_building", 18, 7.7, backoff=True),
            week(3, start + timedelta(weeks=2), "base_building", 27, 11.9),
        ]
        codes = {i.code for i in errors(validate_plan(weeks, MARATHON_DATE))}
        assert "volume_jump" not in codes and "long_run_jump" not in codes
