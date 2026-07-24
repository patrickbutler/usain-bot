"""Typed access to config.yaml plus environment/secrets.

Nothing in here talks to Garmin or a database — it only resolves
*where* things live and *what* the athlete/goal settings are, so the
rest of the system can stay decoupled from how configuration is
sourced (yaml today, maybe a UI-backed store later).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a light convenience dep
    load_dotenv = None

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class AthleteConfig:
    baseline_long_run_mi: float
    injury_history: list[str]
    available_run_days_per_week: int
    cross_training: list[str]


@dataclass
class GoalConfig:
    name: str
    distance_mi: float
    type: str
    date: Optional[str] = None
    target_weeks_from_baseline: Optional[int] = None


@dataclass
class SequencingConfig:
    post_marathon_recovery_weeks: int
    ultra_specific_block_weeks: int


@dataclass
class GuardrailConfig:
    backoff_cadence: int = 3
    long_run_pct_of_weekly_volume: float = 0.35
    long_run_pct_of_weekly_volume_ultra_block: float = 0.40
    weekly_volume_growth_factor: float = 1.10
    long_run_increment_pct: float = 0.10
    long_run_increment_abs_cap_mi: float = 1.0
    acwr_green_max: float = 1.3
    acwr_yellow_max: float = 1.5
    acwr_detrain_min: float = 0.8


@dataclass
class StorageConfig:
    backend: str = "local"
    data_dir: str = "./data"
    db_filename: str = "usain_bot.db"
    references_dir: str = "references"


@dataclass
class Config:
    athlete: AthleteConfig
    goals: list[GoalConfig]
    sequencing: SequencingConfig
    guardrails: GuardrailConfig
    storage: StorageConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def goal(self, name: str) -> Optional[GoalConfig]:
        return next((g for g in self.goals if g.name == name), None)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, env_path: Optional[str | Path] = None) -> Config:
    if load_dotenv is not None:
        load_dotenv(env_path) if env_path else load_dotenv()

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    athlete = AthleteConfig(**raw["athlete"])
    goals = [GoalConfig(**g) for g in raw.get("goals", [])]
    sequencing = SequencingConfig(**raw["sequencing"])
    guardrails = GuardrailConfig(**raw.get("guardrails", {}))

    storage_raw = raw.get("storage", {})
    local_raw = storage_raw.get("local", {})
    storage = StorageConfig(
        backend=os.environ.get("USAIN_BOT_STORAGE_BACKEND", storage_raw.get("backend", "local")),
        data_dir=os.environ.get("USAIN_BOT_DATA_DIR", local_raw.get("data_dir", "./data")),
        db_filename=local_raw.get("db_filename", "usain_bot.db"),
        references_dir=local_raw.get("references_dir", "references"),
    )

    return Config(
        athlete=athlete,
        goals=goals,
        sequencing=sequencing,
        guardrails=guardrails,
        storage=storage,
        raw=raw,
    )


@dataclass
class GarminCredentials:
    email: str
    password: str
    token_store: str

    @classmethod
    def from_env(cls) -> "GarminCredentials":
        email = os.environ.get("GARMIN_EMAIL")
        password = os.environ.get("GARMIN_PASSWORD")
        token_store = os.environ.get("GARMINTOKENS", "~/.usain-bot/garmin_tokens")
        if not email or not password:
            raise RuntimeError(
                "GARMIN_EMAIL and GARMIN_PASSWORD must be set in the environment "
                "(see .env.example). Never hardcode Garmin credentials."
            )
        return cls(email=email, password=password, token_store=os.path.expanduser(token_store))
