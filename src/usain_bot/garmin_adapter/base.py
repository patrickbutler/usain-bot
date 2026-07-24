"""Adapter interface. `agent.py` depends only on this, never on the
concrete `garminconnect`-backed implementation, so tests and the mock
adapter can stand in for it with zero network access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..models import Activity


class GarminUnavailableError(Exception):
    """Raised when Garmin can't be reached. Callers should degrade
    gracefully — fall back to cached data and say so explicitly, biasing
    conservative — rather than crash."""


class GarminAdapter(ABC):
    @abstractmethod
    def fetch_activities(self, start_date: date, end_date: date) -> list[Activity]:
        """Fetch and normalize activities in [start_date, end_date] (inclusive)."""
