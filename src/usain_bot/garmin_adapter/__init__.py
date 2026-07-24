"""All Garmin I/O is isolated behind this package. Nothing outside
garmin_adapter/ should import the `garminconnect` library directly, and
nothing outside it should need network access to be testable — use
`MockGarminAdapter` for that."""

from .base import GarminAdapter, GarminUnavailableError
from .live import GarminConnectAdapter
from .mock import MockGarminAdapter

__all__ = ["GarminAdapter", "GarminUnavailableError", "GarminConnectAdapter", "MockGarminAdapter"]
