"""Abstract persistence interface.

Four logically distinct stores, per §6 of the spec:

1. activities     — Garmin history. Append-only, immutable, deduplicated
                     by Garmin activity ID. Time-series shaped.
2. plan_versions  — full lineage of the training plan. Every change
                     writes a new version; never mutated in place.
3. conversations  — dialogue history, decisions, and overrides.
4. references     — coaching reference articles/notes, chunked and
                     indexed for retrieval, kept separate from
                     conversational memory.

Concrete backends (LocalBackend, GCPBackend) implement this. Agent code
must only ever hold a `StorageBackend` reference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..models import Activity, ConversationEntry, HealthFlag, PlanVersion, ReferenceChunk, ReferenceDoc


class StorageBackend(ABC):
    # --- activities ---------------------------------------------------------

    @abstractmethod
    def save_activities(self, activities: list[Activity]) -> int:
        """Insert new activities, deduped by activity_id. Returns count of new rows."""

    @abstractmethod
    def get_activities(self, since: Optional[datetime] = None) -> list[Activity]:
        """All stored activities, optionally filtered to those on/after `since`."""

    @abstractmethod
    def get_last_sync_time(self) -> Optional[datetime]:
        ...

    @abstractmethod
    def set_last_sync_time(self, ts: datetime) -> None:
        ...

    # --- plan_versions --------------------------------------------------------

    @abstractmethod
    def save_plan_version(self, plan_version: PlanVersion) -> None:
        """Append-only: never mutates a prior version."""

    @abstractmethod
    def get_latest_plan_version(self) -> Optional[PlanVersion]:
        ...

    @abstractmethod
    def get_plan_history(self) -> list[PlanVersion]:
        ...

    # --- conversations ----------------------------------------------------------

    @abstractmethod
    def save_conversation_entry(self, entry: ConversationEntry) -> None:
        ...

    @abstractmethod
    def get_conversation_history(self, limit: Optional[int] = None) -> list[ConversationEntry]:
        ...

    # --- references ---------------------------------------------------------------

    @abstractmethod
    def save_reference(self, doc: ReferenceDoc) -> None:
        ...

    @abstractmethod
    def search_references(self, query: str, top_k: int = 3) -> list[ReferenceChunk]:
        ...

    @abstractmethod
    def list_references(self) -> list[ReferenceDoc]:
        ...

    # --- health flags (§5.8) ---------------------------------------------------------

    @abstractmethod
    def save_health_flag(self, flag: HealthFlag) -> None:
        ...

    @abstractmethod
    def get_recent_health_flags(self, days: int = 30) -> list[HealthFlag]:
        ...
