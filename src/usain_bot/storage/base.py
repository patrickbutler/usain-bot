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

from ..models import (
    Activity,
    ConversationEntry,
    HealthFlag,
    PlanVersion,
    ReferenceChunk,
    ReferenceDoc,
    RunFeedback,
)


class StorageBackend(ABC):
    # --- activities ---------------------------------------------------------

    @abstractmethod
    def save_activities(self, activities: list[Activity]) -> tuple[int, int]:
        """Upsert activities keyed on activity_id. New rows are inserted;
        existing rows whose material fields (distance, duration, name, HR,
        start time) differ are updated in place — this is how edited Garmin
        activities inside the sync overlap window get corrected. Returns
        (new_count, updated_count)."""

    @abstractmethod
    def get_activities(self, since: Optional[datetime] = None) -> list[Activity]:
        """All stored activities, optionally filtered to those on/after `since`."""

    @abstractmethod
    def find_duplicate_activity_groups(self) -> list[list[Activity]]:
        """Groups of activities that look like the same physical run stored
        under different Garmin IDs (near-identical start time and distance).
        Distinct from split-run *merging* (sessions.py), which is a
        read-time analysis concern — these are genuine storage duplicates."""

    @abstractmethod
    def delete_activities(self, activity_ids: list[str]) -> int:
        """Remove specific rows (used by dedupe resolution). Returns count deleted."""

    @abstractmethod
    def get_last_sync_time(self) -> Optional[datetime]:
        ...

    @abstractmethod
    def set_last_sync_time(self, ts: datetime) -> None:
        ...

    # --- preferences (durable user choices that survive plan regeneration) ---

    @abstractmethod
    def get_preference(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    def set_preference(self, key: str, value: str) -> None:
        ...

    # --- run feedback (how runs felt — the coach's subjective-state memory) ---

    @abstractmethod
    def save_run_feedback(self, feedback: RunFeedback) -> None:
        ...

    @abstractmethod
    def get_recent_run_feedback(self, days: int = 14, limit: int = 20) -> list[RunFeedback]:
        """Most recent first."""

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
