"""Stubbed GCP storage backend: BigQuery for structured data, GCS for documents.

Not implemented — this exists so the interface is proven out and the
config-driven backend switch (`storage.backend: gcp`) has somewhere to
land. Method bodies raise NotImplementedError with the intended mapping
noted in each docstring. When this gets built out, no code outside
this file should need to change: agent/planner/guardrails only ever
depend on `StorageBackend`.
"""

from __future__ import annotations

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
from .base import StorageBackend


class GCPBackend(StorageBackend):
    def __init__(self, project: str, bq_dataset: str, gcs_bucket: str):
        self.project = project
        self.bq_dataset = bq_dataset
        self.gcs_bucket = gcs_bucket

    def save_activities(self, activities: list[Activity]) -> tuple[int, int]:
        """Intended: MERGE into BigQuery table `{dataset}.activities`, keyed on activity_id."""
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_activities(self, since: Optional[datetime] = None) -> list[Activity]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def find_duplicate_activity_groups(self) -> list[list[Activity]]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def delete_activities(self, activity_ids: list[str]) -> int:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_preference(self, key: str) -> Optional[str]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def set_preference(self, key: str, value: str) -> None:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def save_run_feedback(self, feedback: RunFeedback) -> None:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_recent_run_feedback(self, days: int = 14, limit: int = 20) -> list[RunFeedback]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_last_sync_time(self) -> Optional[datetime]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def set_last_sync_time(self, ts: datetime) -> None:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def save_plan_version(self, plan_version: PlanVersion) -> None:
        """Intended: append-only insert into BigQuery table `{dataset}.plan_versions`."""
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_latest_plan_version(self) -> Optional[PlanVersion]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_plan_history(self) -> list[PlanVersion]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def save_conversation_entry(self, entry: ConversationEntry) -> None:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_conversation_history(self, limit: Optional[int] = None) -> list[ConversationEntry]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def save_reference(self, doc: ReferenceDoc) -> None:
        """Intended: raw doc to GCS at gs://{bucket}/references/{doc_id}.md,
        chunk metadata + text to BigQuery `{dataset}.reference_chunks` with a
        vector index (e.g. BigQuery ML.GENERATE_EMBEDDING + VECTOR_SEARCH)
        replacing the local keyword fallback."""
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def search_references(self, query: str, top_k: int = 3) -> list[ReferenceChunk]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def list_references(self) -> list[ReferenceDoc]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def save_health_flag(self, flag: HealthFlag) -> None:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")

    def get_recent_health_flags(self, days: int = 30) -> list[HealthFlag]:
        raise NotImplementedError("GCPBackend is a stub — see storage/gcp.py docstring.")
