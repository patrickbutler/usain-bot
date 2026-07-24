"""Swappable persistence layer.

Agent logic must depend only on `StorageBackend` (this package's ABC) and
`get_storage_backend()` — never on `LocalBackend` or `GCPBackend` directly.
That's what lets us start local and lift to GCP later with zero rewrite
of guardrails/planner/agent code.
"""

from .base import StorageBackend
from .factory import get_storage_backend

__all__ = ["StorageBackend", "get_storage_backend"]
