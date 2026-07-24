"""Config-driven backend selection. This is the only place that imports
both concrete backends — everything else imports `StorageBackend`."""

from __future__ import annotations

import os

from ..config import Config
from .base import StorageBackend


def get_storage_backend(config: Config) -> StorageBackend:
    backend = config.storage.backend.lower()

    if backend == "local":
        from .local import LocalBackend

        return LocalBackend(
            data_dir=config.storage.data_dir,
            db_filename=config.storage.db_filename,
            references_dir=config.storage.references_dir,
        )

    if backend == "gcp":
        from .gcp import GCPBackend

        return GCPBackend(
            project=os.environ.get("USAIN_BOT_GCP_PROJECT", ""),
            bq_dataset=os.environ.get("USAIN_BOT_BQ_DATASET", ""),
            gcs_bucket=os.environ.get("USAIN_BOT_GCS_BUCKET", ""),
        )

    raise ValueError(f"Unknown storage backend: {backend!r} (expected 'local' or 'gcp')")
