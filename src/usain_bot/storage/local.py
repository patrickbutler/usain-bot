"""SQLite + filesystem storage backend. This is the default, "start local" path.

Structured records (activities, plan versions, conversation turns, health
flags, reference metadata + chunks) live in a single SQLite file.
Reference *documents* additionally get written verbatim to a filesystem
directory, mirroring the local.data_dir/local.references_dir split — this
is the same shape the GCP backend uses (BigQuery for structured,
GCS for documents), so lifting later is a swap, not a rewrite.

Thread-safety: the web server (web/app.py) runs each request's sync
route handler in a threadpool worker thread, so this backend can't
assume single-threaded access the way the CLI could. The connection is
opened with check_same_thread=False and every public method serializes
through a single lock — simplest correct answer at the scale of one
local SQLite file for one user, rather than a connection pool.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ..models import (
    Activity,
    ActivityType,
    ConversationEntry,
    HealthFlag,
    PlanVersion,
    PlanWeek,
    ReferenceChunk,
    ReferenceDoc,
)
from .base import StorageBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    distance_mi REAL NOT NULL,
    duration_s INTEGER NOT NULL,
    avg_pace_min_per_mi REAL,
    avg_hr INTEGER,
    max_hr INTEGER,
    elevation_gain_ft REAL,
    name TEXT,
    raw_json TEXT NOT NULL,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_versions (
    version INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    rationale TEXT NOT NULL,
    diff_from_prior TEXT,
    weeks_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_docs (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    added_at TEXT NOT NULL,
    file_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_chunks (
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS health_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    flag TEXT NOT NULL,
    note TEXT
);
"""

_CHUNK_WORD_TARGET = 180
_WORD_RE = re.compile(r"\w+")


def _chunk_text(text: str, title: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    word_count = 0
    for para in paragraphs:
        words_in_para = len(para.split())
        if word_count + words_in_para > _CHUNK_WORD_TARGET and buf:
            chunks.append("\n\n".join(buf))
            buf, word_count = [], 0
        buf.append(para)
        word_count += words_in_para
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text.strip()]


class LocalBackend(StorageBackend):
    def __init__(self, data_dir: str | Path, db_filename: str = "usain_bot.db", references_dir: str = "references"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.references_path = self.data_dir / references_dir
        self.references_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.db_path = self.data_dir / db_filename
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._fts_available = self._try_enable_fts()

    def _try_enable_fts(self) -> bool:
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS reference_chunks_fts "
                "USING fts5(text, content='reference_chunks', content_rowid='rowid')"
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- activities -----------------------------------------------------------

    def save_activities(self, activities: list[Activity]) -> int:
        now = datetime.utcnow().isoformat()
        new_count = 0
        with self._lock:
            for a in activities:
                cur = self._conn.execute(
                    """INSERT OR IGNORE INTO activities
                       (activity_id, date, activity_type, distance_mi, duration_s,
                        avg_pace_min_per_mi, avg_hr, max_hr, elevation_gain_ft, name,
                        raw_json, synced_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        a.activity_id, a.date.isoformat(), a.activity_type.value, a.distance_mi,
                        a.duration_s, a.avg_pace_min_per_mi, a.avg_hr, a.max_hr,
                        a.elevation_gain_ft, a.name, json.dumps(a.raw), now,
                    ),
                )
                new_count += cur.rowcount
            self._conn.commit()
        return new_count

    def get_activities(self, since: Optional[datetime] = None) -> list[Activity]:
        with self._lock:
            if since is not None:
                rows = self._conn.execute(
                    "SELECT * FROM activities WHERE date >= ? ORDER BY date", (since.date().isoformat(),)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM activities ORDER BY date").fetchall()
        return [self._row_to_activity(r) for r in rows]

    @staticmethod
    def _row_to_activity(row: sqlite3.Row) -> Activity:
        return Activity(
            activity_id=row["activity_id"],
            date=date.fromisoformat(row["date"]),
            activity_type=ActivityType(row["activity_type"]),
            distance_mi=row["distance_mi"],
            duration_s=row["duration_s"],
            avg_pace_min_per_mi=row["avg_pace_min_per_mi"],
            avg_hr=row["avg_hr"],
            max_hr=row["max_hr"],
            elevation_gain_ft=row["elevation_gain_ft"],
            name=row["name"],
            raw=json.loads(row["raw_json"]),
        )

    def get_last_sync_time(self) -> Optional[datetime]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM sync_state WHERE key = 'last_sync_time'").fetchone()
        return datetime.fromisoformat(row["value"]) if row else None

    def set_last_sync_time(self, ts: datetime) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sync_state (key, value) VALUES ('last_sync_time', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (ts.isoformat(),),
            )
            self._conn.commit()

    # --- plan_versions --------------------------------------------------------

    def save_plan_version(self, plan_version: PlanVersion) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO plan_versions (version, created_at, trigger_reason, rationale,
                   diff_from_prior, weeks_json) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    plan_version.version, plan_version.created_at.isoformat(), plan_version.trigger,
                    plan_version.rationale, plan_version.diff_from_prior,
                    json.dumps([w.to_dict() for w in plan_version.weeks]),
                ),
            )
            self._conn.commit()

    def get_latest_plan_version(self) -> Optional[PlanVersion]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM plan_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()
        return self._row_to_plan_version(row) if row else None

    def get_plan_history(self) -> list[PlanVersion]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM plan_versions ORDER BY version").fetchall()
        return [self._row_to_plan_version(r) for r in rows]

    @staticmethod
    def _row_to_plan_version(row: sqlite3.Row) -> PlanVersion:
        weeks_raw = json.loads(row["weeks_json"])
        weeks = [
            PlanWeek(
                week_number=w["week_number"],
                start_date=date.fromisoformat(w["start_date"]),
                block=w["block"],
                target_volume_mi=w["target_volume_mi"],
                long_run_mi=w["long_run_mi"],
                quality_sessions=w["quality_sessions"],
                is_backoff=w["is_backoff"],
                notes=w.get("notes", ""),
            )
            for w in weeks_raw
        ]
        return PlanVersion(
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            trigger=row["trigger_reason"],
            rationale=row["rationale"],
            weeks=weeks,
            diff_from_prior=row["diff_from_prior"],
        )

    # --- conversations ----------------------------------------------------------

    def save_conversation_entry(self, entry: ConversationEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (timestamp, role, text, metadata_json) VALUES (?, ?, ?, ?)",
                (entry.timestamp.isoformat(), entry.role, entry.text, json.dumps(entry.metadata)),
            )
            self._conn.commit()

    def get_conversation_history(self, limit: Optional[int] = None) -> list[ConversationEntry]:
        query = "SELECT * FROM conversations ORDER BY id"
        if limit:
            query += f" DESC LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        entries = [
            ConversationEntry(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                role=r["role"],
                text=r["text"],
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]
        return list(reversed(entries)) if limit else entries

    # --- references ---------------------------------------------------------------

    def save_reference(self, doc: ReferenceDoc) -> None:
        file_path = self.references_path / f"{doc.doc_id}.md"
        file_path.write_text(doc.content, encoding="utf-8")

        with self._lock:
            self._conn.execute(
                """INSERT INTO reference_docs (doc_id, title, source, added_at, file_path)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title, source=excluded.source,
                   added_at=excluded.added_at, file_path=excluded.file_path""",
                (doc.doc_id, doc.title, doc.source, doc.added_at.isoformat(), str(file_path)),
            )
            self._conn.execute("DELETE FROM reference_chunks WHERE doc_id = ?", (doc.doc_id,))
            for i, chunk_text in enumerate(_chunk_text(doc.content, doc.title)):
                self._conn.execute(
                    "INSERT INTO reference_chunks (doc_id, chunk_index, title, text) VALUES (?, ?, ?, ?)",
                    (doc.doc_id, i, doc.title, chunk_text),
                )
            self._conn.commit()
            if self._fts_available:
                self._conn.execute("INSERT INTO reference_chunks_fts(reference_chunks_fts) VALUES('rebuild')")
                self._conn.commit()

    def list_references(self) -> list[ReferenceDoc]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM reference_docs ORDER BY added_at").fetchall()
        docs = []
        for r in rows:
            content = Path(r["file_path"]).read_text(encoding="utf-8")
            docs.append(
                ReferenceDoc(
                    doc_id=r["doc_id"], title=r["title"], source=r["source"],
                    added_at=datetime.fromisoformat(r["added_at"]), content=content,
                )
            )
        return docs

    def search_references(self, query: str, top_k: int = 3) -> list[ReferenceChunk]:
        terms = [t.lower() for t in _WORD_RE.findall(query) if len(t) > 2]
        if not terms:
            return []

        with self._lock:
            if self._fts_available:
                fts_query = " OR ".join(terms)
                try:
                    rows = self._conn.execute(
                        """SELECT rc.doc_id, rc.chunk_index, rc.title, rc.text, bm25(reference_chunks_fts) AS rank
                           FROM reference_chunks_fts
                           JOIN reference_chunks rc ON rc.rowid = reference_chunks_fts.rowid
                           WHERE reference_chunks_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (fts_query, top_k),
                    ).fetchall()
                    if rows:
                        return [
                            ReferenceChunk(doc_id=r["doc_id"], chunk_index=r["chunk_index"], text=r["text"],
                                            title=r["title"], score=-r["rank"])
                            for r in rows
                        ]
                except sqlite3.OperationalError:
                    pass

            # Fallback: naive term-frequency keyword scoring (no FTS5 available).
            rows = self._conn.execute("SELECT * FROM reference_chunks").fetchall()

        scored = []
        for r in rows:
            text_lower = r["text"].lower()
            score = sum(text_lower.count(t) for t in terms)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ReferenceChunk(doc_id=r["doc_id"], chunk_index=r["chunk_index"], text=r["text"],
                            title=r["title"], score=float(score))
            for score, r in scored[:top_k]
        ]

    # --- health flags (§5.8) ---------------------------------------------------------

    def save_health_flag(self, flag: HealthFlag) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO health_flags (timestamp, flag, note) VALUES (?, ?, ?)",
                (flag.timestamp.isoformat(), flag.flag, flag.note),
            )
            self._conn.commit()

    def get_recent_health_flags(self, days: int = 30) -> list[HealthFlag]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM health_flags ORDER BY timestamp DESC"
            ).fetchall()
        cutoff = datetime.utcnow().timestamp() - days * 86400
        flags = []
        for r in rows:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.timestamp() >= cutoff:
                flags.append(HealthFlag(timestamp=ts, flag=r["flag"], note=r["note"]))
        return flags
