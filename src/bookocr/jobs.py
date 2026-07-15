"""Persistent job and page state for resumable PDF processing."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader


PAGE_PENDING = "pending"
PAGE_RUNNING = "running"
PAGE_COMPLETE = "complete"
PAGE_FAILED = "failed"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Job:
    id: str
    source_pdf: Path
    output_dir: Path
    total_pages: int
    batch_size: int


class JobStore:
    """SQLite-backed job store. Every source page has an independent status."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    total_pages INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    job_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, page_number),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                );
                """
            )

    def create_job(self, source_pdf: Path, output_dir: Path, batch_size: int = 50) -> Job:
        source_pdf = source_pdf.expanduser().resolve()
        if not source_pdf.is_file():
            raise FileNotFoundError(source_pdf)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        total_pages = len(PdfReader(str(source_pdf)).pages)
        job_id = uuid.uuid4().hex[:12]
        output_root = output_dir.expanduser().resolve()
        job = Job(
            id=job_id,
            source_pdf=source_pdf,
            # Multiple books - or two test runs of the same book - must never
            # write page checkpoints into the same folder.
            output_dir=output_root / f"{source_pdf.stem}-{job_id}",
            total_pages=total_pages,
            batch_size=batch_size,
        )
        job.output_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
                (job.id, str(job.source_pdf), str(job.output_dir), job.total_pages, job.batch_size, now),
            )
            connection.executemany(
                "INSERT INTO pages (job_id, page_number, status, updated_at) VALUES (?, ?, ?, ?)",
                [(job.id, page, PAGE_PENDING, now) for page in range(1, job.total_pages + 1)],
            )
        return job

    def get_job(self, job_id: str) -> Job:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return Job(
            id=row["id"],
            source_pdf=Path(row["source_pdf"]),
            output_dir=Path(row["output_dir"]),
            total_pages=row["total_pages"],
            batch_size=row["batch_size"],
        )

    def latest_job(self) -> Job | None:
        """Return the most recently created job in this local workspace."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return Job(
            id=row["id"],
            source_pdf=Path(row["source_pdf"]),
            output_dir=Path(row["output_dir"]),
            total_pages=row["total_pages"],
            batch_size=row["batch_size"],
        )

    def claim_pending_pages(self, job_id: str, limit: int) -> list[int]:
        """Claim a batch. Interrupted `running` pages are recovered separately."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT page_number FROM pages WHERE job_id = ? AND status = ? ORDER BY page_number LIMIT ?",
                (job_id, PAGE_PENDING, limit),
            ).fetchall()
            pages = [row["page_number"] for row in rows]
            now = utc_now()
            connection.executemany(
                "UPDATE pages SET status = ?, attempts = attempts + 1, updated_at = ? WHERE job_id = ? AND page_number = ?",
                [(PAGE_RUNNING, now, job_id, page) for page in pages],
            )
        return pages

    def set_page_status(self, job_id: str, page_number: int, status: str, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pages SET status = ?, error = ?, updated_at = ? WHERE job_id = ? AND page_number = ?",
                (status, error, utc_now(), job_id, page_number),
            )

    def recover_interrupted_pages(self, job_id: str) -> int:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE pages SET status = ?, updated_at = ? WHERE job_id = ? AND status = ?",
                (PAGE_PENDING, utc_now(), job_id, PAGE_RUNNING),
            )
        return result.rowcount

    def reset_failed_pages(self, job_id: str) -> int:
        """Make failed pages eligible for an explicit retry."""
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE pages SET status = ?, error = NULL, updated_at = ? WHERE job_id = ? AND status = ?",
                (PAGE_PENDING, utc_now(), job_id, PAGE_FAILED),
            )
        return result.rowcount

    def summary(self, job_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM pages WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def retry_summary(self, job_id: str) -> dict[str, int]:
        """Summarize pages that have entered a second-or-later attempt."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM pages WHERE job_id = ? AND attempts > 1 GROUP BY status",
                (job_id,),
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}
