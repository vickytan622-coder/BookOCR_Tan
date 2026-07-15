"""Lightweight, local QA report for a completed OCR job."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .jobs import Job, JobStore, PAGE_COMPLETE


@dataclass(frozen=True)
class PageAudit:
    page: int
    characters: int
    flags: list[str]


def audit_job(store: JobStore, job: Job) -> Path:
    """Flag pages that deserve human attention; never alter OCR output."""
    pages_dir = job.output_dir / "pages"
    pages: list[PageAudit] = []
    for number in range(1, job.total_pages + 1):
        md_files = sorted((pages_dir / f"page_{number:06d}").glob("*.md"))
        text = md_files[0].read_text(encoding="utf-8").strip() if md_files else ""
        flags: list[str] = []
        if not md_files:
            flags.append("missing_markdown")
        elif len(text) < 80:
            flags.append("very_short")
        if "<!-- Page" in text:
            flags.append("unexpected_page_marker")
        pages.append(PageAudit(page=number, characters=len(text), flags=flags))

    report = {
        "job_id": job.id,
        "source_pdf": str(job.source_pdf),
        "total_pages": job.total_pages,
        "completed_pages": store.summary(job.id).get(PAGE_COMPLETE, 0),
        "pages_needing_review": [asdict(page) for page in pages if page.flags],
        "pages": [asdict(page) for page in pages],
    }
    destination = job.output_dir / "qa_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
