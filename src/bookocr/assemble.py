"""Assemble page-level OCR Markdown into a traceable book-level draft."""

from __future__ import annotations

from pathlib import Path

from .jobs import Job, JobStore, PAGE_COMPLETE


def assemble_raw_markdown(store: JobStore, job: Job) -> Path:
    """Combine every completed page in source order, retaining page anchors."""
    summary = store.summary(job.id)
    if summary.get(PAGE_COMPLETE, 0) != job.total_pages:
        raise RuntimeError("OCR 尚未完成，不能生成整书 Markdown")

    pages_dir = job.output_dir / "pages"
    destination = job.output_dir / f"{job.source_pdf.stem}.raw.md"
    parts = [f"<!-- BookOCR source: {job.source_pdf.name} -->\n"]
    for page_number in range(1, job.total_pages + 1):
        page_dir = pages_dir / f"page_{page_number:06d}"
        files = sorted(page_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"缺少第 {page_number} 页 Markdown")
        text = files[0].read_text(encoding="utf-8").strip()
        parts.append(f"\n\n<!-- Page {page_number} -->\n\n{text}\n")
    destination.write_text("".join(parts).strip() + "\n", encoding="utf-8")
    return destination
