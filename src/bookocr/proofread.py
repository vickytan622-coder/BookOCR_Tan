"""Checkpointed, conservative LLM proofreading for assembled OCR Markdown."""

from __future__ import annotations

import re
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx

from .assemble import assemble_raw_markdown
from .jobs import Job, JobStore
from .llm import LlmProfile, ProofreadPaused, RetryInfo, proofread_markdown

_PAGE_MARKER = re.compile(r"(?=<!-- Page \d+ -->)")


@dataclass(frozen=True)
class ProofreadSummary:
    completed: int
    skipped: int
    output: Path


def _write_progress(path: Path, **data: object) -> None:
    """Persist non-secret state so progress survives a UI/server restart."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _without_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```markdown") and text.endswith("```"):
        return text[len("```markdown") : -3].strip()
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    return text


def proofread_job(
    store: JobStore,
    job: Job,
    profile: LlmProfile,
    *,
    max_pages: int | None = None,
    pause_event: threading.Event | None = None,
) -> ProofreadSummary:
    """Proofread one source page at a time, allowing interruption/resume."""
    raw_path = assemble_raw_markdown(store, job)
    chunks = [chunk.strip() for chunk in _PAGE_MARKER.split(raw_path.read_text(encoding="utf-8")) if chunk.strip()]
    if chunks and chunks[0].startswith("<!-- BookOCR source:"):
        header, *chunks = chunks
    else:
        header = ""

    checkpoint_dir = job.output_dir / "proofread_pages"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job.output_dir / "proofread_progress.json"
    completed = skipped = 0
    polished: list[str] = [header] if header else []

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    with httpx.Client(http2=False, timeout=timeout) as client:
        for index, chunk in enumerate(chunks, start=1):
            if pause_event is not None and pause_event.is_set():
                _write_progress(
                    progress_path,
                    status="paused",
                    page=index,
                    total_pages=len(chunks),
                    completed=completed,
                    skipped=skipped,
                )
                raise ProofreadPaused("用户已暂停校对")

            if max_pages is not None and index > max_pages:
                polished.extend(chunks[index - 1 :])
                break

            checkpoint = checkpoint_dir / f"page_{index:06d}.md"
            if checkpoint.exists():
                corrected = checkpoint.read_text(encoding="utf-8").strip()
                skipped += 1
            else:
                def record_connection_retry(info: RetryInfo) -> None:
                    _write_progress(
                        progress_path,
                        status=info.status,
                        page=index,
                        total_pages=len(chunks),
                        completed=completed,
                        skipped=skipped,
                        retry_attempt=info.attempt,
                        retry_in_seconds=info.retry_in_seconds,
                        next_retry_at=info.next_retry_at,
                        error=str(info.exc),
                    )

                try:
                    corrected = _without_code_fence(
                        proofread_markdown(
                            chunk,
                            profile,
                            client=client,
                            pause_event=pause_event,
                            on_connection_retry=record_connection_retry,
                        )
                    )
                except ProofreadPaused:
                    _write_progress(
                        progress_path,
                        status="paused",
                        page=index,
                        total_pages=len(chunks),
                        completed=completed,
                        skipped=skipped,
                    )
                    raise
                except Exception as exc:
                    _write_progress(
                        progress_path,
                        status="failed",
                        page=index,
                        total_pages=len(chunks),
                        completed=completed,
                        skipped=skipped,
                        error=str(exc),
                    )
                    raise
                # Page anchors are application-owned metadata, not text to be
                # proofread. Providers sometimes drop or renumber the leading
                # anchor, so always rebuild the known correct anchor locally.
                # An empty reply is still unsafe and is rejected.
                if not corrected.strip():
                    raise RuntimeError(f"LLM 为第 {index} 页返回空内容，已拒绝写入")
                corrected = re.sub(r"^<!-- Page \d+ -->\s*", "", corrected).strip()
                corrected = f"<!-- Page {index} -->\n\n{corrected}"
                checkpoint.write_text(corrected + "\n", encoding="utf-8")
                completed += 1
            polished.append(corrected)
            _write_progress(
                progress_path,
                status="running",
                page=index,
                total_pages=len(chunks),
                completed=completed,
                skipped=skipped,
            )
            print(f"Proofread page {index}/{len(chunks)}")

    destination = job.output_dir / f"{job.source_pdf.stem}.proofread.md"
    destination.write_text("\n\n".join(polished).strip() + "\n", encoding="utf-8")
    _write_progress(
        progress_path,
        status="complete",
        page=len(chunks),
        total_pages=len(chunks),
        completed=completed,
        skipped=skipped,
        output=str(destination),
    )
    return ProofreadSummary(completed=completed, skipped=skipped, output=destination)
