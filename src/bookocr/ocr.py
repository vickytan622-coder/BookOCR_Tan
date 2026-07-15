"""Page-level PaddleOCR-VL runner with resumable SQLite job state."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import fitz

from .jobs import Job, JobStore, PAGE_COMPLETE, PAGE_FAILED


@dataclass(frozen=True)
class OcrRunSummary:
    completed: int
    failed: int
    elapsed_seconds: float


class PaddleOcrRunner:
    """Initializes the complete document-parsing pipeline once per job run."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
    ) -> None:
        # PaddleX and ModelScope otherwise write to hidden home-directory caches.
        # A project-local cache makes first-run setup and later cleanup explicit.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir / "paddlex"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        (cache_dir / "paddlex").mkdir(parents=True, exist_ok=True)
        (cache_dir / "modelscope").mkdir(parents=True, exist_ok=True)

        # Delay this import until cache locations are set.
        from paddleocr import PaddleOCRVL

        backend = vl_rec_backend or os.environ.get("BOOKOCR_VL_BACKEND")
        server_url = vl_rec_server_url or os.environ.get("BOOKOCR_VL_SERVER_URL")
        if bool(backend) != bool(server_url):
            raise ValueError("远程 VLM 加速需要同时提供 backend 和 server URL")

        # mlx-vlm caches models by the exact string received in the request.
        # Point the client at the same absolute local path used by our server,
        # otherwise it mistakes Paddle's display name for a Hugging Face repo.
        api_model_name = None
        if backend == "mlx-vlm-server":
            api_model_name = str((cache_dir / "paddlex" / "official_models" / "PaddleOCR-VL-1.6").resolve())

        self.pipeline = PaddleOCRVL(
            device="cpu",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            vl_rec_backend=backend,
            vl_rec_server_url=server_url,
            vl_rec_api_model_name=api_model_name,
        )

    def process_page(self, image_path: Path, page_output: Path) -> list[object]:
        page_output.mkdir(parents=True, exist_ok=True)
        results = list(self.pipeline.predict(str(image_path)))
        if not results:
            raise RuntimeError("PaddleOCR-VL returned no page result")
        for result in results:
            result.save_to_json(save_path=page_output)
            result.save_to_markdown(save_path=page_output)
        return results

    def restructure_batch(self, page_results: list[object], destination: Path) -> None:
        """Apply PaddleOCR-VL's page-aware post-processing without re-OCR."""
        if not page_results:
            return
        destination.mkdir(parents=True, exist_ok=True)
        results = self.pipeline.restructure_pages(
            page_results,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=True,
        )
        for result in results:
            result.save_to_json(save_path=destination)
            result.save_to_markdown(save_path=destination)


def render_page(pdf: fitz.Document, page_number: int, destination: Path, dpi: int = 300) -> Path:
    """Render a source page at OCR-friendly resolution without splitting the PDF."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    page = pdf.load_page(page_number - 1)
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixmap.save(str(destination))
    return destination


def run_ocr_job(
    store: JobStore,
    job: Job,
    *,
    cache_dir: Path,
    max_pages: int | None = None,
    vl_rec_backend: str | None = None,
    vl_rec_server_url: str | None = None,
    retry_failed: bool = False,
) -> OcrRunSummary:
    """Run pending pages, saving each result before moving to the next page.

    ``max_pages`` supports a safe first-run verification on representative
    pages before a user commits a long scanned book to OCR.
    """
    started = time.perf_counter()
    recovered = store.recover_interrupted_pages(job.id)
    if recovered:
        print(f"Recovered {recovered} interrupted page(s).")
    if retry_failed:
        reset = store.reset_failed_pages(job.id)
        if reset:
            print(f"Reset {reset} failed page(s) for retry.")

    runner = PaddleOcrRunner(
        cache_dir,
        vl_rec_backend=vl_rec_backend,
        vl_rec_server_url=vl_rec_server_url,
    )
    completed = 0
    failed = 0
    rendered_dir = job.output_dir / "rendered"
    pages_dir = job.output_dir / "pages"

    with fitz.open(str(job.source_pdf)) as pdf:
        while max_pages is None or completed + failed < max_pages:
            limit = job.batch_size if max_pages is None else min(
                job.batch_size, max_pages - completed - failed
            )
            pages = store.claim_pending_pages(job.id, limit)
            if not pages:
                break
            batch_results: list[object] = []
            for page_number in pages:
                page_dir = pages_dir / f"page_{page_number:06d}"
                image_path = rendered_dir / f"page_{page_number:06d}.png"
                try:
                    render_page(pdf, page_number, image_path)
                    batch_results.extend(runner.process_page(image_path, page_dir))
                    store.set_page_status(job.id, page_number, PAGE_COMPLETE)
                    completed += 1
                    print(f"Completed page {page_number}/{job.total_pages}")
                except Exception as exc:  # Keep one bad scan from killing a book.
                    store.set_page_status(job.id, page_number, PAGE_FAILED, str(exc))
                    failed += 1
                    print(f"Failed page {page_number}/{job.total_pages}: {exc}")
            # This only consumes cached in-memory page results; it never calls
            # OCR again for the overlap used to join pages in the batch.
            if batch_results:
                batch_label = f"pages_{pages[0]:06d}_{pages[-1]:06d}"
                runner.restructure_batch(batch_results, job.output_dir / "restructured" / batch_label)

    return OcrRunSummary(
        completed=completed,
        failed=failed,
        elapsed_seconds=time.perf_counter() - started,
    )
