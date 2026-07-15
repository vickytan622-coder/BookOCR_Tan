from pathlib import Path

import fitz

from bookocr.jobs import JobStore, PAGE_COMPLETE, PAGE_PENDING
from bookocr.proofread import proofread_job
from bookocr.llm import LlmProfile
from bookocr.config import LlmSettingsStore


def make_pdf(path: Path, pages: int) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    document.save(path)


def test_create_job_and_page_state(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    make_pdf(source, 3)
    store = JobStore(tmp_path / "state.sqlite3")
    job = store.create_job(source, tmp_path / "output", batch_size=2)

    assert job.total_pages == 3
    assert job.output_dir.parent == (tmp_path / "output").resolve()
    assert job.id in job.output_dir.name
    assert store.summary(job.id) == {PAGE_PENDING: 3}

    first_batch = store.claim_pending_pages(job.id, 2)
    assert first_batch == [1, 2]
    store.set_page_status(job.id, 1, PAGE_COMPLETE)
    assert store.recover_interrupted_pages(job.id) == 1
    assert store.claim_pending_pages(job.id, 10) == [2, 3]
    assert store.latest_job() == job


def test_proofread_is_checkpointed_and_preserves_page_anchors(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    make_pdf(source, 2)
    store = JobStore(tmp_path / "state.sqlite3")
    job = store.create_job(source, tmp_path / "output")
    for page in (1, 2):
        store.claim_pending_pages(job.id, 1)
        store.set_page_status(job.id, page, PAGE_COMPLETE)
        page_dir = job.output_dir / "pages" / f"page_{page:06d}"
        page_dir.mkdir(parents=True)
        (page_dir / "result.md").write_text(f"第 {page} 页正文", encoding="utf-8")

    monkeypatch.setattr("bookocr.proofread.proofread_markdown", lambda text, profile, **kwargs: text)
    profile = LlmProfile("test", "https://example.invalid/v1", "fake", "secret")
    first = proofread_job(store, job, profile)
    second = proofread_job(store, job, profile)

    assert first.completed == 2
    assert second.skipped == 2
    output = first.output.read_text(encoding="utf-8")
    assert "<!-- Page 1 -->" in output
    assert "<!-- Page 2 -->" in output


def test_proofread_restores_omitted_anchor_but_not_conflicting_anchor(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    make_pdf(source, 1)
    store = JobStore(tmp_path / "state.sqlite3")
    job = store.create_job(source, tmp_path / "output")
    store.claim_pending_pages(job.id, 1)
    store.set_page_status(job.id, 1, PAGE_COMPLETE)
    page_dir = job.output_dir / "pages" / "page_000001"
    page_dir.mkdir(parents=True)
    (page_dir / "result.md").write_text("正文", encoding="utf-8")

    monkeypatch.setattr(
        "bookocr.proofread.proofread_markdown", lambda text, profile, **kwargs: "校对后的正文"
    )
    output = proofread_job(store, job, LlmProfile("test", "https://example.invalid", "fake", "key")).output
    assert output.read_text(encoding="utf-8").count("<!-- Page 1 -->") == 1


def test_saved_llm_url_and_model_can_be_resolved_without_saving_secret(tmp_path: Path, monkeypatch) -> None:
    store = LlmSettingsStore(tmp_path / "settings")
    monkeypatch.setattr(store, "_read_key", lambda: None)
    saved = store.save(
        base_url="https://api.example.com/",
        model="my-model",
        api_key=None,
        auto_proofread=True,
    )

    assert saved.base_url == "https://api.example.com"
    assert saved.auto_proofread is True
    assert store.resolve(base_url="", model="", api_key="one-time-key") == (
        "https://api.example.com",
        "my-model",
        "one-time-key",
    )
