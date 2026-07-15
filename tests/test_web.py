import json
import time
from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from bookocr import web
from bookocr.web import create_app


def test_local_web_ui_and_doctor_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    homepage = client.get("/")
    assert homepage.status_code == 200
    assert "BookOCR" in homepage.text
    assert "保存模型配置" in homepage.text
    assert "停止 BookOCR 服务" in homepage.text
    assert "本次 OCR 页数" in homepage.text
    assert "创建成功后会自动填入任务 ID" in homepage.text
    assert "启动 Apple 加速" in homepage.text
    assert "ocr-progress" in homepage.text
    assert "兼容重试进度" in homepage.text
    assert "retrying_cpu" in homepage.text
    assert "assemble-status" in homepage.text
    assert "audit-progress" in homepage.text
    assert "未发现需要人工复核的页面" in homepage.text
    assert "查看技术详情（仅排障时需要）" in homepage.text
    assert "proofread-progress" in homepage.text
    assert "正在连接模型，请稍候" in homepage.text
    assert "校对已暂停" in homepage.text
    assert "网络波动，等待" in homepage.text
    assert "后自动重试" in homepage.text
    assert "正在连接模型，准备从第" in homepage.text
    assert "继续校对" in homepage.text
    assert "暂停校对" in homepage.text
    assert "bookocr.lastJobId" in homepage.text
    assert "/api/config/llm" in homepage.text

    doctor = client.get("/api/doctor")
    assert doctor.status_code == 200
    assert doctor.json()["workspace"] == str(tmp_path.resolve())
    assert client.get("/api/latest-job").json() == {}


def test_create_job_accepts_a_path_wrapped_in_quotes(tmp_path: Path) -> None:
    pdf_path = tmp_path / "带 空格.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/jobs",
        json={"pdf_path": f"'{pdf_path}'", "output_dir": "output/jobs"},
    )

    assert response.status_code == 200
    assert response.json()["total_pages"] == 1


def test_native_file_and_folder_choosers_return_local_paths(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.touch()
    client = TestClient(create_app(tmp_path))
    monkeypatch.setattr("bookocr.web._choose_with_macos", lambda script: str(pdf_path))

    assert client.post("/api/pick-pdf").json() == {"path": str(pdf_path)}
    assert client.post("/api/pick-output-dir").json() == {"path": str(pdf_path)}


def test_mlx_start_reports_when_service_is_already_running(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(create_app(tmp_path))
    monkeypatch.setattr(web, "_local_port_is_open", lambda port: True)

    response = client.post("/api/mlx/start")

    assert response.status_code == 200
    assert response.json()["status"] == "already_running"
    assert client.get("/api/mlx/status").json() == {"running": True}


def test_ocr_automatically_retries_failed_pages_with_cpu_fallback(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    job = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()
    calls: list[bool] = []

    def fake_run(store, stored_job, **kwargs) -> None:
        retry_failed = kwargs.get("retry_failed", False)
        calls.append(retry_failed)
        if retry_failed:
            store.reset_failed_pages(stored_job.id)
            page = store.claim_pending_pages(stored_job.id, 1)[0]
            store.set_page_status(stored_job.id, page, "complete")
        else:
            page = store.claim_pending_pages(stored_job.id, 1)[0]
            store.set_page_status(stored_job.id, page, "failed", "simulated MLX shape error")

    monkeypatch.setattr(web, "run_ocr_job", fake_run)
    assert client.post(f"/api/jobs/{job['id']}/run", json={"vl_backend": "mlx-vlm-server"}).status_code == 200
    for _ in range(50):
        if len(calls) == 2:
            break
        time.sleep(0.01)

    status = client.get(f"/api/jobs/{job['id']}").json()
    assert calls == [False, True]
    assert status["pages"] == {"complete": 1}
    assert status["ocr_status"]["status"] == "complete"


def test_proofread_status_survives_web_service_restart(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    created = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()
    job = web.JobStore(tmp_path / "output" / "bookocr.sqlite3").get_job(created["id"])
    progress = {
        "status": "failed",
        "page": 1,
        "total_pages": 1,
        "completed": 0,
        "skipped": 0,
        "error": "temporary SSL error",
    }
    (job.output_dir / "proofread_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    response = TestClient(create_app(tmp_path)).get(f"/api/jobs/{job.id}/proofread-status")

    assert response.status_code == 200
    assert response.json() == progress


def test_proofread_pause_endpoint_sets_paused_state(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    created = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()

    pause_event_captured: list = []

    def slow_proofread_job(store, job, profile, **kwargs) -> None:
        pause_event = kwargs.get("pause_event")
        pause_event_captured.append(pause_event)
        # Write a running state then wait so the pause endpoint can act.
        progress_path = job.output_dir / "proofread_progress.json"
        progress_path.write_text(
            json.dumps({
                "status": "running",
                "page": 1,
                "total_pages": 1,
                "completed": 0,
                "skipped": 0,
            }),
            encoding="utf-8",
        )
        for _ in range(200):
            if pause_event is not None and pause_event.is_set():
                from bookocr.llm import ProofreadPaused
                raise ProofreadPaused("用户已暂停校对")
            time.sleep(0.01)

    monkeypatch.setattr(web, "proofread_job", slow_proofread_job)

    assert client.post(f"/api/jobs/{created['id']}/proofread", json={}).status_code == 200
    # Wait briefly for the worker to enter its running state.
    time.sleep(0.05)

    pause_response = client.post(f"/api/jobs/{created['id']}/proofread-pause")
    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "paused"

    # Wait for the worker to observe the paused event and exit.
    for _ in range(100):
        status = client.get(f"/api/jobs/{created['id']}/proofread-status").json()
        if status.get("status") == "paused":
            break
        time.sleep(0.01)

    assert status["status"] == "paused"
    assert "用户已暂停" in status.get("error", "")


def test_proofread_resume_restarts_from_saved_checkpoint(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    created = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()
    job = web.JobStore(tmp_path / "output" / "bookocr.sqlite3").get_job(created["id"])

    # Simulate a previously paused job with one checkpoint saved.
    checkpoint_dir = job.output_dir / "proofread_pages"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "page_000001.md").write_text("<!-- Page 1 -->\n\nok", encoding="utf-8")
    (job.output_dir / "proofread_progress.json").write_text(
        json.dumps({
            "status": "paused",
            "page": 2,
            "total_pages": 3,
            "completed": 0,
            "skipped": 1,
        }),
        encoding="utf-8",
    )

    calls: list = []

    def fake_proofread_job(store, job, profile, **kwargs) -> None:
        calls.append({
            "pause_event": kwargs.get("pause_event") is not None,
            "saved_checkpoints": len(list((job.output_dir / "proofread_pages").glob("page_*.md"))),
        })
        (job.output_dir / "proofread_progress.json").write_text(
            json.dumps({
                "status": "complete",
                "page": 3,
                "total_pages": 3,
                "completed": 2,
                "skipped": 1,
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(web, "proofread_job", fake_proofread_job)

    response = client.post(f"/api/jobs/{job.id}/proofread-resume")
    assert response.status_code == 200
    assert response.json()["status"] == "proofread_started"

    for _ in range(100):
        if calls:
            break
        time.sleep(0.01)

    assert len(calls) == 1
    assert calls[0]["pause_event"] is True
    assert calls[0]["saved_checkpoints"] == 1


def test_proofread_waiting_network_state_contains_countdown(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    created = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()
    job = web.JobStore(tmp_path / "output" / "bookocr.sqlite3").get_job(created["id"])

    progress = {
        "status": "waiting_network",
        "page": 13,
        "total_pages": 276,
        "completed": 9,
        "skipped": 3,
        "retry_attempt": 2,
        "retry_in_seconds": 15,
        "next_retry_at": "2026-07-14T08:30:00+00:00",
        "error": "ConnectError",
    }
    (job.output_dir / "proofread_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    response = client.get(f"/api/jobs/{job.id}/proofread-status")
    assert response.status_code == 200
    data = response.json()
    # No worker is active, so the stale waiting_network state is surfaced as a
    # resumable failure, but the countdown fields remain available.
    assert data["status"] == "failed"
    assert data["retry_attempt"] == 2
    assert data["retry_in_seconds"] == 15
    assert data["next_retry_at"] == "2026-07-14T08:30:00+00:00"


def test_shutdown_endpoint_signals_process_to_stop(tmp_path: Path, monkeypatch) -> None:
    import os
    import signal

    client = TestClient(create_app(tmp_path))
    killed: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)

    response = client.post("/api/shutdown")
    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"

    # The endpoint starts a background thread that sleeps before killing.
    # Wait briefly for the thread to schedule the kill.
    for _ in range(50):
        if killed:
            break
        time.sleep(0.01)

    assert len(killed) == 1
    assert killed[0][0] == os.getpid()
    assert killed[0][1] == signal.SIGTERM


def test_stale_running_status_is_returned_as_failed_when_no_worker_is_active(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "book.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    client = TestClient(create_app(tmp_path))
    created = client.post("/api/jobs", json={"pdf_path": str(pdf_path)}).json()
    job = web.JobStore(tmp_path / "output" / "bookocr.sqlite3").get_job(created["id"])

    stale_progress = {
        "status": "running",
        "page": 50,
        "total_pages": 100,
        "completed": 49,
        "skipped": 0,
    }
    (job.output_dir / "proofread_progress.json").write_text(
        json.dumps(stale_progress), encoding="utf-8"
    )

    response = client.get(f"/api/jobs/{job.id}/proofread-status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "继续校对" in data["error"]

    # The file should also be updated so the UI stays consistent on refresh.
    updated = json.loads(
        (job.output_dir / "proofread_progress.json").read_text(encoding="utf-8")
    )
    assert updated["status"] == "failed"
