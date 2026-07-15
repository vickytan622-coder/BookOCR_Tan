import io
import json
import threading
import time

import httpx
import pytest

from bookocr.llm import (
    LlmProfile,
    ProofreadPaused,
    proofread_markdown,
)


def _ok_response(content: str) -> httpx.Response:
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
    return httpx.Response(
        200,
        content=body,
        request=httpx.Request("POST", "https://example.invalid/chat/completions"),
    )


def test_transient_connect_error_is_retried(monkeypatch) -> None:
    outcomes = [httpx.ConnectError("connection refused"), _ok_response("修正后")]
    sleeps: list[float] = []
    retries: list[tuple[int, float]] = []

    def fake_post(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("httpx.Client.post", fake_post)
    monkeypatch.setattr("bookocr.llm.time.sleep", sleeps.append)

    result = proofread_markdown(
        "原文",
        LlmProfile("test", "https://example.invalid", "model", "key"),
        client=httpx.Client(),
        on_connection_retry=lambda info: retries.append((info.attempt, info.retry_in_seconds)),
    )

    assert result == "修正后"
    assert retries == [(1, 5.0)]
    # The implementation sleeps in short chunks to support user pause.
    assert sum(sleeps) == 5.0
    assert all(s == 0.5 for s in sleeps)


def test_unexpected_eof_error_is_retried_then_succeeds(monkeypatch) -> None:
    outcomes = [
        httpx.ReadError("SSL: UNEXPECTED_EOF_WHILE_READING"),
        httpx.ConnectError("unexpected_eof while reading"),
        _ok_response("ok"),
    ]
    sleeps: list[float] = []

    def fake_post(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("httpx.Client.post", fake_post)
    monkeypatch.setattr("bookocr.llm.time.sleep", sleeps.append)

    result = proofread_markdown(
        "原文",
        LlmProfile("test", "https://example.invalid", "model", "key"),
        client=httpx.Client(),
    )

    assert result == "ok"
    assert sum(sleeps) == 20.0  # 5 + 15


def test_backoff_sequence_caps_at_max(monkeypatch) -> None:
    outcomes = [httpx.ConnectError("connection refused")] * 7 + [_ok_response("ok")]
    sleeps: list[float] = []

    def fake_post(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("httpx.Client.post", fake_post)
    monkeypatch.setattr("bookocr.llm.time.sleep", sleeps.append)

    proofread_markdown(
        "原文",
        LlmProfile("test", "https://example.invalid", "model", "key"),
        client=httpx.Client(),
    )

    assert sum(sleeps) == sum([5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 300.0])


def test_401_fails_immediately_and_is_not_retried(monkeypatch) -> None:
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            content=b'{"error":"invalid key"}',
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
        )

    monkeypatch.setattr("httpx.Client.post", fake_post)

    with pytest.raises(RuntimeError, match="401"):
        proofread_markdown(
            "原文",
            LlmProfile("test", "https://example.invalid", "model", "key"),
            client=httpx.Client(),
        )

    assert calls == 1


def test_404_fails_immediately_and_is_not_retried(monkeypatch) -> None:
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(
            404,
            content=b'{"error":"model not found"}',
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
        )

    monkeypatch.setattr("httpx.Client.post", fake_post)

    with pytest.raises(RuntimeError, match="404"):
        proofread_markdown(
            "原文",
            LlmProfile("test", "https://example.invalid", "bad", "key"),
            client=httpx.Client(),
        )

    assert calls == 1


def test_malformed_json_after_200_is_not_retried(monkeypatch) -> None:
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
        )

    monkeypatch.setattr("httpx.Client.post", fake_post)

    with pytest.raises(RuntimeError):
        proofread_markdown(
            "原文",
            LlmProfile("test", "https://example.invalid", "model", "key"),
            client=httpx.Client(),
        )

    assert calls == 1


def test_budget_exhaustion_raises_last_error(monkeypatch) -> None:
    monkeypatch.setattr("bookocr.llm.time.sleep", lambda s: None)

    call_count = [0]

    def fake_monotonic() -> float:
        call_count[0] += 1
        # First call sets start_time, second call checks elapsed, third call
        # exceeds the retry budget.
        return 0.0 if call_count[0] < 3 else 1300.0

    monkeypatch.setattr("bookocr.llm.time.monotonic", fake_monotonic)

    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("persistent failure")

    monkeypatch.setattr("httpx.Client.post", fake_post)

    with pytest.raises(httpx.ConnectError, match="persistent failure"):
        proofread_markdown(
            "原文",
            LlmProfile("test", "https://example.invalid", "model", "key"),
            client=httpx.Client(),
            max_total_retry_seconds=20.0,
        )


def test_pause_event_aborts_retry_and_raises_proofread_paused(monkeypatch) -> None:
    pause_event = threading.Event()

    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    def fake_sleep(seconds):
        # Simulate the user pausing during the first retry wait.
        pause_event.set()

    monkeypatch.setattr("httpx.Client.post", fake_post)
    monkeypatch.setattr("bookocr.llm.time.sleep", fake_sleep)

    with pytest.raises(ProofreadPaused):
        proofread_markdown(
            "原文",
            LlmProfile("test", "https://example.invalid", "model", "key"),
            client=httpx.Client(),
            pause_event=pause_event,
        )


def test_retry_callback_receives_waiting_network_info(monkeypatch) -> None:
    outcomes = [httpx.ConnectError("connection refused"), _ok_response("ok")]
    infos: list = []

    def fake_post(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("httpx.Client.post", fake_post)
    monkeypatch.setattr("bookocr.llm.time.sleep", lambda s: None)

    proofread_markdown(
        "原文",
        LlmProfile("test", "https://example.invalid", "model", "key"),
        client=httpx.Client(),
        on_connection_retry=infos.append,
    )

    assert len(infos) == 1
    assert infos[0].status == "waiting_network"
    assert infos[0].attempt == 1
    assert infos[0].retry_in_seconds == 5.0
    assert infos[0].next_retry_at.endswith("+00:00")
