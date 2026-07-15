"""Provider-neutral, conservative Markdown proofreading.

The user supplies the endpoint and key.  No vendor URL or credential is
embedded in the project; any OpenAI-compatible chat-completions gateway works.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx


SYSTEM_PROMPT = """你是扫描书籍 OCR 的保守校对员。只修正明显的 OCR 错字、断行和标点；
绝不概括、删节、补写、改写原意，绝不修改 Markdown 标题/表格结构。若不确定，保持原文。
只返回校对后的 Markdown，不要解释。"""


@dataclass(frozen=True)
class LlmProfile:
    name: str
    base_url: str
    model: str
    api_key: str

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class RetryInfo:
    status: str
    attempt: int
    retry_in_seconds: float
    next_retry_at: str
    exc: Exception


RetryCallback = Callable[[RetryInfo], None]


class ProofreadPaused(Exception):
    """Raised when the user pauses proofreading during a retry wait."""


DEFAULT_BACKOFF_DELAYS: Sequence[float] = (5.0, 15.0, 30.0, 60.0, 120.0)
DEFAULT_BACKOFF_CAP: float = 300.0
DEFAULT_MAX_TOTAL_RETRY_SECONDS: float = 1200.0


def _is_transient_error(exc: Exception) -> bool:
    """Return True for connection/transport errors that may recover.

    HTTP-level errors (e.g. 401/404) are never retried: the server already
    produced a response, so resending could double-bill the user.
    """
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ReadError,
            httpx.WriteError,
        ),
    ):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "unexpected_eof",
            "connection reset",
            "timed out",
            "network is unreachable",
            "name or service not known",
        )
    )


def proofread_markdown(
    markdown: str,
    profile: LlmProfile,
    *,
    client: httpx.Client,
    timeout_seconds: float = 300.0,
    max_total_retry_seconds: float = DEFAULT_MAX_TOTAL_RETRY_SECONDS,
    backoff_delays: Sequence[float] = DEFAULT_BACKOFF_DELAYS,
    backoff_cap: float = DEFAULT_BACKOFF_CAP,
    pause_event: threading.Event | None = None,
    on_connection_retry: RetryCallback | None = None,
) -> str:
    """Return a correction from a standard ``/chat/completions`` endpoint.

    The request is retried only when a connection/transport error occurs
    before a complete response is received. Once the server returns any
    response (including an HTTP error status) the request is never resent
    automatically, to avoid duplicate billed inference.
    """
    if not markdown.strip():
        return markdown

    body = {
        "model": profile.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
    }

    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)
    headers = {"Authorization": f"Bearer {profile.api_key}"}

    start_time = time.monotonic()
    last_exc: Exception | None = None
    attempt = 0

    while True:
        attempt += 1
        try:
            response = client.post(
                profile.chat_url,
                json=body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("LLM 返回不符合 OpenAI Chat Completions 格式") from exc
            try:
                return payload["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("LLM 返回不符合 OpenAI Chat Completions 格式") from exc
        except httpx.HTTPStatusError as exc:
            # Configuration errors fail immediately. We do not retry any
            # HTTP-level error because the server already processed the
            # request and may have billed for it.
            status = exc.response.status_code
            if status in {401, 403, 404, 422}:
                raise RuntimeError(
                    f"模型服务返回 {status}，请检查 Base URL、模型名或 API Key"
                ) from exc
            raise
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc

            elapsed = time.monotonic() - start_time
            if elapsed >= max_total_retry_seconds:
                raise last_exc

            if attempt <= len(backoff_delays):
                delay = backoff_delays[attempt - 1]
            else:
                delay = backoff_cap

            next_retry_at = datetime.now(timezone.utc).isoformat()
            if on_connection_retry is not None:
                on_connection_retry(
                    RetryInfo(
                        status="waiting_network",
                        attempt=attempt,
                        retry_in_seconds=delay,
                        next_retry_at=next_retry_at,
                        exc=exc,
                    )
                )

            # Sleep in short chunks so a user pause can interrupt quickly.
            slept = 0.0
            while slept < delay:
                if pause_event is not None and pause_event.is_set():
                    raise ProofreadPaused("用户已暂停校对") from exc
                chunk = min(0.5, delay - slept)
                time.sleep(chunk)
                slept += chunk
