"""User-owned LLM settings for the macOS local application.

Non-secret settings live in Application Support.  The API key stays in the
logged-in user's macOS Keychain, never in the project or SQLite job database.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


SERVICE_NAME = "BookOCR LLM API Key"
ACCOUNT_NAME = "default"


@dataclass(frozen=True)
class SavedLlmSettings:
    base_url: str = ""
    model: str = ""
    key_configured: bool = False
    auto_proofread: bool = False


class LlmSettingsStore:
    def __init__(self, config_dir: Path | None = None) -> None:
        default_dir = Path.home() / "Library" / "Application Support" / "BookOCR"
        self.config_dir = (config_dir or Path(os.environ.get("BOOKOCR_CONFIG_DIR", default_dir))).expanduser()
        self.config_path = self.config_dir / "settings.json"

    def load(self) -> SavedLlmSettings:
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raw = {}
        return SavedLlmSettings(
            base_url=str(raw.get("base_url", "")),
            model=str(raw.get("model", "")),
            key_configured=self._read_key() is not None,
            auto_proofread=bool(raw.get("auto_proofread", False)),
        )

    def save(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        auto_proofread: bool = False,
    ) -> SavedLlmSettings:
        base_url, model = base_url.strip().rstrip("/"), model.strip()
        if not base_url or not model:
            raise ValueError("请填写 Base URL 和模型名")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(
                {"base_url": base_url, "model": model, "auto_proofread": auto_proofread},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if api_key and api_key.strip():
            self._write_key(api_key.strip())
        return self.load()

    def resolve(self, *, base_url: str, model: str, api_key: str) -> tuple[str, str, str]:
        saved = self.load()
        resolved_base = base_url.strip() or saved.base_url
        resolved_model = model.strip() or saved.model
        resolved_key = api_key.strip() or self._read_key() or ""
        if not (resolved_base and resolved_model and resolved_key):
            raise ValueError("请先保存模型配置，或在本次校对中填写 Base URL、模型名和 API Key")
        return resolved_base, resolved_model, resolved_key

    def _read_key(self) -> str | None:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None

    def _write_key(self, api_key: str) -> None:
        # `security` is the native Keychain command.  The secret is not written
        # to our config file, database, terminal output, or application logs.
        result = subprocess.run(
            ["security", "add-generic-password", "-U", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME, "-w", api_key],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("无法写入 macOS 钥匙串，请检查钥匙串访问权限")
