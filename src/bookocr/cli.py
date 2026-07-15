"""Small CLI used by the future local UI and by development diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .assemble import assemble_raw_markdown
from .audit import audit_job
from .jobs import JobStore
from .ocr import run_ocr_job
from .llm import LlmProfile
from .proofread import proofread_job
from .config import LlmSettingsStore


def default_database() -> Path:
    return Path.cwd() / "output" / "bookocr.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(prog="bookocr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")

    create = subparsers.add_parser("create-job")
    create.add_argument("pdf", type=Path)
    create.add_argument("--output", type=Path, default=Path("output/jobs"))
    create.add_argument("--batch-size", type=int, default=50)

    status = subparsers.add_parser("status")
    status.add_argument("job_id")

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("job_id")

    audit = subparsers.add_parser("audit")
    audit.add_argument("job_id")

    proofread = subparsers.add_parser("proofread")
    proofread.add_argument("job_id")
    proofread.add_argument("--base-url", default="", help="留空时使用已保存的本机配置")
    proofread.add_argument("--model", default="", help="留空时使用已保存的本机配置")
    proofread.add_argument("--api-key", default="", help="留空时从 macOS 钥匙串读取")
    proofread.add_argument("--max-pages", type=int)

    run = subparsers.add_parser("run-ocr")
    run.add_argument("job_id")
    run.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    run.add_argument("--max-pages", type=int, help="仅处理指定数量的待处理页，用于试跑")
    run.add_argument("--vl-backend", help="可选 VLM 服务后端，例如 mlx-vlm-server")
    run.add_argument("--vl-server-url", help="可选本地 VLM 服务地址，例如 http://127.0.0.1:8118/v1")
    run.add_argument("--retry-failed", action="store_true", help="将失败页重置后重试")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.command == "doctor":
        print(json.dumps({"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()}, ensure_ascii=False, indent=2))
        return

    store = JobStore(default_database())
    if args.command == "create-job":
        job = store.create_job(args.pdf, args.output, args.batch_size)
        print(json.dumps(job.__dict__, ensure_ascii=False, indent=2, default=str))
    elif args.command == "status":
        job = store.get_job(args.job_id)
        print(json.dumps({"job": job.__dict__, "pages": store.summary(job.id)}, ensure_ascii=False, indent=2, default=str))
    elif args.command == "assemble":
        job = store.get_job(args.job_id)
        output = assemble_raw_markdown(store, job)
        print(str(output))
    elif args.command == "audit":
        job = store.get_job(args.job_id)
        print(str(audit_job(store, job)))
    elif args.command == "proofread":
        job = store.get_job(args.job_id)
        try:
            base_url, model, api_key = LlmSettingsStore().resolve(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )
        except ValueError as exc:
            parser.error(str(exc))
        summary = proofread_job(
            store,
            job,
            LlmProfile("custom", base_url, model, api_key),
            max_pages=args.max_pages,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, default=str))
    elif args.command == "run-ocr":
        if args.max_pages is not None and args.max_pages < 1:
            parser.error("--max-pages 必须至少为 1")
        job = store.get_job(args.job_id)
        summary = run_ocr_job(
            store,
            job,
            cache_dir=args.cache_dir.resolve(),
            max_pages=args.max_pages,
            vl_rec_backend=args.vl_backend,
            vl_rec_server_url=args.vl_server_url,
            retry_failed=args.retry_failed,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        from .web import serve

        serve(Path.cwd(), port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
