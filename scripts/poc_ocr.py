"""Run a minimal, reproducible PaddleOCR-VL smoke test on one page image."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from paddleocr import PaddleOCRVL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/poc"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipeline = PaddleOCRVL(
        device="cpu",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    initialized = time.perf_counter()
    results = list(pipeline.predict(str(args.input)))
    completed = time.perf_counter()

    for result in results:
        result.save_to_json(save_path=args.output)
        result.save_to_markdown(save_path=args.output)

    print(f"initialization_seconds={initialized - started:.1f}")
    print(f"inference_seconds={completed - initialized:.1f}")
    print(f"total_seconds={completed - started:.1f}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
