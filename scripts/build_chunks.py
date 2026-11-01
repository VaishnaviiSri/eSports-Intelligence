"""Build deterministic, page-safe chunks from a source PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from app.processing.pipeline import ProcessingConfig, process_pdf
from app.processing.structure import DocumentProfile


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured chunks from a rulebook PDF.")
    parser.add_argument(
        "--input",
        default="data/raw/rulebooks/bgmi_bmic_2026_rules.pdf",
        help="Source PDF path",
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/rulebooks",
        help="Root used to make source metadata paths relative",
    )
    parser.add_argument("--output", default="data/processed/chunks.jsonl", help="JSONL output path")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the processing pipeline and print a concise summary."""
    args = _parse_args(argv)
    try:
        result = process_pdf(
            Path(args.input),
            Path(args.output),
            input_dir=Path(args.input_dir),
            profile=DocumentProfile(),
            config=ProcessingConfig(args.max_chars, args.overlap_chars),
        )
    except Exception as exc:
        print(f"Chunk build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.summary(), ensure_ascii=False, sort_keys=True))
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())