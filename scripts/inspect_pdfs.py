"""Command-line entry point for the read-only PDF inspection utility."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.ingestion.pdf_inspector import (
    InspectionConfig,
    build_report,
    discover_pdfs,
    inspect_pdf,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PDF text and image characteristics.")
    parser.add_argument("--input-dir", default="data/raw/rulebooks", help="Source directory")
    parser.add_argument(
        "--output",
        default="data/processed/pdf_inspection_report.json",
        help="Report file path, or - for stdout",
    )
    parser.add_argument("--scan-threshold", type=int, default=50)
    parser.add_argument("--low-text-threshold", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect PDFs, write a deterministic JSON report, and print a summary."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    config = InspectionConfig(
        scan_threshold=args.scan_threshold,
        low_text_threshold=args.low_text_threshold,
        max_pages=args.max_pages,
    )
    pdf_paths, discovery_warnings = discover_pdfs(input_dir)
    documents = []
    errors = list(discovery_warnings)
    for path in pdf_paths:
        document = inspect_pdf(path, config)
        document["relative_path"] = path.relative_to(input_dir).as_posix()
        if document["classification"] == "inspection_error":
            errors.append({"path": document["relative_path"], "error": document["error"]})
        documents.append(document)

    report = build_report(input_dir, config, documents, errors)
    try:
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output == "-":
            print(serialized, end="")
            summary_stream = sys.stderr
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized, encoding="utf-8")
            summary_stream = sys.stdout
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 1

    print(f"Inspection completed at {datetime.now().astimezone().isoformat(timespec='seconds')}", file=summary_stream)
    for document in report["documents"]:
        warning_text = f" warnings={'; '.join(document['warnings'])}" if document["warnings"] else ""
        print(
            f"{document['relative_path']}: pages={document['pages_inspected']}/"
            f"{document['page_count']} chars={document['extraction']['total_chars']} "
            f"classification={document['classification']}{warning_text}",
            file=summary_stream,
        )
    for warning in discovery_warnings:
        print(f"{warning['path']}: {warning['error']}", file=summary_stream)
    print(
        f"PDFs: {report['summary']['total_pdfs']}; errors: {len(errors)}; "
        f"report: {args.output}",
        file=summary_stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())