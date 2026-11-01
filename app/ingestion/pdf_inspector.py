"""Read-only inspection of PDF files for ingestion planning."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

LOW_TEXT_WARNING = (
    "Low extractable text may indicate scanned pages, but can also result from "
    "unusual PDF encoding, image-only layouts, or text stored in vector graphics. "
    "Verify visually before assuming OCR is required."
)
IMAGE_ONLY_WARNING = (
    "Document appears to contain image-only pages with no extractable text. "
    "OCR will likely be required for retrieval."
)


@dataclass
class InspectionConfig:
    """Thresholds and bounds used while inspecting PDF files."""

    scan_threshold: int = 50
    low_text_threshold: int = 100
    max_sample_chars: int = 500
    max_pages: int | None = None


def classify_page(
    text: str | None,
    image_count: int,
    config: InspectionConfig,
) -> str:
    """Classify one page using its extractable character and image counts."""
    char_count = len(text or "")
    if char_count == 0 and image_count > 0:
        return "potentially_scanned"
    if char_count == 0 and image_count == 0:
        return "empty"
    if 0 < char_count < config.scan_threshold:
        return "potentially_scanned"
    if config.scan_threshold <= char_count < config.low_text_threshold:
        return "lowtext"
    return "text_based"


def classify_document(page_classes: list[str], total_chars: int) -> str:
    """Classify a PDF from its page classifications and extracted text volume."""
    if "inspection_error" in page_classes:
        return "inspection_error"
    if "encrypted" in page_classes:
        return "encrypted"
    if page_classes and all(label == "empty" for label in page_classes):
        return "empty_or_unreadable"
    if not page_classes:
        return "empty_or_unreadable"

    page_count = len(page_classes)
    text_pages = sum(label == "text_based" for label in page_classes)
    if text_pages / page_count >= 0.8:
        return "text_based"

    scanned_or_empty_pages = sum(
        label in {"potentially_scanned", "empty"} for label in page_classes
    )
    if scanned_or_empty_pages / page_count >= 0.8:
        return "potentially_scanned"
    if total_chars == 0:
        return "empty_or_unreadable"
    return "mixed"


def inspect_page(page: Any, page_number: int, config: InspectionConfig) -> dict[str, Any]:
    """Extract bounded text and image information from one page."""
    record: dict[str, Any] = {
        "page_number": page_number,
        "char_count": 0,
        "word_count": 0,
        "image_count": 0,
        "has_large_image": False,
        "classification": "inspection_error",
        "sample_text": "",
    }
    try:
        text = page.get_text() or ""
        image_info = page.get_image_info()
        page_area = float(page.rect.width) * float(page.rect.height)
        large_image = any(
            page_area > 0
            and max(0.0, float(image["bbox"][2]) - float(image["bbox"][0]))
            * max(0.0, float(image["bbox"][3]) - float(image["bbox"][1]))
            > page_area * 0.5
            for image in image_info
        )
        record.update(
            {
                "char_count": len(text),
                "word_count": len(text.split()),
                "image_count": len(image_info),
                "has_large_image": large_image,
                "classification": classify_page(text, len(image_info), config),
                "sample_text": text[: config.max_sample_chars],
            }
        )
    except Exception as exc:
        logger.exception("Could not inspect page %s", page_number)
        record["error"] = str(exc)
    return record


def _base_document_record(path: Path) -> dict[str, Any]:
    """Build the stable fields shared by successful and failed inspections."""
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return {
        "relative_path": path.name,
        "size_bytes": size_bytes,
        "page_count": 0,
        "pages_inspected": 0,
        "encrypted": False,
        "pdf_metadata": {},
        "extraction": {
            "total_chars": 0,
            "total_words": 0,
            "avg_chars_per_page": 0.0,
        },
        "classification": "inspection_error",
        "warnings": [],
    }


def _page_error_record(page_number: int, error: Exception) -> dict[str, Any]:
    """Return a page record for a page that could not be loaded or inspected."""
    return {
        "page_number": page_number,
        "char_count": 0,
        "word_count": 0,
        "image_count": 0,
        "has_large_image": False,
        "classification": "inspection_error",
        "sample_text": "",
        "error": str(error),
    }


def inspect_pdf(path: Path, config: InspectionConfig) -> dict[str, Any]:
    """Inspect a PDF without modifying it; convert all document errors to records."""
    record = _base_document_record(Path(path))
    try:
        with pymupdf.open(path) as document:
            record["page_count"] = len(document)
            metadata = document.metadata
            record["pdf_metadata"] = metadata or {}

            if document.needs_pass:
                record["encrypted"] = True
                record["classification"] = "encrypted"
                return record

            page_limit = config.max_pages
            selected_pages = min(record["page_count"], page_limit) if page_limit is not None else record["page_count"]
            pages: list[dict[str, Any]] = []
            for page_index in range(selected_pages):
                try:
                    page = document.load_page(page_index)
                    pages.append(inspect_page(page, page_index + 1, config))
                except Exception as exc:
                    logger.exception("Could not load page %s from %s", page_index + 1, path)
                    pages.append(_page_error_record(page_index + 1, exc))

            total_chars = sum(page["char_count"] for page in pages)
            total_words = sum(page["word_count"] for page in pages)
            record["pages_inspected"] = len(pages)
            record["pages"] = pages
            record["extraction"] = {
                "total_chars": total_chars,
                "total_words": total_words,
                "avg_chars_per_page": total_chars / len(pages) if pages else 0.0,
            }
            record["classification"] = classify_document(
                [page["classification"] for page in pages], total_chars
            )

            if page_limit is not None and record["pages_inspected"] < record["page_count"]:
                record["warnings"].append(
                    f"Inspection limited to {record['pages_inspected']} of "
                    f"{record['page_count']} pages by max_pages."
                )

            if record["classification"] == "potentially_scanned":
                record["warnings"].append(IMAGE_ONLY_WARNING)
            if record["classification"] in {"potentially_scanned", "mixed"}:
                record["warnings"].append(LOW_TEXT_WARNING)
    except Exception as exc:
        logger.exception("Could not inspect PDF %s", path)
        record.update(
            {
                "classification": "inspection_error",
                "error": str(exc),
            }
        )
    return record


def discover_pdfs(input_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Find PDF files in a directory tree, skipping and reporting symlinks."""
    pdfs: list[Path] = []
    warnings: list[dict[str, str]] = []
    for root, directories, filenames in os.walk(input_dir, followlinks=False):
        root_path = Path(root)
        kept_directories: list[str] = []
        for directory in sorted(directories):
            candidate = root_path / directory
            if candidate.is_symlink():
                warnings.append(
                    {"path": candidate.relative_to(input_dir).as_posix(), "error": "Symlink skipped."}
                )
            else:
                kept_directories.append(directory)
        directories[:] = kept_directories

        for filename in sorted(filenames):
            candidate = root_path / filename
            if candidate.is_symlink():
                warnings.append(
                    {"path": candidate.relative_to(input_dir).as_posix(), "error": "Symlink skipped."}
                )
            elif candidate.suffix.lower() == ".pdf":
                pdfs.append(candidate)

    return sorted(pdfs, key=lambda item: item.relative_to(input_dir).as_posix()), sorted(
        warnings, key=lambda item: item["path"]
    )


def build_report(
    input_dir: Path,
    config: InspectionConfig,
    documents: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Assemble the versioned report in a deterministic key and document order."""
    documents.sort(key=lambda item: item["relative_path"])
    classification_counts = {
        label: 0
        for label in (
            "text_based",
            "potentially_scanned",
            "mixed",
            "empty_or_unreadable",
            "encrypted",
            "inspection_error",
        )
    }
    for document in documents:
        classification = document["classification"]
        classification_counts[classification] += 1
        document["pages"] = sorted(document.get("pages", []), key=lambda page: page["page_number"])

    return {
        "schema_version": "1.0",
        "input_dir": Path(input_dir).as_posix(),
        "config": asdict(config),
        "summary": {
            "total_pdfs": len(documents),
            "by_classification": classification_counts,
            "total_pages": sum(document["page_count"] for document in documents),
            "total_extracted_chars": sum(
                document["extraction"]["total_chars"] for document in documents
            ),
        },
        "documents": documents,
        "errors": sorted(errors, key=lambda item: (item.get("path", ""), item.get("error", ""))),
    }