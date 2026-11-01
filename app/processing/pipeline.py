"""End-to-end deterministic PDF-to-chunks processing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.processing.chunker import Chunk, ChunkingConfig, chunk_page, chunk_to_dict
from app.processing.cleaner import clean_page
from app.processing.extractor import extract_pdf_pages
from app.processing.metadata import (
    DocumentMetadata,
    build_chunk_metadata,
    extract_document_metadata,
)
from app.processing.structure import DocumentProfile, analyze_structure


@dataclass(frozen=True)
class ProcessingConfig:
    """Stable settings for chunking and JSONL output."""

    max_chars: int = 1200
    overlap_chars: int = 120

    @property
    def chunking(self) -> ChunkingConfig:
        """Return the chunker configuration represented by this config."""
        return ChunkingConfig(self.max_chars, self.overlap_chars)


@dataclass
class ProcessingResult:
    """Summary of one PDF processing run."""

    source_file: str
    page_count: int
    chunk_count: int
    table_count: int
    formula_count: int
    legal_form_count: int
    output_path: str
    metadata: DocumentMetadata
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready concise CLI summary."""
        return {
            "source_file": self.source_file,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "table_count": self.table_count,
            "formula_count": self.formula_count,
            "legal_form_count": self.legal_form_count,
            "output_path": self.output_path,
            "document_id": self.metadata.document_id,
            "version": self.metadata.version,
            "document_date": self.metadata.document_date,
            "effective_date": self.metadata.effective_date,
            "warnings": self.warnings,
        }


def serialize_chunks(chunks: list[Chunk]) -> str:
    """Serialize ordered chunks as compact, deterministic UTF-8 JSONL."""
    lines = [
        json.dumps(
            chunk_to_dict(chunk),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Create the destination directory and write deterministic JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_chunks(chunks), encoding="utf-8", newline="\n")


def process_pdf(
    pdf_path: Path,
    output_path: Path,
    input_dir: Path | None = None,
    profile: DocumentProfile | None = None,
    config: ProcessingConfig | None = None,
) -> ProcessingResult:
    """Process one PDF into deterministic, page-safe chunks and write JSONL."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    profile = profile or DocumentProfile()
    config = config or ProcessingConfig()
    if input_dir is None:
        input_dir = pdf_path.parent
    input_dir = Path(input_dir)
    try:
        source_file = pdf_path.relative_to(input_dir).as_posix()
    except ValueError:
        source_file = pdf_path.name

    pages = extract_pdf_pages(pdf_path, source_file)
    if not pages:
        raise ValueError(f"PDF contains no pages: {pdf_path}")
    structure = analyze_structure(pages, profile)
    metadata = extract_document_metadata(pages[0].raw_text, profile, source_file)
    chunks: list[Chunk] = []

    for page in pages:
        segments = clean_page(page, structure.regions_by_page[page.page_number], profile)
        chunks.extend(chunk_page(page, segments, config.chunking))

    page_indices: dict[int, int] = {}
    for chunk in chunks:
        chunk_index = page_indices.get(chunk.page_number, 0)
        chunk.metadata = build_chunk_metadata(chunk, metadata, source_file, chunk_index)
        page_indices[chunk.page_number] = chunk_index + 1

    chunks.sort(key=lambda chunk: (chunk.page_number, chunk.chunk_id))
    write_chunks(chunks, output_path)

    warnings = list(structure.warnings) + list(metadata.warnings)
    for page in pages:
        warnings.extend(page.warnings)
        for table in page.table_regions:
            if not table.verified:
                warnings.append(
                    f"Unverified {table.classification} candidate on page {page.page_number}, "
                    f"candidate {table.candidate_index}."
                )

    table_count = sum(chunk.content_type == "table" for chunk in chunks)
    formula_count = sum(chunk.content_type == "formula_image" for chunk in chunks)
    legal_form_count = sum(chunk.content_type == "legal_form" for chunk in chunks)
    return ProcessingResult(
        source_file=source_file,
        page_count=len(pages),
        chunk_count=len(chunks),
        table_count=table_count,
        formula_count=formula_count,
        legal_form_count=legal_form_count,
        output_path=output_path.as_posix(),
        metadata=metadata,
        warnings=sorted(set(warnings)),
    )