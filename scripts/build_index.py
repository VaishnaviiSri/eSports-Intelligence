"""Build the persistent Chroma index from Phase 2 JSONL chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.embedder import embed_texts
from app.rag.vector_store import ALLOWED_CONTENT_TYPES, ALLOWED_SCOPES, get_collection, upsert_chunks

INPUT_PATH = Path("data/processed/chunks.jsonl")
BATCH_SIZE = 32


def _render_content(record: dict[str, Any]) -> str:
    """Render text or table rows as a labeled string for embedding."""
    content = record.get("content")
    content_type = record.get("content_type")
    if content_type == "table" and isinstance(content, list):
        return "Table:\n" + "\n".join(
            " | ".join("" if cell is None else str(cell).strip() for cell in row)
            for row in content
            if isinstance(row, list)
        )
    if isinstance(content, str):
        return content.strip()
    return ""


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten useful source and citation metadata for Chroma."""
    source = record.get("metadata") or {}
    return {
        "content_type": record.get("content_type"),
        "doc_scope": record.get("doc_scope"),
        "page_number": record.get("page_number"),
        "source_file": source.get("source_file"),
        "heading": source.get("heading"),
        "chunk_id": record.get("chunk_id"),
        "logical_block_id": source.get("logical_block_id"),
        "document_id": source.get("document_id"),
        "tournament": source.get("tournament"),
        "version": source.get("version"),
        "document_date": source.get("document_date"),
    }


def read_eligible_records(path: Path = INPUT_PATH) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read JSONL and count eligible, excluded, and malformed records."""
    eligible: list[dict[str, Any]] = []
    stats = {"read": 0, "eligible": 0, "excluded": 0, "invalid": 0}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            stats["read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            content_type = record.get("content_type")
            scope = record.get("doc_scope")
            if content_type not in ALLOWED_CONTENT_TYPES or scope not in ALLOWED_SCOPES:
                stats["excluded"] += 1
                continue
            if not isinstance(record.get("chunk_id"), str) or not record["chunk_id"]:
                stats["invalid"] += 1
                continue
            if not isinstance(record.get("page_number"), int):
                stats["invalid"] += 1
                continue
            content = _render_content(record)
            if not content:
                stats["invalid"] += 1
                continue
            record["_embedding_document"] = content
            eligible.append(record)
            stats["eligible"] += 1
    return eligible, stats


def build_index(path: Path = INPUT_PATH, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Embed eligible chunks and upsert them to the persistent collection."""
    records, stats = read_eligible_records(path)
    stats["failed"] = 0
    stats["indexed"] = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            documents = [record["_embedding_document"] for record in batch]
            vectors = embed_texts(documents)
            upsert_chunks(
                [record["chunk_id"] for record in batch],
                documents,
                [_metadata(record) for record in batch],
                vectors,
            )
            stats["indexed"] += len(batch)
        except Exception:
            stats["failed"] += len(batch)
            raise
    stats["collection_total"] = get_collection().count()
    return stats


def main() -> int:
    """Build index and print a concise summary."""
    summary = build_index()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())