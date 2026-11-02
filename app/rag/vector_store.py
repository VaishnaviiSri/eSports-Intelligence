"""Persistent Chroma collection management and mandatory metadata filters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings

ALLOWED_CONTENT_TYPES = ("text", "table", "formula_image")
ALLOWED_SCOPES = ("general", "event_specific")


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    """Return a cached persistent Chroma client for the configured path."""
    persist_path = Path(settings.chroma_persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_collection() -> Collection:
    """Get or create the persistent cosine-distance collection."""
    return get_client().get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_where_filter(scope: str) -> dict[str, Any]:
    """Build mandatory content allowlist and requested rule-scope filters."""
    if scope not in {"auto", *ALLOWED_SCOPES}:
        raise ValueError(f"Unsupported scope: {scope}")
    scope_filter: dict[str, Any]
    if scope == "auto":
        scope_filter = {"doc_scope": {"$in": list(ALLOWED_SCOPES)}}
    else:
        scope_filter = {"doc_scope": scope}
    return {
        "$and": [
            {"content_type": {"$in": list(ALLOWED_CONTENT_TYPES)}},
            scope_filter,
        ]
    }


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only Chroma-supported scalar metadata values."""
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def upsert_chunks(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    """Upsert one validated batch into the persistent collection."""
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise ValueError("IDs, documents, metadata, and embeddings must have equal lengths")
    if not ids:
        return
    if any(not identifier for identifier in ids):
        raise ValueError("Chunk IDs must not be empty")
    get_collection().upsert(
        ids=ids,
        documents=documents,
        metadatas=[_validate_metadata(metadata) for metadata in metadatas],
        embeddings=embeddings,
    )