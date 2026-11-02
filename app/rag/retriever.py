"""Scope-filtered Chroma retrieval and result diversification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.rag.embedder import embed_query
from app.rag.vector_store import build_where_filter, get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved source passage with database metadata and distance."""

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float


def select_diverse_results(
    results: list[RetrievedChunk],
    k: int,
    max_per_page: int = 2,
) -> list[RetrievedChunk]:
    """Deduplicate source chunks and limit same-page repetition with backfill."""
    if k <= 0:
        return []
    unique: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    seen_blocks: set[str] = set()
    for result in results:
        block_id = result.metadata.get("logical_block_id")
        if (
            result.chunk_id in seen_ids
            or result.content in seen_content
            or (block_id and block_id in seen_blocks)
        ):
            continue
        seen_ids.add(result.chunk_id)
        seen_content.add(result.content)
        if block_id:
            seen_blocks.add(block_id)
        unique.append(result)

    selected: list[RetrievedChunk] = []
    page_counts: dict[int, int] = {}
    deferred: list[RetrievedChunk] = []
    for item in unique:
        page = int(item.metadata.get("page_number", 0))
        if page_counts.get(page, 0) >= max_per_page:
            deferred.append(item)
            continue
        selected.append(item)
        page_counts[page] = page_counts.get(page, 0) + 1
        if len(selected) == k:
            return selected
    for item in deferred:
        selected.append(item)
        if len(selected) == k:
            break
    return selected


def retrieve(question: str, scope: str = "auto", k: int | None = None) -> list[RetrievedChunk]:
    """Query Chroma using the required content/scope filter and distance bound."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    count = k or settings.retrieval_top_k
    query = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=max(count * 4, count),
        where=build_where_filter(scope),
        include=["documents", "metadatas", "distances"],
    )
    ids = (query.get("ids") or [[]])[0]
    documents = (query.get("documents") or [[]])[0]
    metadatas = (query.get("metadatas") or [[]])[0]
    distances = (query.get("distances") or [[]])[0]
    gathered: list[RetrievedChunk] = []
    for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        numeric_distance = float(distance)
        if numeric_distance > settings.retrieval_max_distance:
            continue
        if metadata.get("content_type") not in {"text", "table", "formula_image"}:
            continue
        if metadata.get("doc_scope") not in {"general", "event_specific"}:
            continue
        if scope in {"general", "event_specific"} and metadata.get("doc_scope") != scope:
            continue
        gathered.append(
            RetrievedChunk(str(chunk_id), str(content or ""), metadata or {}, numeric_distance)
        )
    gathered.sort(key=lambda result: (result.distance, result.chunk_id))
    return select_diverse_results(gathered, count)