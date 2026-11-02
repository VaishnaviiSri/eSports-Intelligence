"""Lazy sentence-transformers embedding adapter."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str | None = None):
    """Load and cache the configured embedding model on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name or settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode document texts into normalized dense embeddings."""
    if not texts:
        return []
    vectors = load_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def embed_query(question: str) -> list[float]:
    """Encode a single user query with the same normalized model."""
    vectors = embed_texts([question])
    return vectors[0]