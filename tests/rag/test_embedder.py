from app.rag import embedder


def test_embed_texts_uses_normalized_cached_model(monkeypatch) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            assert texts == ["a", "b"]
            assert kwargs["normalize_embeddings"] is True
            return [[0.6, 0.8], [1.0, 0.0]]

    monkeypatch.setattr(embedder, "load_embedding_model", lambda: FakeModel())
    assert embedder.embed_texts(["a", "b"]) == [[0.6, 0.8], [1.0, 0.0]]
    assert embedder.embed_texts([]) == []


def test_embed_query_adds_bge_search_prefix(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            captured["kwargs"] = kwargs
            return [[0.6, 0.8]]

    monkeypatch.setattr(embedder, "load_embedding_model", lambda: FakeModel())
    assert embedder.embed_query("What is the minimum age?") == [0.6, 0.8]
    assert captured["texts"] == [
        "Represent this sentence for searching relevant passages: What is the minimum age?"
    ]
    assert captured["kwargs"]["normalize_embeddings"] is True
