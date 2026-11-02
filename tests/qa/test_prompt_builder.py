from app.qa.prompt_builder import SYSTEM_PROMPT_TEMPLATE, build_messages
from app.rag.retriever import RetrievedChunk


def test_prompt_contains_grounding_scope_and_json_requirements() -> None:
    messages = build_messages("question", "general", [])
    system = messages[0]["content"]
    assert '"I don\'t know based on the provided rulebook."' in system
    assert "using only the supplied rulebook context" in system
    assert "General Rules" in system
    assert '{"answer": "...", "citation_ids": ["C1"]}' in system
    assert "invent citation IDs" in system
    assert "Auto scope" in SYSTEM_PROMPT_TEMPLATE


def test_prompt_labels_passages_with_trusted_source_metadata() -> None:
    chunk = RetrievedChunk(
        "id", "Age is 16.",
        {"page_number": 8, "heading": "2.1 Eligibility", "doc_scope": "general"},
        0.1,
    )
    messages = build_messages("age?", "general", [chunk])
    assert '"citation_id": "C1"' in messages[1]["content"]
    assert '"page_number": 8' in messages[1]["content"]
    assert "2.1 Eligibility" in messages[1]["content"]
