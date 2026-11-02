from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.api import chat
from app.main import app
from app.qa.answerer import IndexUnavailableError
from app.schemas.chat import ChatRequest, ChatResponse, Citation

client = TestClient(app)


def test_chat_request_validates_scope_and_blank_question() -> None:
    assert ChatRequest(question="hello", scope="general").scope == "general"
    with pytest.raises(ValidationError):
        ChatRequest(question="hello", scope="unknown")
    with pytest.raises(ValidationError):
        ChatRequest(question="   ")


def test_chat_response_and_citations_come_from_service(monkeypatch) -> None:
    expected = ChatResponse(
        answer="Answer [C1].",
        citations=[Citation(page_number=8, section="2.1 Eligibility", content_snippet="source")],
        scope_used="general",
    )
    monkeypatch.setattr(chat, "answer_question", lambda _: expected)
    response = client.post("/api/chat", json={"question": "age?", "scope": "general"})
    assert response.status_code == 200
    assert response.json() == expected.model_dump()


def test_empty_index_returns_service_unavailable(monkeypatch) -> None:
    def fail(_):
        raise IndexUnavailableError("Index empty")

    monkeypatch.setattr(chat, "answer_question", fail)
    response = client.post("/api/chat", json={"question": "age?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Index empty"


def test_frontend_and_assets_are_served() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert "Ask the rulebook" in home.text
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "textContent" in script.text
    assert "innerHTML" not in script.text
